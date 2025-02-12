import os
import argparse
from collections import defaultdict
from pathlib import Path
import cv2
import torch
import numpy as np
from shapely import Polygon
from ultralytics.utils.metrics import OBBMetrics, batch_probiou, ConfusionMatrix
from fusion.detector import TrunkDetector, poly2xywhr, TRUNK
from fusion.formats import DebugImage


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='run trunk detection and apply yolo metrics to evaluate oriented bounding boxes of fused results')
    parser.add_argument('--exp_dir', type=str, required=True,
                        help='directory containing images and ground-truth annotations '
                             'in yolo instance-segmentation format')
    parser.add_argument('--results_dir', type=str, default=None,
                        help='results directory for optional output visualizations')
    parser.add_argument('--image_list', type=str, default=None,
                        help='path to a text file containing a list of included image names in separate lines '
                        '(e.g. files containing train/val/test lists in yolo format)')
    parser.add_argument('--obb_model', type=str, default=None,
                        help='path to oriented-object-detection model (loads default model if not specified)')
    parser.add_argument('--seg_model', type=str, default=None,
                        help='path to instance-segmentation model (loads default model if not specified)')
    parser.add_argument('--min_confidence', type=float, default=0.4,
                        help='minimum confidence for results of detection and segmentation models')
    parser.add_argument('--classes', type=int, nargs='*', default=[],
                        help='list of class ids to be included in evaluation (all if empty)')

    return parser.parse_args()


def eval_fusion(exp_dir, results_dir, image_list, obb_model_path, seg_model_path, min_confidence, classes):
    # get ground truth from yolo annotation directory
    gt_dir = os.path.join(exp_dir, 'labels')

    # initialize detector
    print('initializing detector...')
    detector = TrunkDetector(obb_model_path, seg_model_path)
    n_classes = len(detector.get_class_names())
    train_size = detector.get_train_size()

    # initialize debug output
    formatter = None if results_dir is None else DebugImage()
    confusion_matrix = None if results_dir is None else ConfusionMatrix(nc=n_classes, conf=min_confidence)
    if results_dir is not None:
        os.makedirs(os.path.join(results_dir, 'det'), exist_ok=True)

    plots = {}

    def on_plot(name, data=None):
        plots[name] = data

    def included(label):
        return len(classes) == 0 or label in classes

    iouv = np.linspace(0.5, 0.95, 10)  # IoU vector for mAP @0.5:0.95
    stats = defaultdict(list)
    n_images = defaultdict(int)

    # iterate included images
    for image_path in [os.path.join('images', p)
                       for p in os.listdir(os.path.join(exp_dir, 'images'))] if image_list is None else image_list:
        image_id = os.path.splitext(os.path.split(image_path)[-1])[0]
        print(f'processing image {image_id}...')

        # read image and scale to training size
        image = cv2.imread(os.path.join(exp_dir, image_path))
        if train_size is not None:
            scale = train_size / max(image.shape[:2])
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        # apply detector and extract relevant results
        detections = detector.process_image(image, min_confidence)
        pred = []
        conf = []
        labels = []
        for det in detections.values():
            # only use instances detected by both models (obb and iseg)
            for label, component in det.components.items():
                if included(label) and component.match_score is not None:
                    pred.append(np.array(poly2xywhr(component.obb.contour), dtype=float))
                    conf.append(component.obb.score)
                    labels.append(label)
            # derive obb for entire trunk instance from components if included and not explicitly detected
            if included(TRUNK) and not det.has_score(TRUNK) and det.has_contour() and det.has_detected_obb():
                pred.append(np.array(det.xywhr(), dtype=float))
                conf.append(det.confidence())
                labels.append(TRUNK)
        pred = np.array(pred, dtype=float)
        conf = np.array(conf, dtype=float)
        labels = np.array(labels, dtype=int)

        vis = None if formatter is None else formatter.generate(image, detections)

        # count images with detections
        for c in list(np.unique(labels)) + ['all']:
            n_images[c] += 1

        # read ground truth for image in iseg format and convert to obbs
        gt_path = os.path.join(gt_dir, f'{image_id}.txt')
        if not os.path.exists(gt_path):
            raise ValueError(f'no ground truth for image id {image_id}')
        gt = []
        gt_labels = []
        with open(gt_path, 'r') as gt_file:
            for line in gt_file.readlines():
                anno = [a.strip().split(' ') for a in line.split(':')]
                label = int(anno[0].pop(0))
                if included(label):
                    gt_labels.append(label)
                    contours = [[(float(x) * image.shape[1], float(y) * image.shape[0])
                                 for x, y in zip(a[0::2], a[1::2])] for a in anno]
                    obb = cv2.boxPoints(cv2.minAreaRect(np.array(sum(contours, []), dtype=np.float32)))
                    gt.append(np.array(poly2xywhr(Polygon([(pt[0], pt[1]) for pt in obb])), dtype=float))

                    # visualize ground truth
                    if vis is not None:
                        cv2.polylines(vis, [obb.astype(int)], True, (0, 0, 255), 3, cv2.LINE_AA)
                        cv2.polylines(vis, [np.array([(c[0] + image.shape[1], c[1]) for c in contour], dtype=int)
                                            for contour in contours], True, (0, 0, 255), 3, cv2.LINE_AA)
        gt = np.array(gt)
        gt_labels = np.array(gt_labels, dtype=int)

        if vis is not None:
            cv2.imwrite(os.path.join(results_dir, 'det', f'{image_id}.png'), vis)

        if len(pred) > 0 or len(gt) > 0:
            tp = np.zeros((len(pred), iouv.shape[0])).astype(bool)
            if len(pred) == 0:
                if len(gt) > 0 and confusion_matrix is not None:
                    confusion_matrix.process_batch(detections=None, gt_bboxes=gt, gt_cls=torch.tensor(gt_labels))

            elif len(gt) > 0:
                # compute iou matrix and filter by matching classes
                iou = batch_probiou(gt, pred).numpy()
                correct_class = gt_labels.reshape(-1, 1) == labels
                iou = iou * correct_class
                # compute best matches between detections and ground truth for each threshold
                for i, threshold in enumerate(iouv):
                    matches = np.nonzero(iou >= threshold)
                    matches = np.array(matches).T
                    if matches.shape[0]:
                        if matches.shape[0] > 1:
                            matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                        tp[matches[:, 1].astype(int), i] = True

                if confusion_matrix is not None:
                    confusion_matrix.process_batch(torch.tensor(np.hstack(
                        [pred[:, :4], conf.reshape(-1, 1), labels.reshape(-1, 1),
                         pred[:, 4].reshape(-1, 1)])), gt, torch.tensor(gt_labels))

            # save stats for current frame
            stats['target_cls'].append(gt_labels)
            stats['conf'].append(conf)
            stats['pred_cls'].append(labels)
            stats['tp'].append(tp)

    # concatenate stats for all images and compute metrics
    stats = {k: np.concatenate(v) for k, v in stats.items()}
    if stats['tp'].any():
        metrics = OBBMetrics(save_dir=Path('.') if results_dir is None else Path(results_dir),
                             names=detector.get_class_names(), plot=results_dir is not None, on_plot=on_plot)
        metrics.process(**stats)
        nt_per_class = np.bincount(stats["target_cls"].astype(int), minlength=n_classes)

        # print results
        keys = metrics.keys + ['images', 'instances']
        o = max(len(name) for name in keys) + 1
        print('\n\nresults:\n' + ' ' * o + ''.join([f'{k:{o}}' for k in keys]))
        print(f'{"all":{o}}' + ''.join(
            [f'{r:<{o}.3f}' for r in metrics.mean_results()]) + f'{n_images["all"]:<{o}}{nt_per_class.sum()}')
        for i, c in enumerate(metrics.ap_class_index):
            print(f'{c:<{o}}' + ''.join(
                [f'{r:<{o}.3f}' for r in metrics.class_result(i)]) + f'{n_images[c]:<{o}}{nt_per_class[c]}')
        if confusion_matrix is not None:
            for normalize in True, False:
                confusion_matrix.plot(save_dir=results_dir, names=('trunk'), normalize=normalize, on_plot=on_plot)

        return metrics.results_dict, plots

    print('no matches found - no stats available')
    return None, plots


def main():
    args = parse_arguments()
    image_list = None
    if args.image_list is not None:
        with open(args.image_list, 'r') as list_file:
            image_list = [line.strip() for line in list_file.readlines()]

    eval_fusion(args.exp_dir, args.results_dir, image_list,
                args.obb_model, args.seg_model, args.min_confidence, args.classes)


if __name__ == '__main__':
    main()
