import os
import argparse
from collections import defaultdict
from typing import TextIO
import numpy as np
import cv2
import motmetrics as mm
from shapely import Polygon, MultiPolygon, GeometryCollection
from tqdm import tqdm
from ultralytics.cfg import trackers
from fusion.detector import TrunkTracker, Detection, TRUNK
from fusion.formats import DebugImage


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='run trunk tracking and apply motmetrics to evaluate results on multiple image sequences')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='input directory containing images of all sequences named as <sequence_id>-<frame_id>.jpg')
    parser.add_argument('--gt_dir', type=str, required=True,
                        help='directory containing csv ground-truth annotations with names corresponding to images and '
                             'each line specifying a single track as <track id>,<label id>,<x1>,<y1>,<x2>,<y2>,...')
    parser.add_argument('--vis_dir', type=str, default=None,
                        help='directory for optional output visualizations')
    parser.add_argument('--obb_model', type=str, default=None,
                        help='path to oriented-object-detection model (loads default model if not specified)')
    parser.add_argument('--seg_model', type=str, default=None,
                        help='path to instance-segmentation model (loads default model if not specified)')
    parser.add_argument('--min_confidence', type=float, default=0.4,
                        help='minimum confidence for results of detection and segmentation models')
    parser.add_argument('--tracker_config', type=str, default='config/botsort_optimized.yaml',
                        help='path to tracker-configuration file (defaults to yolo configs if not found otherwise)')
    parser.add_argument('--eval_obb', action='store_true', default=False,
                        help='evaluate based on ground truth for overall trunk obbs '
                             'instead of contours for individual components')
    parser.add_argument('--frame_offset', type=int, default=1,
                        help='offset between evaluated frames for simulating lower input frame rate')

    return parser.parse_args()


# generic class for handling ground-truth processing and metrics accumulation
class SeqAccumulator:
    def __init__(self, gt_dir) -> None:
        self.gt_dir = gt_dir
        self.acc = defaultdict(lambda: mm.MOTAccumulator(auto_id=False))
        self.gt = {}

        # identify sequences with annotations
        self.gt_seq = set()
        for gt_file_name in os.listdir(gt_dir):
            image_id, ext = os.path.splitext(gt_file_name)
            if ext == '.csv':
                self.gt_seq.add(self.extract_seq_id(image_id))

    # update accumulator with tracks and ground truth for current frame if available
    def update(self, image_id, tracks, image_size):
        seq_id = self.extract_seq_id(image_id)
        self.gt = defaultdict(dict)
        gt_file_path = os.path.join(self.gt_dir, f'{image_id}.csv')
        if os.path.exists(gt_file_path):
            with open(gt_file_path, 'r') as gt_file:
                self._extract_gt(gt_file, image_size)
            valid_tracks = {k: t for k, t in tracks.items() if self._is_valid(t)}
            self.acc[seq_id].update(list(self.gt), list(valid_tracks), self.iou_dist(valid_tracks.values()),
                                    int(image_id[len(seq_id) + 1:]))

    # check if ground truth is available for sequence
    def has_gt(self, seq_id):
        return seq_id in self.gt_seq

    # get sequence id from image id
    @staticmethod
    def extract_seq_id(image_id):
        return '-'.join(image_id.split('-')[:-1])

    # create a matrix of inverse iou scores between ground-truth and track components
    def iou_dist(self, tracks: list[Detection]):
        dist = np.ones((len(self.gt), len(tracks)), dtype=float) * np.nan
        for i_gt, gt in enumerate(self.gt.values()):
            for i_track, track in enumerate(tracks):
                track_labels = self._get_labels(track)
                intersection = sum(gt[label].intersection(
                    self._get_component(track, label)).area for label in gt if label in track_labels)
                if intersection > 0:
                    dist[i_gt, i_track] = 1 - intersection / sum(
                        gt[label].area if label not in track_labels else self._get_component(
                            track, label).area if label not in gt else gt[label].union(
                                self._get_component(track, label)).area for label in set(gt).union(track_labels))
        return dist

    # visualize latest ground truth on given image
    def visualize(self, image, offset):
        offset_vec = np.array(offset, dtype=int)
        for contours in self.gt.values():
            for contour in contours.values():
                cv2.polylines(
                    image, [np.array(c.exterior.coords, dtype=int) + offset_vec
                            for c in (list(contour.geoms) if isinstance(contour, GeometryCollection) else [contour])],
                    False, (0, 0, 255), 2, cv2.LINE_AA)

    # compute metrics from accumulated data
    def compute_metrics(self):
        metrics = mm.metrics.create()
        return metrics.compute_many(
            list(self.acc.values()), names=list(self.acc.keys()), generate_overall=True,
            metrics=['num_frames', 'num_matches', 'num_detections', 'num_false_positives', 'num_detections',
                     'num_objects', 'num_predictions', 'idfp', 'idfn', 'idtp'] + mm.metrics.motchallenge_metrics)

    # extract specific ground-truth format to self.gt dictionary
    def _extract_gt(self, anno_file: TextIO, image_size: tuple[int, int]):
        raise NotImplementedError()

    # extract relevant component from track
    @staticmethod
    def _get_component(track: Detection, label: int) -> Polygon:
        raise NotImplementedError()

    # list relevant detection labels for track components
    @staticmethod
    def _get_labels(track: Detection) -> list[int]:
        raise NotImplementedError()

    # check if track is valid for evaluation
    @staticmethod
    def _is_valid(track: Detection) -> bool:
        return track.is_valid()


# accumulator for ground truth containing obbs of trunks
class OBBAccumulator(SeqAccumulator):
    def _extract_gt(self, anno_file, image_size):
        for line in anno_file.readlines():
            anno = line.split(',')
            label = int(anno[1])
            if label == TRUNK:
                self.gt[anno[0]][label] = Polygon(
                    [(float(anno[i]) * image_size[1], float(anno[i + 1]) * image_size[0]) for i in range(2, 10, 2)])

    @staticmethod
    def _get_component(track, label):
        return track.components[label].obb.contour

    @staticmethod
    def _get_labels(track):
        return [TRUNK]


# accumulator for ground truth containing segmentation masks of trunk components
class ISegAccumulator(SeqAccumulator):
    def _extract_gt(self, anno_file, image_size):
        gt = defaultdict(list)
        for line in anno_file.readlines():
            anno = line.split(',')
            seg = Polygon([(float(x) * image_size[1], float(y) * image_size[0]) for x, y in zip(anno[2::2], anno[3::2])])
            if not seg.is_valid:
                seg = seg.buffer(0)
                if isinstance(seg, MultiPolygon):
                    seg = max(list(seg.geoms), key=lambda c: c.length)
                if not isinstance(seg, Polygon) or not seg.is_valid:
                    return
            gt[(anno[0], int(anno[1]))].append(seg)

        for (track_id, label), seg_list in gt.items():
            if len(seg_list) == 1:
                self.gt[track_id][label] = seg_list[0]
            else:
                self.gt[track_id][label] = GeometryCollection(seg_list)

    @staticmethod
    def _get_component(track, label):
        return track.components[label].seg.contour

    @staticmethod
    def _get_labels(track):
        return [k for k in track.components.keys() if track.components[k].seg is not None]

    @staticmethod
    def _is_valid(track):
        return track.is_valid() and track.has_contour()


# evaluate tracking on multiple sequences based on mot metrics
def mot_eval(input_dir, gt_dir, vis_dir, obb_model_path, seg_model_path,
             min_confidence, tracker_config, eval_obb, frame_offset):
    # initialize tracker and result format
    print('initializing tracker...')
    tracker = TrunkTracker(obb_model_path, seg_model_path, include_invalid=True,
                           tracker_config=tracker_config if os.path.exists(tracker_config) else os.path.join(
                               trackers.__path__[0], tracker_config))
    formatter = None if vis_dir is None else DebugImage()
    if vis_dir is not None:
        os.makedirs(vis_dir, exist_ok=True)

    # initialize ground truth and accumulation
    acc = OBBAccumulator(gt_dir) if eval_obb else ISegAccumulator(gt_dir)

    train_size = tracker.get_train_size()

    # store list of images for each annotated sequence
    seq = defaultdict(list)
    for image_name in sorted(os.listdir(input_dir)):
        image_id, ext = os.path.splitext(image_name)
        if ext == '.jpg':
            seq_id = acc.extract_seq_id(image_id)
            if acc.has_gt(seq_id):
                seq[seq_id].append(image_id)
    print(f'found {len(seq)} valid sequences\n')

    # evaluate selected sequences with individual accumulators
    for seq_id, seq_images in seq.items():
        tracker.reset()
        for image_id in tqdm(seq_images, desc=f'processing sequence {seq_id}'):
            if int(image_id[len(seq_id) + 1:]) % frame_offset == 0:
                # read image and apply tracking
                image = cv2.imread(os.path.join(input_dir, f'{image_id}.jpg'))
                if train_size is not None:
                    scale = train_size / max(image.shape[:2])
                    image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                tracks = tracker.process_image(image, min_confidence)

                # add frame to accumulator if ground truth exists
                acc.update(image_id, tracks, image.shape[:2])

                # generate debug visualization
                if formatter is not None:
                    vis = formatter.generate(image, tracks)
                    acc.visualize(vis, [(0 if eval_obb else 1) * image.shape[1], 0])
                    cv2.imwrite(os.path.join(vis_dir, f'{image_id}.png'), vis)

    # evaluate accumulated results
    summary = acc.compute_metrics()

    with open('results_mot.csv', 'w') as results_file:
        for seq_id in summary.axes[0]:
            results_file.write(f',{seq_id}')
        for metric in summary.axes[1]:
            results_file.write(f'\n{metric}')
            for seq_id in summary.axes[0]:
                results_file.write(f',{summary[metric][seq_id]}')

        results_file.write('\n\n')
        for k, v in {'mode': 'obb' if eval_obb else 'seg', 'obb model': obb_model_path, 'seg model': seg_model_path,
                     'min_confidence': min_confidence, 'frame_offset': frame_offset,
                     **tracker.tracker.args.__dict__}.items():
            results_file.write(f'\n{k},{v}')

    print(f'\n{summary}')


def main():
    args = parse_arguments()
    mot_eval(args.input_dir, args.gt_dir, args.vis_dir, args.obb_model, args.seg_model,
             args.min_confidence, args.tracker_config, args.eval_obb, args.frame_offset)


if __name__ == '__main__':
    main()
