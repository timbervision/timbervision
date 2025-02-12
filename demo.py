import argparse
import os
import cv2
from fusion.detector import TrunkDetector, TrunkTracker
from fusion.formats import DebugImage, MaskImage


def parse_arguments():
    parser = argparse.ArgumentParser(description='demonstrator for tree-trunk detection and tracking '
                                                 'based on oriented object detection and instance segmentation')
    parser.add_argument('--input', type=str, required=True, help='input image directory or video file')
    parser.add_argument('--results_dir', type=str, required=True, help='directory for storing results')
    parser.add_argument('--obb_model', type=str, default=None,
                        help='path to model providing oriented bounding boxes (loads default model if not specified)')
    parser.add_argument('--seg_model', type=str, default=None,
                        help='path to model providing instance-segmentation contours '
                             '(loads default model if not specified)')
    parser.add_argument('--min_confidence', type=float, default=0.4,
                        help='minimum confidence for results provided by detection and segmentation models')
    parser.add_argument('--track', action='store_true', default=False, help='apply tracking across frames')
    parser.add_argument('--tracker_config', type=str, default='config/botsort_optimized.yaml',
                        help='path to tracker-configuration file')
    parser.add_argument('--clip_margin', type=float, default=0,
                        help='pixel margin from image border for identifying boundaries of clipped detections')
    parser.add_argument('--mask', action='store_true', default=False,
                        help='generate single-channel mask images instead of debug visualizations')
    parser.add_argument('--show_scores', action='store_true', default=False,
                        help='include scores and fused trunk obbs in visualization images')

    return parser.parse_args()


# read a directory of images or a video file and apply detection and optionally tracking for each frame
def demo(obb_model_path, seg_model_path, input_path, results_dir, min_confidence, clip_margin, tracker_config,
         tracking, generate_masks, show_scores):
    os.makedirs(results_dir, exist_ok=True)

    # initialize detection/tracking and result format
    print('loading models...')
    fusion = TrunkTracker(obb_model_path, seg_model_path, tracker_config=tracker_config,
                          include_invalid=not generate_masks) if tracking else TrunkDetector(obb_model_path,
                                                                                             seg_model_path)
    output_format = MaskImage(clip_margin=clip_margin) if generate_masks else DebugImage(
        clip_margin=clip_margin, draw_bounds=False, verbose=show_scores, draw_axis=True)

    if os.path.isdir(input_path):
        # read input data from image files and generate result images
        for image_name in sorted(os.listdir(input_path)):
            print(f'processing image {os.path.splitext(image_name)[0]}...')
            image = cv2.imread(os.path.join(input_path, image_name))
            cv2.imwrite(os.path.join(results_dir, f'{os.path.splitext(image_name)[0]}.png'),
                        output_format.generate(image, fusion.process_image(image, min_confidence)))
    else:
        # read input data from video file and generate result video
        cap_input = cv2.VideoCapture(input_path)
        output = cv2.VideoWriter(
            os.path.join(results_dir, f'result_{os.path.splitext(os.path.split(input_path)[-1])[0]}.avi'),
            cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), cap_input.get(cv2.CAP_PROP_FPS),
            output_format.get_output_size((int(cap_input.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                           int(cap_input.get(cv2.CAP_PROP_FRAME_HEIGHT)))))
        n_frames = int(cap_input.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_num = 0
        while cap_input.isOpened():
            print(f'processing frame {frame_num} / {n_frames}')
            frame_num += 1
            ret, frame = cap_input.read()
            if not ret:
                break
            output.write(output_format.generate(frame, fusion.process_image(frame, min_confidence)))
        cap_input.release()
        output.release()


def main():
    args = parse_arguments()
    demo(args.obb_model, args.seg_model, args.input, args.results_dir, args.min_confidence,
         args.clip_margin, args.tracker_config, args.track, args.mask, args.show_scores)


if __name__ == '__main__':
    main()
