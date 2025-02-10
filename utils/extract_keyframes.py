import argparse
import os
import cv2
from tqdm import tqdm


def parse_arguments():
    parser = argparse.ArgumentParser(description='extract keyframes from video files')
    parser.add_argument('--videos_dir', type=str, required=True, help='directory containing input videos')
    parser.add_argument('--output_dir', type=str, required=True, help='output directory for extracted images')
    parser.add_argument('--fps', type=int, default=None,
                        help='number of extracted frames per second (defaults to video frame rate if None)')
    return parser.parse_args()


def extract_keyframes(videos_dir, output_dir, fps):
    if os.path.isdir(videos_dir):
        os.makedirs(output_dir, exist_ok=True)
        for file_name in os.listdir(videos_dir):
            print(f'extracting video file {file_name}')
            cap = cv2.VideoCapture(os.path.join(videos_dir, file_name))
            if cap.isOpened():
                video_fps = int(cap.get(propId=cv2.CAP_PROP_FPS))
                frame_count = int(cap.get(propId=cv2.CAP_PROP_FRAME_COUNT))
                offset = 1 if fps is None else round(video_fps / fps)
                target_prefix = os.path.splitext(file_name)[0]

                for idx in tqdm(range(frame_count), desc=f'processing {file_name}'):
                    if not cap.grab():
                        break
                    if idx % offset == 0:
                        ret, image = cap.retrieve()
                        if ret:
                            cv2.imwrite(os.path.join(output_dir, target_prefix + f'-{idx:06d}' + '.jpg'), image)
                        else:
                            print(f'\tError retrieving video frame {idx}')
        print('Done.')
    else:
        print(f'Invalid input directory: {videos_dir}')


def main():
    args = parse_arguments()
    extract_keyframes(args.videos_dir, args.output_dir, args.fps)


if __name__ == '__main__':
    main()
