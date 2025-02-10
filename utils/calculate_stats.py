import os
import argparse
from collections import defaultdict
import cv2
import numpy as np


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='accumulate dataset statistics regarding image resolutions, as well as instance '
                    'counts, sizes, positions and orientations from yolo instance-segmentation annotations')
    parser.add_argument('--anno_dir', type=str, required=True,
                        help='directory containing annotations in yolov8 instance-segmentation format')
    parser.add_argument('--images_dir', type=str, required=True,
                        help='directory containing images corresponding to annotations')
    parser.add_argument('--output_dir', type=str, default='.',
                        help='output directory for result images and csv files')
    parser.add_argument('--heatmap_size', type=int, default=1024,
                        help='image width and height for square heat maps')
    parser.add_argument('--exclude_filters', type=str, nargs='*', default=[],
                        help='exclude image ids starting with any of the given filters')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    return args


# read image size from jpg file
def get_jpg_image_size(image_path):
    with open(image_path, 'rb') as file:
        file.seek(163)
        h = file.read(2)
        w = file.read(2)
        return ((w[0] << 8) + w[1], (h[0] << 8) + h[1])


def calculate_stats(anno_dir, images_dir, output_dir, heatmap_size, exclude_filters):
    resolutions = defaultdict(lambda: defaultdict(int))
    heat_maps = defaultdict(lambda: np.zeros((heatmap_size, heatmap_size), dtype=np.uint16))
    orientation_range = range(180)
    orientations = defaultdict(lambda: {i: 0 for i in orientation_range})
    instance_sizes = defaultdict(list)
    instances_per_subset = defaultdict(lambda: defaultdict(int))

    for anno_file_name in os.listdir(anno_dir):
        image_id, ext = os.path.splitext(anno_file_name)
        if ext == '.txt' and not any(image_id.startswith(filter) for filter in exclude_filters):
            print(f'processing image {image_id}')
            subset = image_id.split('-')[0]
            img_size = get_jpg_image_size(os.path.join(images_dir, f'{image_id}.jpg'))
            resolutions[img_size][subset] += 1

            with open(os.path.join(anno_dir, anno_file_name), 'r') as anno_file:
                for line in anno_file.readlines():
                    anno = line.split(' ')
                    label = anno.pop(0)
                    if len(anno) < 4:
                        print(f'\tinvalid polygon for label {label}')
                        continue

                    contour = np.array(list(zip(anno[::2], anno[1::2])), dtype=np.float32)
                    _, (w, h), angle = cv2.minAreaRect(contour)

                    if w <= 0:
                        print(f'\tinvalid bounding rect for label {label}')
                        continue

                    instances_per_subset[label][subset] += 1
                    instance_sizes[label].append((w, h))
                    angle += 90 if h > w else 180 if h <= 0 and angle < 0 else 0
                    orientations[label][int(angle) % 180] += 1

                    heat_map = np.zeros_like(heat_maps[label])
                    cv2.fillPoly(heat_map, [(contour * np.array((heatmap_size, heatmap_size),
                                                                dtype=np.float32)).astype(int)], 1, cv2.LINE_8)
                    heat_maps[label] += heat_map

    print('\nwriting image resolutions...')
    with open(os.path.join(output_dir, 'resolutions.csv'), 'w') as resolutions_file:
        resolutions_file.write('image width,image height,subset,count\n')
        for resolution in sorted(resolutions, key=lambda x: x[0] * x[1]):
            for subset, count in sorted(resolutions[resolution].items()):
                resolutions_file.write(f'{resolution[0]},{resolution[1]},{subset},{count}\n')

    print('writing labels statistics...')
    with open(os.path.join(output_dir, 'subset_labels.csv'), 'w') as labels_file:
        subsets = set(sum([list(subset_list.keys()) for subset_list in instances_per_subset.values()], start=[]))
        labels_file.write('label,' + ','.join(subsets))
        for label, subset_counts in sorted(instances_per_subset.items()):
            labels_file.write(f'\n{label}')
            for subset in subsets:
                labels_file.write(f',{subset_counts[subset] if subset in subset_counts else ""}')

    print('writing instance sizes...')
    for size_list in instance_sizes.values():
        size_list.sort(key=lambda s: s[0] * s[1])
    sorted_labels = sorted(instance_sizes.keys())

    with open(os.path.join(output_dir, 'instance_sizes.csv'), 'w') as instance_sizes_file:
        instance_sizes_file.write(',,'.join(sorted_labels) + '\n' + ','.join(['width', 'height'] * len(sorted_labels)))
        while True:
            line = []
            for label in sorted_labels:
                line.extend(instance_sizes[label].pop() if len(instance_sizes[label]) > 0 else [None] * 2)
            if all(v is None for v in line):
                break
            instance_sizes_file.write('\n' + ','.join('' if v is None else f'{v:.4f}' for v in line))

    print('writing heat maps...')
    with open(os.path.join(output_dir, 'heatmap_scale.csv'), 'w') as scale_file:
        scale_file.write('label,instances,min (abs),max (abs),min(rel),max(rel)')

        for label, heat_map in sorted(heat_maps.items()):
            n_instances = sum(instances_per_subset[label].values())
            min_val = np.min(heat_map)
            max_val = np.max(heat_map)
            val_range = max_val - min_val

            scale_file.write(
                f'\n{label},{n_instances},{min_val},{max_val},{min_val / n_instances:.4f},{max_val / n_instances:.4f}')

            scaled_map = cv2.convertScaleAbs(heat_map - min_val, alpha=255/val_range)
            cv2.imwrite(os.path.join(output_dir, f'heatmap_gray_{label}.png'), scaled_map)
            cv2.imwrite(os.path.join(output_dir, f'heatmap_color_{label}.png'),
                        cv2.applyColorMap(scaled_map, cv2.COLORMAP_JET))

    scale_image = cv2.resize(np.array(list(range(255, -1, -1)), dtype=np.uint8), dsize=(50, heatmap_size))
    cv2.imwrite(os.path.join(output_dir, 'heatmap_gray_scale.png'), scale_image)
    cv2.imwrite(os.path.join(output_dir, 'heatmap_color_scale.png'), cv2.applyColorMap(scale_image, cv2.COLORMAP_JET))

    print('writing orientations...')
    sorted_labels = sorted(orientations.keys())
    with open(os.path.join(output_dir, 'orientations.csv'), 'w') as orientation_file:
        orientation_file.write('orientation,' + ','.join(sorted_labels))
        for orientation in orientation_range:
            orientation_file.write(f'\n{orientation},' +
                                   ','.join(str(orientations[label][orientation]) for label in sorted_labels))

    print('done.')


def main():
    args = parse_arguments()
    calculate_stats(args.anno_dir, args.images_dir, args.output_dir, args.heatmap_size, args.exclude_filters)


if __name__ == '__main__':
    main()
