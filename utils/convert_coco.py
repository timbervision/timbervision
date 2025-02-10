import json
import os
import argparse
from collections import defaultdict


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='convert coco annotation file to original or adapted yolo instance-segmentation format')
    parser.add_argument('--input_file', type=str, required=True,
                        help='path to input json file containing coco annotations')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='directory for storing converted annotations as individual txt files')
    parser.add_argument('--input_labels', type=str, nargs='*', default=['log'],
                        help='list of input annotation label names to be converted')
    parser.add_argument('--output_label', type=str, default=3,
                        help='output label id assigned to instances of all included labels')
    parser.add_argument('--fusion', default=False, action='store_true',
                        help='convert multi-part contours to custom format for fusion evaluation '
                             'instead splitting them into multiple objects compatible with yolo')

    return parser.parse_args()


def convert_coco(input_file, output_dir, input_labels, output_label, fusion_format):
    # load file and create output directory
    with open(input_file, 'r') as file:
        coco_data = json.load(file)
    os.makedirs(output_dir, exist_ok=True)

    print(f'converting {input_file} to {output_dir}...')

    # get category names
    categories = {category['id']: category['name'] for category in coco_data['categories']}

    # collect relevant annotations
    annotations = defaultdict(list)
    for anno in coco_data['annotations']:
        if categories[anno['category_id']] in input_labels:
            annotations[anno['image_id']].append(anno['segmentation'])

    # store data for each image
    for image_data in coco_data['images']:
        image_id = image_data['id']
        if image_id in annotations:
            width = image_data['width']
            height = image_data['height']
            with open(os.path.join(output_dir, f'{os.path.splitext(os.path.basename(image_data["file_name"]))[0]}.txt'),
                      'w') as output_file:
                for anno in annotations[image_id]:
                    if fusion_format:
                        output_file.write(f'{output_label} ' + ':'.join(
                            [' '.join([f'{float(x) / width} {float(y) / height}'
                                       for x, y in zip(seg[::2], seg[1::2])]) for seg in anno]) + '\n')
                    else:
                        for seg in anno:
                            output_file.write(f'{output_label} ' + ' '.join(
                                [f'{min(float(x) / width, 1.0)} {min(float(y) / height, 1.0)}'
                                 for x, y in zip(seg[::2], seg[1::2])]) + '\n')

    print('conversion complete.')


def main():
    args = parse_arguments()
    convert_coco(args.input_file, args.output_dir, args.input_labels, args.output_label, args.fusion)


if __name__ == '__main__':
    main()
