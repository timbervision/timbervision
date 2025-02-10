import os
import argparse
import random
import cv2
import numpy as np

from extract_instances import formats, labels, SIDE, CUT

categories = {v: k for k, v in labels.items()}


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='visualize instance annotations in given format')
    parser.add_argument('--annotation_dir', type=str, required=True, help='input directory containing annotations')
    parser.add_argument('--images_dir', type=str, required=True, help='input directory containing images')
    parser.add_argument('--output_dir', type=str, required=True, help='output directory for visualizations')
    parser.add_argument('--format', choices=list(formats.keys()), default='yolov8-iseg', help='annotation format')
    parser.add_argument('--output_size', type=int, default=0,
                        help='output image size along larger dimension (original if 0)')
    parser.add_argument('--filter_labels', type=int, nargs='*', default=[],
                        help='list of classes (for yolo formats) or instance ids (for tracking formats) to be visualized'
                             '(includes all if empty)')
    parser.add_argument('--filter_ids', type=str, nargs='*', default=[],
                        help='only include image ids containing any of the given filters')
    return parser.parse_args()


# class for assigning different shades of fixed label colors
class Colors:
    BASE_COLORS = {0: [0, 255, 0],
                   1: [0, 255, 244],
                   2: [255, 0, 255],
                   3: [0, 133, 255]}

    def __init__(self, max_val=1.0, min_val=0.4, step=0.2):
        self.max = max_val
        self.min = min_val
        self.step = step
        self.adaptions = {k: self.max +
                          self.step for k in self.BASE_COLORS}

    def get_color(self, label):
        self.adaptions[label] -= self.step
        if self.adaptions[label] < self.min:
            self.adaptions[label] = self.max + self.step
        return [int(c * self.adaptions[label]) for c in self.BASE_COLORS[label]]


# class for consistently assigning randomly generated colors for tracked instances
class TrackColors:
    def __init__(self, min_val=50, max_val=200):
        self.min_val = min_val
        self.max_val = max_val
        self.colors = {}

    def get_color(self, label):
        if label not in self.colors:
            self.colors[label] = [random.randint(self.min_val, self.max_val) for c in range(3)]
        return self.colors[label]


# classes for drawing annotations in specific formats
class AnnoReader:
    def __init__(self, line: str, separator: str):
        self.anno = line.split(separator)

    @staticmethod
    def create_colors():
        return Colors(step=0)

    def get_label(self) -> int:
        raise NotImplementedError()

    def draw(self, image: np.array, color):
        raise NotImplementedError()


class YoloReader(AnnoReader):
    def __init__(self, line):
        super().__init__(line, ' ')

    def get_label(self):
        return int(self.anno[0])


class YoloBBReader(YoloReader):
    def draw(self, image, color):
        center = (float(self.anno[1]), float(self.anno[2]))
        half_size = (float(self.anno[3]) * 0.5, float(self.anno[4]) * 0.5)
        cv2.rectangle(image, (int((center[0] - half_size[0]) * image.shape[1]),
                              int((center[1] - half_size[1]) * image.shape[0])),
                      (int((center[0] + half_size[0]) * image.shape[1]),
                       int((center[1] + half_size[1]) * image.shape[0])), color, thickness=4)


class YoloOBBReader(YoloReader):
    def draw(self, image, color):
        cv2.polylines(image,
                      [np.array([[float(self.anno[i]) * image.shape[1],
                                  float(self.anno[i + 1]) * image.shape[0]]
                                 for i in range(1, 9, 2)], dtype=int)],
                      True, color, thickness=4)


class YoloISegReader(YoloReader):
    def draw(self, image, color):
        polyline = [(int(round(float(x) * image.shape[1])),
                     int(round(float(y) * image.shape[0])))
                    for x, y in zip(self.anno[1::2], self.anno[2::2])]
        overlay = image.copy()
        cv2.fillPoly(overlay, [np.array(polyline, dtype=int)], color, lineType=cv2.LINE_AA)
        cv2.addWeighted(image, 0.5, overlay, 0.5, 0.0, dst=image)
        cv2.polylines(image, [np.array(polyline, dtype=int)], True, [
            int(c * 0.3) for c in color], 3, lineType=cv2.LINE_AA)


class TrackReader(AnnoReader):
    def __init__(self, line):
        super().__init__(line, ',')

    @staticmethod
    def create_colors():
        return TrackColors()

    def get_label(self):
        return int(self.anno[0])


class OBBTrackReader(TrackReader):
    def draw(self, image, color):
        cv2.polylines(
            image,
            [np.array([[round(float(self.anno[i]) * image.shape[1]), round(float(self.anno[i + 1]) * image.shape[0])]
                       for i in range(2, 10, 2)], dtype=int)],
            True, color, thickness=4)


class ISegTrackReader(TrackReader):
    def draw(self, image, color):
        label = int(self.anno[1])
        draw_color = color if categories[label] == SIDE else [
            c + 30 for c in color] if categories[label] == CUT else [250, 250, 250]
        polyline = np.array([[(round(float(x) * image.shape[1]), round(float(y) * image.shape[0]))
                              for x, y in zip(self.anno[2::2], self.anno[3::2])]], dtype=int)
        overlay = image.copy()
        cv2.fillPoly(overlay, [polyline], draw_color, lineType=cv2.LINE_AA)
        cv2.addWeighted(image, 0.4, overlay, 0.6, 0.0, dst=image)
        cv2.polylines(image, [polyline], True, [int(c - 40) for c in draw_color], 4, lineType=cv2.LINE_AA)


def visualize_instances(anno_dir, images_dir, output_dir, anno_format, output_size, filter_labels, filter_ids):
    readers: dict[str, type[TrackReader]] = {'yolov8-det': YoloBBReader,
                                             'yolov8-obb': YoloOBBReader,
                                             'yolov8-iseg': YoloISegReader,
                                             'track-obb': OBBTrackReader,
                                             'track-iseg': ISegTrackReader}

    os.makedirs(output_dir, exist_ok=True)

    colors = readers[anno_format].create_colors()
    for annotation in os.listdir(anno_dir):
        image_id = os.path.splitext(annotation)[0]
        if len(filter_ids) == 0 or any(f in image_id for f in filter_ids):
            image_path = os.path.join(images_dir, image_id + '.jpg')
            if not os.path.exists(image_path):
                image_path = os.path.join(images_dir, image_id + '.png')
            if os.path.exists(image_path):
                print(f'visualizing annotations for image {image_id}')
                with open(os.path.join(anno_dir, annotation), 'r') as anno_file:
                    image = cv2.imread(image_path)
                    if output_size > 0:
                        scale = output_size / max(image.shape[:2])
                        image = cv2.resize(image, dsize=None, fx=scale, fy=scale)

                    for line in anno_file.readlines():
                        reader = readers[anno_format](line)
                        label = reader.get_label()
                        if filter_labels == [] or label in filter_labels:
                            reader.draw(image, colors.get_color(label))

                cv2.imwrite(os.path.join(output_dir, image_id + '.jpg'), image)
    print('Done.')


def main():
    args = parse_arguments()
    visualize_instances(args.annotation_dir, args.images_dir, args.output_dir,
                        args.format, args.output_size, args.filter_labels, args.filter_ids)


if __name__ == '__main__':
    main()
