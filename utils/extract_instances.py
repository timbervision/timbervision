import argparse
import json
import os
import collections
from scipy import interpolate
import numpy as np
import cv2
from calculate_stats import get_jpg_image_size


CONTOUR_WIDTH = 2
N_LINE_SAMPLES = 2000

# mapping for scalabel annotation labels
EDGE = 'side'
SECTION_AREA = 'front'
SECTION_LINE = 'back'
UNDEFINED = 'undefined'
MARKER = 'marker'

# output class names
SIDE = 'side'
CUT = 'cut'
BOUND = 'bound'
TRUNK = 'trunk'

labels = {CUT: 0,
          BOUND: 1,
          SIDE: 2,
          TRUNK: 3}


class BBox:
    def __init__(self, vertices, category):
        x = [v[0] for v in vertices]
        y = [v[1] for v in vertices]
        self.left = min(x)
        self.top = min(y)
        self.right = max(x)
        self.bottom = max(y)
        self.label = labels[category]

    def width(self):
        return self.right - self.left

    def height(self):
        return self.bottom - self.top

    def center_x(self):
        return (self.left + self.right) * 0.5

    def center_y(self):
        return (self.top + self.bottom) * 0.5


# extract vertices for given categories from json entry corresponding to an image;
# returns lists of vertices by trunk-instance id and category
def extract_vertices(image_labels, categories=None, include_trees=False):
    instances = collections.defaultdict(lambda: collections.defaultdict(list))
    for label in image_labels:
        category = label['category']
        if label['poly2d']:
            trunk_id = label['attributes']['id'][0]
            if include_trees or trunk_id != 0:
                if category == 'undefined':
                    vertices = label['poly2d'][0]['vertices']
                    if len(vertices) == 2:
                        if categories is None or MARKER in categories:
                            instances[trunk_id][MARKER].append(vertices)
                    elif categories is None or UNDEFINED in categories:
                        instances[trunk_id][UNDEFINED].append(vertices)
                elif categories is None or category in categories:
                    instances[trunk_id][category].append(
                        label['poly2d'][0]['vertices'])
    return instances


# extract bounding boxes for vertices defining cut, boundary, side (edge + section line)
# and trunk (cut + side) instances
def extract_bboxes(instances):
    bboxes = []
    for entries in instances.values():
        for category in [SECTION_AREA, SECTION_LINE]:
            if category in entries:
                for vertices in entries[category]:
                    bboxes.append(BBox(vertices, CUT if category == SECTION_AREA else BOUND))
        if EDGE in entries or SECTION_LINE in entries:
            bboxes.append(BBox([v for vertices in (entries[EDGE] if EDGE in entries else [])
                                + (entries[SECTION_LINE] if SECTION_LINE in entries else []) for v in vertices], SIDE))
        bboxes.append(BBox([v for vertices in entries.values() for v in sum(vertices, [])], TRUNK))
    return bboxes


# interpolate a smoothed spline for a list of vertices
def smooth(vertices):
    try:
        f = interpolate.splprep([[int(v[0]) for v in vertices], [int(v[1]) for v in vertices]], s=0, per=False)
        x, y = interpolate.splev(np.linspace(0, 1, N_LINE_SAMPLES), f[0])
        return np.reshape(np.concatenate((np.reshape(x, (-1, 1)), np.reshape(y, (-1, 1))), axis=1),
                          (-1, 1, 2)).astype(int)
    except ValueError:
        return None


# extract contours of objects in given categories and export with defined writer
def extract_contours(label_entries, image_size, writer, categories):
    instances = extract_vertices(
        label_entries, [SECTION_AREA, SECTION_LINE, EDGE, MARKER, UNDEFINED], True)

    # get side-surface delineation from full contour image,
    # since shared edges are only annotated for one trunk id each
    contour_image = np.zeros((image_size[1], image_size[0]), dtype=np.uint8)
    contour_image[0, :] = contour_image[-1, :] = contour_image[:, 0] = contour_image[:, -1] = 255
    trunk_parts = collections.defaultdict(lambda: collections.defaultdict(list))
    markers = set()

    for trunk_id, entries in instances.items():
        # interpolate spline for each side contour (edge or boundary line) and add to contour image
        for part in [EDGE, SECTION_LINE]:
            for side_vertices in entries[part] if part in entries else []:
                vertices = smooth(side_vertices) if len(
                    side_vertices) > 3 else np.array(side_vertices, np.int32).reshape((-1, 1, 2))
                if vertices is not None:
                    cv2.polylines(contour_image, [np.array(
                        vertices, np.int32)], False, 255, CONTOUR_WIDTH, lineType=cv2.LINE_AA)
                    if part == SECTION_LINE and SECTION_LINE in categories:
                        writer(BOUND, trunk_id, vertices)
                else:
                    print(
                        f'\tError smoothing side vertices for trunk id {trunk_id}')

        # interpolate cut surfaces either as spline or ellipse, save by trunk id and add to contour image
        if SECTION_AREA in entries:
            for cut_vertices in entries[SECTION_AREA]:
                if len(cut_vertices) > 5:
                    vertices = smooth(cut_vertices + [cut_vertices[0]])
                    if vertices is not None:
                        cv2.polylines(contour_image, [vertices], False, 255, CONTOUR_WIDTH, lineType=cv2.LINE_AA)
                    else:
                        print(
                            f'\tError smoothing cut vertices for trunk id {trunk_id}')
                elif len(cut_vertices) == 5:
                    ellipse = cv2.fitEllipse(
                        np.array(cut_vertices, dtype=int))
                    vertices = cv2.ellipse2Poly((int(ellipse[0][0]), int(ellipse[0][1])),
                                                (int(ellipse[1][0] * 0.5),
                                                 int(ellipse[1][1] * 0.5)),
                                                int(ellipse[2]), 0, 360, 5).reshape(-1, 1, 2)
                    cv2.ellipse(contour_image, box=ellipse, color=255,
                                thickness=CONTOUR_WIDTH, lineType=cv2.LINE_AA)
                else:
                    print(
                        f'\tWarning: ignoring cut with less than five points for trunk id {trunk_id} '
                        f'- check scalabel annotation!')
                    vertices = None
                if vertices is not None:
                    if SECTION_AREA in categories:
                        writer(CUT, trunk_id, vertices)
                    if TRUNK in categories:
                        trunk_parts[trunk_id][CUT].append(vertices)

        # interpolate spline for undefined shapes and add to contour image
        if UNDEFINED in entries:
            for shape_vertices in entries[UNDEFINED]:
                vertices = smooth(shape_vertices + [shape_vertices[0]])
                if vertices is not None:
                    if TRUNK in categories:
                        trunk_parts[trunk_id][UNDEFINED].append(vertices)
                    cv2.polylines(contour_image, [vertices],
                                  True, 255, CONTOUR_WIDTH, lineType=cv2.LINE_AA)
                else:
                    print(
                        f'\tError smoothing vertices of undefined shape for trunk id {trunk_id}')

        # store markers with corresponding trunk ids
        for marker in entries[MARKER] if MARKER in entries else []:
            markers.add(((int(np.mean([int(x[0]) for x in marker])),
                          int(np.mean([int(x[1]) for x in marker]))), trunk_id))

    # extract contours and sort by area to ensure selection of innermost shape for each side surface
    contours = sorted(cv2.findContours(contour_image, cv2.RETR_LIST,
                      cv2.CHAIN_APPROX_SIMPLE)[0], key=cv2.contourArea)

    # find side contour containing each trunk marker
    for contour in contours:
        if len(contour) > 2:
            found = []
            for marker in markers:
                if cv2.pointPolygonTest(contour, marker[0], False) > 0:
                    if len(found) == 0:
                        if SIDE in categories:
                            writer(SIDE, marker[1], contour)
                        if TRUNK in categories:
                            trunk_parts[marker[1]][SIDE].append(contour)
                    else:
                        print(
                            f'\tWarning: found multiple markers for same contour of trunk {marker[1]} '
                            f'- check scalabel annotation!')
                    found.append(marker)
            for marker in found:
                markers.remove(marker)

    # extract trunk contours by merging sides, cuts, and undefined shapes
    for trunk_id, parts in trunk_parts.items():
        if len(parts) > 1:
            trunk_image = np.zeros(
                (image_size[1], image_size[0]), dtype=np.uint8)
            for side in parts[SIDE]:
                cv2.fillPoly(trunk_image, side.reshape(1, -1, 2), 1, cv2.LINE_AA)
            for cut in parts[CUT]:
                cv2.fillPoly(trunk_image, cut.reshape(1, -1, 2), 1, lineType=cv2.LINE_AA)
            for shape in parts[UNDEFINED]:
                cv2.fillPoly(trunk_image, shape.reshape(1, -1, 2), 1, lineType=cv2.LINE_AA)
            trunk_image = cv2.dilate(trunk_image, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
            trunk_image = cv2.erode(trunk_image, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
            trunk_contours, _ = cv2.findContours(trunk_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            for contour in trunk_contours:
                writer(TRUNK, trunk_id, contour)
        elif CUT in parts:
            for cut in parts[CUT]:
                writer(TRUNK, trunk_id, cut)
        elif SIDE in parts:
            for side in parts[SIDE]:
                writer(TRUNK, trunk_id, side)


# generate axis-aligned detection annotations in yolov8 format
def generate_yolov8_det(label_entries, output_dir, name, images_dir):
    image_size = get_jpg_image_size(os.path.join(images_dir, name + '.jpg'))
    bboxes = extract_bboxes(extract_vertices(
        label_entries, [SECTION_AREA, SECTION_LINE, EDGE]))

    with open(os.path.join(output_dir, name + '.txt'), 'w') as file:
        for bbox in bboxes:
            file.write(
                f'{bbox.label} {bbox.center_x() / image_size[0]} {bbox.center_y() / image_size[1]} '
                f'{bbox.width() / image_size[0]} {bbox.height() / image_size[1]}\n')


# generate oriented bounding boxes in yolov8 format
def generate_yolov8_obb(label_entries, output_dir, name, images_dir):
    image_size = get_jpg_image_size(os.path.join(images_dir, name + '.jpg'))

    with open(os.path.join(output_dir, name + '.txt'), 'w') as file:

        def write_bbox(category, _, vertices):
            file.write(f'{labels[category]}')
            for pt in cv2.boxPoints(cv2.minAreaRect(np.array(vertices, dtype=np.float32))):
                file.write(
                    f' {pt[0] / image_size[0]} {pt[1] / image_size[1]}')
            file.write('\n')

        extract_contours(label_entries, image_size,
                         write_bbox, [SECTION_AREA, SECTION_LINE, SIDE, TRUNK])


# generate instance-segmentation annotations in yolov8 format
def generate_yolov8_iseg(label_entries, output_dir, name, images_dir):
    image_size = get_jpg_image_size(os.path.join(images_dir, name + '.jpg'))

    # write annotations to file
    with open(os.path.join(output_dir, name + '.txt'), 'w') as file:

        def write_entry(category, _, vertices):
            file.write(f'{labels[category]}')
            for vertex in vertices:
                file.write(
                    f' {np.clip(vertex[0][0] / image_size[0], 0.0, 1.0)}'
                    f' {np.clip(vertex[0][1] / image_size[1], 0.0, 1.0)}')
            file.write('\n')

        extract_contours(label_entries, image_size,
                         write_entry, [SECTION_AREA, SIDE, TRUNK, SECTION_LINE])


# generate ground truth for tracking obbs of trunk instances
def generate_obb_track(label_entries, output_dir, name, images_dir):
    image_size = get_jpg_image_size(os.path.join(images_dir, name + '.jpg'))

    # write annotations to file
    with open(os.path.join(output_dir, name + '.csv'), 'w') as file:

        def write_entry(_, trunk_id, vertices):
            file.write(f'{trunk_id},{labels[TRUNK]}')
            for pt in cv2.boxPoints(cv2.minAreaRect(np.array(vertices, dtype=np.float32))):
                file.write(f',{pt[0] / image_size[0]},{pt[1] / image_size[1]}')
            file.write('\n')

        extract_contours(label_entries, image_size, write_entry, [TRUNK])


# generate ground truth for tracking contours of cuts and sides
def generate_iseg_track(label_entries, output_dir, name, images_dir):
    image_size = get_jpg_image_size(os.path.join(images_dir, name + '.jpg'))

    # write annotations to file
    with open(os.path.join(output_dir, name + '.csv'), 'w') as file:

        def write_entry(category, trunk_id, vertices):
            file.write(f'{trunk_id},{labels[category]}')
            for vertex in vertices:
                file.write(
                    f',{np.clip(vertex[0][0] / image_size[0], 0.0, 1.0)}'
                    f',{np.clip(vertex[0][1] / image_size[1], 0.0, 1.0)}')
            file.write('\n')

        extract_contours(label_entries, image_size, write_entry, [SECTION_AREA, SIDE])


formats = {'yolov8-det': generate_yolov8_det,
           'yolov8-obb': generate_yolov8_obb,
           'yolov8-iseg': generate_yolov8_iseg,
           'track-obb': generate_obb_track,
           'track-iseg': generate_iseg_track}


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='extract instance annotations from scalabel data')
    parser.add_argument('--scalabel_annotation', type=str, required=True,
                        help='input path containing scalabel annotations')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='output directory for converted annotations')
    parser.add_argument('--images_dir', type=str, required=True,
                        help='directory containing images corresponding to annotations')
    parser.add_argument('--format', choices=list(formats.keys()),
                        default='yolov8-iseg', help='output annotation format')
    return parser.parse_args()


def extract_instances(scalabel_annotation, output_dir, images_dir, anno_format):
    os.makedirs(output_dir, exist_ok=True)
    generator = formats[anno_format]
    print(f'extracting annotations from {scalabel_annotation}...')
    with open(scalabel_annotation) as json_file:
        for entry in json.load(json_file):
            if entry['labels']:
                image_id = os.path.splitext(entry['name'].split('/')[-1])[0]
                print(f'converting image id {image_id}...')
                generator(entry['labels'], output_dir, image_id, images_dir)
    print('Done.')


def main():
    args = parse_arguments()
    extract_instances(args.scalabel_annotation, args.output_dir, args.images_dir, args.format)


if __name__ == '__main__':
    main()
