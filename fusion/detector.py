import os
from collections import defaultdict
from types import SimpleNamespace
import yaml
import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.trackers import byte_tracker, bot_sort
from ultralytics.cfg import trackers
from shapely.geometry import Polygon, LineString, MultiPolygon, GeometryCollection
from scipy.optimize import linear_sum_assignment

# label ids used in default models
CUT = 0
SIDE = 2
TRUNK = 3


# class for storing the contour and score of a detected object
class ObjectContour:
    def __init__(self, contour: Polygon, score=None) -> None:
        self.contour = contour
        self.score = score

    def center(self):
        return self.contour.centroid

    def obb(self):
        obb = self.contour.minimum_rotated_rectangle
        return obb if obb.is_valid else None


# class for storing the oriented bounding box and score of a detected object
class OBB(ObjectContour):
    def obb(self):
        return self.contour


# calculate intersection over union for two polygons
def iou(a: Polygon, b: Polygon):
    intersection = a.intersection(b).area
    return (intersection / a.union(b).area) if intersection > 0 else 0


# find the line of polygon a containing the hightest overlap with polygon b relative to its length and return the overlap
def line_overlap(a: Polygon, b: Polygon):
    max_overlap = 0.0
    points = list(a.exterior.coords)
    for p0, p1 in zip(points[:-1], points[1:]):
        line = LineString([p0, p1])
        intersection = b.intersection(line).length
        if intersection > 0:
            max_overlap = max(intersection / line.length, max_overlap)
    return max_overlap


# match two lists of contours using a defined score metric and threshold;
# returns two dictionaries mapping matched indices of a to indices in b and corresponding matching scores, respectively
def match(polys_a: list[ObjectContour], polys_b: list[ObjectContour], thresh, metric=iou):
    scores = np.ones((len(polys_a), len(polys_b)), dtype=float) * -1.0
    for i, a in enumerate(polys_a):
        for j, b in enumerate(polys_b):
            scores[i, j] = metric(a.contour, b.contour)

    ind_a, ind_b = linear_sum_assignment(scores, maximize=True)
    valid = scores[ind_a, ind_b] > thresh
    return dict(zip(np.array(ind_a[valid]), np.array(ind_b[valid]))), dict(
        zip(np.array(ind_a[valid]), scores[ind_a, ind_b][valid]))


# convert a bounding box in xywhr format to a polygon
def xywhr2poly(xywhr):
    center = np.array(xywhr[:2], dtype=float)
    sin_a = np.sin(xywhr[4])
    cos_a = np.cos(xywhr[4])
    w = np.array([cos_a, sin_a], dtype=float) * xywhr[2] * 0.5
    h = np.array([-sin_a, cos_a], dtype=float) * xywhr[3] * 0.5
    return Polygon([(center + offset).astype(int) for offset in [w + h, w - h, -w - h, -w + h]])


# convert a polygon to a bounding box in xywhr format
def poly2xywhr(obb):
    points = list(obb.exterior.coords)
    lines = [LineString([points[i], points[i + 1]]).length for i in [0, 1]]
    idx = np.argmax(lines)
    angle = np.arctan2(points[idx + 1][1] - points[idx][1], points[idx + 1][0] - points[idx][0])
    return [obb.centroid.x,
            obb.centroid.y] + (lines if idx == 0 else lines[::-1]) + [angle if angle >= 0 else angle + np.pi]


# class representing a single trunk detection consisting of fused obb and instance-segmentation results
# for up to one side and up to one cut surface
class Detection:
    # detected component consisting of an obb and optional contour, as well as their matching score
    class Component:
        def __init__(self, obb: OBB, seg: ObjectContour = None, match_score=None) -> None:
            self.obb = obb
            self.seg = seg
            self.match_score = None if seg is None else match_score

        def scores(self):
            return [shape.score for shape in [self.obb, self.seg] if shape is not None and shape.score is not None]

        def has_score(self):
            return self.obb.score is not None or (self.seg is not None and self.seg.score is not None)

        def get_match_score(self):
            return 0 if self.match_score is None else self.match_score

        def contour(self):
            return self.obb.contour if self.seg is None else self.seg.contour

        def center(self):
            return self.obb.center() if self.seg is None else self.seg.center()

    def __init__(self, label, component: Component, trunk_score=None) -> None:
        self.components = {label: component}
        self.trunk_score = trunk_score
        self.track_score = None

    def add_component(self, label, component: Component):
        if label in self.components:
            raise ValueError('duplicate label for detection')
        self.components[label] = component

    def track(self, score):
        self.track_score = score
        return self

    def is_matched(self):
        return self.trunk_score is not None or any(c.match_score is not None for c in self.components.values())

    def is_tracked(self):
        return self.track_score is not None

    def is_valid(self):
        return self.track_score is None or self.track_score >= 0

    def has_contour(self):
        return any(c.seg is not None for c in self.components.values())

    def has_detected_obb(self):
        return any(c.obb.score is not None for c in self.components.values())

    def has_score(self, label):
        return label in self.components and self.components[label].has_score()

    def confidence(self):
        return (sum(sum(c.scores()) for c in self.components.values()) / sum(len(c.scores()) for c in self.components.values())
                + sum(c.get_match_score() for c in self.components.values()) / len(self.components)
                + (0.5 if self.trunk_score is None else self.trunk_score)) / 3

    def xywhr(self):
        obb = GeometryCollection([c.contour() for c in self.components.values()
                                 if c.has_score()]).minimum_rotated_rectangle
        return poly2xywhr(obb)


# class applying oriented object detection and instance segmentation on individual frames and fusing results
class TrunkDetector:
    # initialize with trained models for oriented object detection and instance segmentation
    def __init__(self, obb_model_path, seg_model_path):
        self.obb_model = YOLO(obb_model_path)
        self.seg_model = YOLO(seg_model_path)

    # return list of class names defined during obb training
    def get_class_names(self):
        return self.obb_model.names

    # return image size defined during obb training if available
    def get_train_size(self):
        return int(self.obb_model.model.args['imgsz']) if 'imgsz' in self.obb_model.model.args else None

    # apply detection on an image
    def process_image(self, image, min_confidence=0.4) -> dict[int, Detection]:
        # detect obbs
        obbs = self.detect_obbs(image, min_confidence)

        # apply instance segmentation and store results for each label
        contours = self.detect_contours(image, min_confidence)

        # match segmentation results to obbs for each label and aggregate resulting components
        components = defaultdict(list)
        for label in set(contours).union(obbs):
            boxes = [OBB(seg.obb()) for seg in contours[label]]
            matches, scores = match(obbs[label], boxes, 0.2)
            components[label] = [Detection.Component(obb, contours[label][matches[i]], scores[i])
                                 if i in matches else Detection.Component(obb) for i, obb in enumerate(obbs[label])]
            components[label].extend([Detection.Component(box, contours[label][i])
                                     for i, box in enumerate(boxes) if i not in matches.values()])

        # match side to cut surfaces
        trunk_matches, trunk_match_scores = match([c.obb for c in components[SIDE]],
                                                  [c.obb for c in components[CUT]],
                                                  0.05, metric=line_overlap)

        # generate detections from matches between side and cut surfaces along with their corresponding segmentations
        matched_indices = defaultdict(set)
        detections = []
        for i, (i_side, i_cut) in enumerate(trunk_matches.items()):
            detections.append(Detection(SIDE, components[SIDE][i_side], trunk_match_scores[i_side]))
            detections[-1].add_component(CUT, components[CUT][i_cut])
            matched_indices[SIDE].add(i_side)
            matched_indices[CUT].add(i_cut)

        # generate detections from unmatched components
        for label, label_components in components.items():
            for i, component in enumerate(label_components):
                if i not in matched_indices[label]:
                    detections.append(Detection(label, component))

        return {i: det for i, det in enumerate(detections)}

    # apply model for obb detection and return relevant results for each label
    def detect_obbs(self, image, min_confidence):
        obb_results = self.obb_model.predict(image, verbose=False, conf=min_confidence)[0].obb
        obbs = defaultdict(list)
        for label, pts, conf in zip(obb_results.cls.tolist(), obb_results.xyxyxyxy.tolist(), obb_results.conf.tolist()):
            obbs[int(label)].append(OBB(Polygon(pts), conf))
        return obbs

    # apply model for instance segmentation and return relevant results for each label
    def detect_contours(self, image, min_confidence):
        seg_results = defaultdict(list)
        for res in self.seg_model.predict(image, verbose=False, retina_masks=True, conf=min_confidence)[0]:
            box = res.boxes.xyxy[0]
            mask = res.masks.data[0, int(box[1]):int(box[3]), int(box[0]):int(box[2])].cpu().numpy().astype("uint8")
            contours = [c for c in cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
                        if c.shape[0] > 2]
            if len(contours) > 0:
                contour = Polygon(
                    max(contours, key=lambda x: x.shape[0]).reshape(-1, 2) + np.array([int(box[0]), int(box[1])]))
                if not contour.is_valid:
                    contour = contour.buffer(0)
                    if isinstance(contour, MultiPolygon):
                        contour = max(list(contour.geoms), key=lambda c: c.length)
                    if not isinstance(contour, Polygon) or not contour.is_valid:
                        continue
                seg_results[int(res.boxes.cls)].append(ObjectContour(contour, float(res.boxes.conf[0])))
        return seg_results


# class for tracking trunks with consistent ids across multiple frames
class TrunkTracker(TrunkDetector):
    # initialize with models for oriented object detection detection and instance segmentation,
    # as well as tracker configuration
    def __init__(self, obb_model_path, seg_model_path, include_invalid=False,
                 tracker_config=os.path.join(trackers.__path__[0], 'botsort.yaml')):
        super().__init__(obb_model_path, seg_model_path)
        self.include_invalid = include_invalid
        with open(tracker_config, 'r') as config_file:
            config = SimpleNamespace(**yaml.safe_load(config_file))
        if config.tracker_type == 'bytetrack':
            self.tracker = byte_tracker.BYTETracker(config)
        elif config.tracker_type == 'botsort':
            self.tracker = bot_sort.BOTSORT(config)
        else:
            raise ValueError(f'invalid tracker type in configuration: {config.tracker_type}')

    def process_image(self, image, min_confidence=0.4):
        detections = list(super().process_image(image, min_confidence).values())
        result = {}
        if len(detections) > 0:
            tracks = self.tracker.update(
                SimpleNamespace(conf=np.array([det.confidence() for det in detections], dtype=float),
                                cls=np.ones(len(detections)),
                                xywhr=np.array([det.xywhr() for det in detections]), dtype=float))
            for track in tracks:
                det = detections[int(track[-1])].track(track[6])
                det.add_component(TRUNK, Detection.Component(OBB(xywhr2poly(track[:5]))))
                result[int(track[5])] = det
            if self.include_invalid:
                result.update({-i: det.track(-1) for i, det in enumerate(detections, start=1) if not det.is_tracked()})
        return result

    def reset(self):
        self.tracker.reset()
