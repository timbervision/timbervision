from collections import defaultdict
import numpy as np
import cv2
from shapely import LineString, Point
from .detector import SIDE, CUT, TRUNK, Detection


# find any overlaps of a contour with the given image borders
# and return their center points sorted by overlap length
def find_clip_centers(contour, max_x, max_y, margin=0):
    intersections = []
    for p0, p1 in [((x, 0), (x, max_y)) for x in [margin, max_x - margin]] + [
            ((0, y), (max_x, y)) for y in [margin, max_y - margin]]:
        intersection = LineString([p0, p1]).intersection(contour)
        if intersection.length > 0:
            intersections.append(intersection)
    return sorted(intersections, key=lambda x: x.length, reverse=True)


# try to infer a total of two boundary points for a given detection, including existing bounds
def find_missing_bounds(det: Detection, bounds):
    if len(bounds) < 2:
        if SIDE in det.components:
            corners = list(det.components[SIDE].obb.contour.boundary.coords)
            if len(corners) == 5:
                if len(bounds) == 1:
                    # find the obb line opposite the existing boundary
                    edges = [np.argmax([LineString([p0, p1]).distance(bounds[0])
                                        for p0, p1 in zip(corners[:-1], corners[1:])])]
                else:
                    # fall back to using the two short lines of the obb as boundaries
                    p_min = min([0, 1], key=lambda i: Point(corners[i]).distance(Point(corners[i + 1])))
                    edges = [p_min, p_min + 2]
                return [LineString([corners[edge], corners[edge + 1]]).centroid for edge in edges]
    return []


# generic interface for generating results from images and corresponding detections
class DetectionFormat:
    def __init__(self) -> None:
        pass

    # return expected size of output image based on input size
    def get_output_size(self, input_size) -> tuple[int, int]:
        raise NotImplementedError

    # generate output from image and detections sorted by their ids
    def generate(self, image, detections: dict[int, Detection]):
        raise NotImplementedError


# generator for debug images visualizing instance contours, obbs and their fusion
# along with descriptions of their id mapping, labels and matching scores
# ([id]|[score of side-to-cut matching]|[track score]|
# <[label]:[score of obb-to-segmentation matching]|o[obb detection score]|s[segmentation score]>)
class DebugImage(DetectionFormat):
    COLOR_OFFSETS = {CUT: 0,
                     SIDE: 40,
                     TRUNK: 40}

    INVALID_COLOR = (50, 50, 50)  # color for visualizing invalid trunk detections
    CLIP_COLOR = (255, 0, 255)    # color for visualizing center points of trunk boundaries clipped at image border
    BOUNDS_COLOR = (0, 255, 255)  # color for visualizing visible trunk-boundary points

    ID_IMAGE_HEIGHT = 150

    # initialize visualization with optional boundary points, middle axes and description texts for each trunk
    def __init__(self, draw_bounds=False, draw_axis=False, verbose=False, clip_margin=0) -> None:
        super().__init__()
        self.draw_bounds = draw_bounds
        self.draw_axis = draw_axis
        self.verbose = verbose
        self.track_colors = {}
        self.font = cv2.FONT_HERSHEY_PLAIN
        self.clip_margin = clip_margin
        self.color_range = range(50 + max(0, -min(self.COLOR_OFFSETS.values())),
                                 240 - max(0, *self.COLOR_OFFSETS.values()))

    def get_output_size(self, input_size):
        return (input_size[0] * 2, input_size[1] + (self.ID_IMAGE_HEIGHT if self.verbose else 0))

    def generate(self, image, detections: dict[int, Detection]):
        obb_image = image.copy()
        seg_image = np.zeros(image.shape, dtype=np.uint8)
        id_image = np.zeros((self.ID_IMAGE_HEIGHT if self.verbose else 0, image.shape[1] * 2, 3), dtype=np.uint8)
        font_scale = image.shape[1] / 1080
        text_line_offset = 18 * font_scale
        id_text_offset = np.array([id_image.shape[1] - 5, text_line_offset], dtype=int)
        line_width = max(image.shape[0:1]) * 0.005
        contour_width = line_width * 0.3
        center_radius = line_width * 1.2
        axis_dash_length = line_width * 3.5
        axis_gap = axis_dash_length * 0.5
        axis_dash_width = axis_dash_length * 0.18
        axis_dot_radius = axis_dash_width * 0.9
        font_thickness = round(line_width * 0.15)

        bound_points = defaultdict(list)

        for track_id, det in detections.items():
            color = self.assign_color(track_id)

            texts = []
            text_colors = []
            if self.verbose:
                texts = [f'{f"{track_id}" if det.is_valid() else "-"}|'
                         f'{"" if det.trunk_score is None else f"{det.trunk_score:.2f}|"}'
                         f'{f"{det.track_score:.2f}" if det.is_tracked() and det.is_valid() else f"{det.confidence():.2f}"}|']
                text_colors = [self.get_color(TRUNK, color)]

            for label, component in det.components.items():
                # draw obb results
                if self.verbose or label != TRUNK:
                    box = [np.array(component.obb.contour.exterior.coords, dtype=int)]
                    cv2.polylines(obb_image, box, True, color=self.get_color(label, color), thickness=round(line_width))
                    if det.trunk_score is not None:
                        cv2.polylines(obb_image, box, True, color=color, thickness=1)

                if component.seg is not None:
                    # draw fused results
                    contour = np.array(component.seg.contour.exterior.coords, dtype=int)
                    cv2.fillPoly(seg_image, [contour],
                                 color=self.get_color(label, color if det.is_valid() else self.INVALID_COLOR),
                                 lineType=cv2.LINE_AA)
                    cv2.polylines(seg_image, [contour], isClosed=True,
                                  color=(0, 0, 0) if det.is_valid() else self.get_color(label, color),
                                  thickness=round(contour_width) if det.is_valid() else round(contour_width * 2),
                                  lineType=cv2.LINE_AA)
                    if det.trunk_score is not None:
                        center = component.seg.center()
                        cv2.circle(seg_image, (int(center.x), int(center.y)), round(center_radius),
                                   self.get_color(TRUNK, color), -1, lineType=cv2.LINE_AA)

                # generate debug text
                if self.verbose and component.has_score():
                    texts.append(f'{label}:{"" if component.match_score is None else f"{component.match_score:.2f}"}'
                                 f'{"" if component.obb.score is None else f"o{component.obb.score:.2f}"}'
                                 f'{"" if component.seg is None or component.seg.score is None else f"s{component.seg.score:.2f}"}|')
                    text_colors.append(self.get_color(label, color))

            if self.verbose:
                # write debug text
                text_widths = [cv2.getTextSize(text, self.font, font_scale, font_thickness)[0][0] for text in texts]
                text_width = sum(text_widths) + 10
                id_text_offset[0] -= text_width
                if id_text_offset[0] < 0:
                    id_text_offset[0] = id_image.shape[1] - 5 - text_width
                    id_text_offset[1] += text_line_offset
                offset = id_text_offset.copy()
                for text, c, w in zip(texts, text_colors, text_widths):
                    cv2.putText(id_image, text, offset, self.font, font_scale, c, font_thickness, lineType=cv2.LINE_AA)
                    offset[0] += w

            if self.draw_bounds or self.draw_axis:
                # find clipped and back-facing boundaries of trunks
                bounds = [(det.components[CUT].center(), None)] if CUT in det.components else []
                if SIDE in det.components and det.components[SIDE].seg is not None:
                    for clip_center in find_clip_centers(det.components[SIDE].seg.contour, image.shape[1] - 1,
                                                         image.shape[0] - 1, self.clip_margin):
                        center = clip_center.centroid
                        bounds.append((center, self.CLIP_COLOR))
                        if len(bounds) >= 2:
                            break
                bound_points[track_id] = bounds + [(center, self.BOUNDS_COLOR)
                                                   for center in find_missing_bounds(det, [b[0] for b in bounds])]

        if self.draw_bounds:
            # visualize trunk boundaries
            for track_id, bounds in bound_points.items():
                color = self.assign_color(track_id)
                for center, border_color in bounds:
                    if border_color is not None:
                        cv2.circle(seg_image, (int(center.x), int(center.y)), round(center_radius),
                                   border_color, -1, cv2.LINE_AA)
                        cv2.circle(seg_image, (int(center.x), int(center.y)), round(center_radius * 0.6),
                                   color, -1, cv2.LINE_AA)

        if self.draw_axis:
            # visualize middle axes
            for track_id, bounds in bound_points.items():
                if len(bounds) == 2:
                    color = [c - 20 for c in self.get_color(CUT, self.assign_color(track_id))]
                    line = LineString([bounds[0][0], bounds[1][0]])
                    l = 0
                    dot = False
                    while l < line.length:
                        if dot:
                            pt = line.interpolate(l)
                            cv2.circle(seg_image, (round(pt.x), round(pt.y)), round(axis_dot_radius), color, -1)
                            l += axis_gap
                        else:
                            pt0 = line.interpolate(l)
                            pt1 = line.interpolate(min(l + axis_dash_length, line.length))
                            cv2.line(seg_image, (round(pt0.x), round(pt0.y)), (round(pt1.x), round(pt1.y)), color,
                                     round(axis_dash_width), cv2.LINE_AA)
                            l += axis_dash_length + axis_gap
                        dot = not dot

        return cv2.vconcat([cv2.hconcat([obb_image, seg_image]), id_image])

    # modify color for given label to visually match side and cut surfaces of the same trunk
    def get_color(self, label, color):
        if label in self.COLOR_OFFSETS:
            return [c + self.COLOR_OFFSETS[label] for c in color]
        return self.INVALID_COLOR

    # assign a consistent color for each track id
    def assign_color(self, track_id):
        if track_id in self.track_colors:
            return self.track_colors[track_id]
        color = [int(np.random.choice(self.color_range)) for _ in range(3)]
        self.track_colors[track_id] = color
        return color


# generator for uint16 masks encoding labels and track ids based on a given format
class MaskImage(DetectionFormat):
    # label keys to be assigned integer values for custom mappings
    BG = 'background'                    # background pixels
    LATERAL_SURFACE = 'lateral_surface'  # side surfaces of trunks
    VISIBLE_CUT = 'visible_cut'          # front-facing cut surfaces of trunks
    CUT_CENTER = 'cut_center'            # center points of front-facing cut surfaces
    BOUNDARY = 'boundary'                # boundary points of back-facing cut surfaces
    CLIP_CENTER = 'clip_center'          # center points of trunk boundaries clipped at image border

    # initialize with mapping for output labels and number of bits used for storing track ids in each pixel
    def __init__(self, mapping=None, id_bits=12, clip_margin=0) -> None:
        super().__init__()
        self.mapping = {self.BG: 0, self.LATERAL_SURFACE: SIDE + 1,
                        self.VISIBLE_CUT: CUT + 1} if mapping is None else mapping
        if id_bits > 15:
            raise ValueError('invalid value for id_bits - must be < 16')
        self.id_bits = id_bits
        self.clip_margin = clip_margin

    def get_output_size(self, input_size):
        return input_size

    def generate(self, image, detections):
        # initialize output image with background label
        result = np.ones((image.shape[0], image.shape[1]),
                         dtype=np.uint16) * (self.mapping[self.BG] if self.BG in self.mapping else 0)

        # iterate detections
        for track_id, det in detections.items():
            if det.is_valid() and det.is_matched() and det.has_contour():
                bounds = []
                if CUT in det.components:
                    bounds.append(det.components[CUT].center())
                    if self.VISIBLE_CUT in self.mapping and det.components[CUT].seg is not None:
                        cv2.fillPoly(result, [np.array(det.components[CUT].seg.contour.exterior.coords, dtype=int)],
                                     color=int(self.encode(self.VISIBLE_CUT, track_id)), lineType=cv2.LINE_AA)
                    if self.CUT_CENTER in self.mapping:
                        result[int(bounds[0].y), int(bounds[0].x)] = self.encode(self.CUT_CENTER, track_id)
                if SIDE in det.components and det.components[SIDE].seg is not None:
                    contour = det.components[SIDE].seg.contour
                    if self.LATERAL_SURFACE in self.mapping:
                        cv2.fillPoly(result, [np.array(contour.exterior.coords, dtype=int)],
                                     color=int(self.encode(self.LATERAL_SURFACE, track_id)), lineType=cv2.LINE_AA)
                    if self.CLIP_CENTER in self.mapping:
                        for clip_center in find_clip_centers(contour, image.shape[1] - 1, image.shape[0] - 1,
                                                             self.clip_margin):
                            center = clip_center.centroid
                            result[int(center.y), int(center.x)] = self.encode(self.CLIP_CENTER, track_id)
                            bounds.append(center)
                            if len(bounds) >= 2:
                                break
                if self.BOUNDARY in self.mapping:
                    for center in find_missing_bounds(det, bounds):
                        result[int(np.clip(center.y, 0, result.shape[0] - 1)),
                               int(np.clip(center.x, 0, result.shape[1] - 1))] = self.encode(self.BOUNDARY, track_id)

        return result

    # encode label identifier and track id into uint16 pixel value using defined format
    def encode(self, mask_type, track_id):
        return self.mapping[mask_type] << self.id_bits | (track_id & (2 ** self.id_bits - 1))
