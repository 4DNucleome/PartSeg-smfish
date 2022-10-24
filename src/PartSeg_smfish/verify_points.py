from itertools import product
from typing import List

import numpy as np
from magicgui import magic_factory
from napari.layers import Labels, Points
from napari.types import LayerDataTuple
from scipy.spatial.distance import cdist


def group_points(points: np.ndarray, max_dist=1) -> List[List[np.ndarray]]:
    points = np.copy(points)
    points[:, -3] = np.round(points[:, -3])
    sort = np.argsort(points[:, -3])
    points = points[sort]
    max_val = points[-1, -3]
    prev_data = points[points[:, -3] == 0]
    point_groups = []
    index_info = {}
    for i in range(1, int(max_val + 1)):
        new_points = points[points[:, -3] == i]

        if new_points.size == 0 or prev_data.size == 0:
            index_info = {}
            for j, point in enumerate(new_points):
                index_info[j] = len(point_groups)
                point_groups.append([point])
            prev_data = new_points
            continue
        new_index_info = {}
        dist_array = cdist(prev_data[:, -2:], new_points[:, -2:])
        close_object = dist_array < max_dist
        consumed_set = set()
        close_indices = np.nonzero(close_object)
        for first, second in zip(*close_indices):
            consumed_set.add(second)
            point_groups[index_info[first]].append(new_points[second])
            new_index_info[second] = index_info[first]
        for j, point in enumerate(new_points):
            if j in consumed_set:
                continue
            new_index_info[j] = len(point_groups)
            point_groups.append([point])
        prev_data = new_points
        index_info = new_index_info

    return point_groups


def _shift_array(points_to_roi: int, ndim: int) -> np.ndarray:
    base = tuple([0] * (ndim - 2))
    return np.array(
        [
            base + (x, y)
            for x, y in product(
                range(-points_to_roi, points_to_roi + 1), repeat=2
            )
            if x**2 + y**2 <= points_to_roi**2
        ],
        dtype=np.uint16,
    )


class MatchResults:
    def __init__(self, points_grouped, labels):
        self.points_grouped: List[List[np.ndarray]] = points_grouped
        self.matched_points: List[bool] = [False for _ in self.points_grouped]
        if 0 in labels:
            labels.remove(0)
        self.labels_preserve: np.ndarray = np.arange(
            (max(labels) if labels else 0) + 1
        )
        self.labels = labels
        self.ignored = 0

    def __repr__(self):
        matched_points_count = sum(self.matched_points)
        return (
            f"MatchResults(ignored={self.ignored}, matched "
            f"{matched_points_count} of {len(self.matched_points)}, "
            f"labels_preserved {np.count_nonzero(self.labels_preserve)}"
            f" of {len(self.labels_preserve)})"
        )


def verify_sm_segmentation(
    segmentation: np.ndarray,
    points: np.ndarray,
    points_dist: int = 2,
    points_to_roi: int = 1,
    ignore_single_points: bool = True,
) -> MatchResults:
    points_grouped = group_points(points, points_dist)
    result = MatchResults(points_grouped, set(np.unique(segmentation)))

    shift_array = _shift_array(points_to_roi, segmentation.ndim)

    for i, points_group in enumerate(points_grouped):
        values = []
        for point in points_group:
            coords = (shift_array + point.astype(np.int16)).astype(np.int16)
            values.extend(segmentation[tuple(coords.T)])
        for value in values:
            if value == 0:
                continue
            if value in result.labels:
                result.labels.remove(value)
                result.labels_preserve[value] = 0
            result.matched_points[i] = True
        if (
            ignore_single_points
            and len(points_group) == 1
            and not result.matched_points[i]
        ):
            result.matched_points[i] = True
            result.ignored += 1
    result.matched_points[0] = False
    return result


@magic_factory(info={"widget_type": "TextEdit"}, call_button=True)
def verify_segmentation(
    segmentation: Labels,
    points: Points,
    points_dist: int = 2,
    points_to_roi: int = 1,
    ignore_single_points: bool = True,
    info: str = "",
) -> List[LayerDataTuple]:
    match_result = verify_sm_segmentation(
        segmentation.data,
        points.data,
        points_dist,
        points_to_roi,
        ignore_single_points,
    )
    all_labels = np.count_nonzero(np.unique(segmentation.data))

    verify_segmentation.info.value = (
        f"matched {np.sum(match_result.matched_points)} of"
        f" {len(match_result.matched_points)}"
        f"\nconsumed {all_labels - len(match_result.labels)} of"
        f" {all_labels} segmentation components"
        + f"\nignored {match_result.ignored}"
        if ignore_single_points
        else ""
    )
    res = []
    for ok, points_group in zip(
        match_result.matched_points, match_result.points_grouped
    ):
        if not ok:
            res.extend(points_group)

    missed_labels = (
        match_result.labels_preserve[segmentation.data],
        {"name": "Missed ROI", "scale": points.scale},
        "labels",
    )
    missed_points = (
        np.array(res) if res else None,
        {
            "name": "Missed points",
            "scale": points.scale,
            "face_color": "red",
            "ndim": segmentation.data.ndim,
        },
        "points",
    )

    return [LayerDataTuple(missed_points), LayerDataTuple(missed_labels)]


@magic_factory(info={"widget_type": "TextEdit"}, call_button=True)
def find_single_points(
    points: Points,
    points_dist: int = 2,
    info: str = "",
) -> LayerDataTuple:
    points_grouped = group_points(points.data, points_dist)
    points_res = [x[0] for x in points_grouped if len(x) == 1]
    find_single_points.info.value = (
        f"Single points count: {len(points_res)} of {len(points_grouped)},"
        f" ratio {len(points_res)/len(points_grouped)}"
    )
    points_res = np.array(points_res) if points_res else None
    return LayerDataTuple(
        (
            points_res,
            {
                "name": "Single points",
                "scale": points.scale,
                "face_color": "green",
            },
            "points",
        )
    )
