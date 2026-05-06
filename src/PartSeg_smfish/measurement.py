from enum import Enum

import numpy as np
from PartSegCore import autofit as af
from PartSegCore.analysis.measurement_base import (
    AreaType,
    Leaf,
    MeasurementMethodBase,
    PerComponent,
)
from PartSegCore.analysis.measurement_calculation import get_border
from PartSegCore.roi_info import BoundInfo
from PartSegImage.image import Spacing
from sympy import symbols
from PartSegCore.utils import BaseModel
from toolz.functoolz import return_none


class ComponentType(MeasurementMethodBase):
    text_info = "Component type", "If roi is in nucleus or in cytoplasm"

    __argument_class__ = BaseModel

    @classmethod
    def get_starting_leaf(cls):
        return Leaf(
            name=cls.text_info[0],
            area=AreaType.ROI,
            per_component=PerComponent.Yes,
        )

    @staticmethod
    def calculate_property(roi_annotation, _component_num, **kwargs):
        return roi_annotation[_component_num].get("type")

    @classmethod
    def get_units(cls, ndim):
        return 1


class DistanceToNucleusCenter(MeasurementMethodBase):
    text_info = (
        "Distance to nucleus center",
        "Distance to nucleus center in units",
    )

    __argument_class__ = BaseModel

    @classmethod
    def need_full_data(cls):
        return True

    @classmethod
    def get_starting_leaf(cls):
        return Leaf(
            name=cls.text_info[0],
            area=AreaType.ROI,
            per_component=PerComponent.Yes,
        )

    @staticmethod
    def calculate_property(
        area_array,
        roi_alternative: dict[str, np.ndarray],
        help_dict: dict,
        result_scalar: float,
        voxel_size: tuple[float, float, float],
        **kwargs,
    ):
        if "nucleus_center" not in help_dict:
            nucleus = roi_alternative["nucleus"]
            help_dict["nucleus_center"] = (
                af.density_mass_center(nucleus, voxel_size) * result_scalar
            )
        nucleus_center = help_dict["nucleus_center"]
        roi_center = (
            af.density_mass_center(area_array, voxel_size) * result_scalar
        )
        return np.sqrt(np.sum((nucleus_center - roi_center) ** 2))

    @classmethod
    def get_units(cls, ndim):
        return symbols("{}")


class DistanceFromNucleusBorder(MeasurementMethodBase):
    text_info = (
        "Distance from nucleus border",
        "Distance from nucleus border in units",
    )

    __argument_class__ = BaseModel

    @classmethod
    def need_full_data(cls):
        return True

    @classmethod
    def get_starting_leaf(cls):
        return Leaf(
            name=cls.text_info[0],
            area=AreaType.ROI,
            per_component=PerComponent.Yes,
        )

    @staticmethod
    def calculate_property(
        area_array,
        roi_alternative: dict[str, np.ndarray],
        help_dict: dict,
        result_scalar: float,
        roi_annotation: dict,
        _component_num: int,
        voxel_size: tuple[float, float, float],
        **kwargs,
    ):
        if "nucleus_border" not in help_dict:
            nucleus = roi_alternative["nucleus"]
            area_pos = np.transpose(np.nonzero(get_border(nucleus))).astype(
                float
            )
            area_pos += 0.5
            for i, val in enumerate(
                (x * result_scalar for x in reversed(voxel_size)), start=1
            ):
                area_pos[:, -i] *= val
            help_dict["nucleus_border"] = area_pos

        nucleus_border = help_dict["nucleus_border"]
        roi_center = (
            af.density_mass_center(area_array, voxel_size) * result_scalar
        )
        res = np.sqrt(
            np.min(np.sum((nucleus_border - roi_center) ** 2, axis=1))
        )
        if roi_annotation[_component_num].get("type") == "Nucleus":
            return -res
        return res

    @classmethod
    def get_units(cls, ndim):
        return symbols("{}")


class DimensionName(Enum):
    X = -1
    Y = -2
    Z = -3


class CenterType(Enum):
    Geometrical_center = 1
    Mass_center = 2


class CenterCoordinateParameters(BaseModel):
    dimension: DimensionName = DimensionName.X
    center_type: CenterType = CenterType.Geometrical_center


class CenterCoordinate(MeasurementMethodBase):
    text_info = "Center coordinate", "Center coordinate in units"

    __argument_class__ = CenterCoordinateParameters

    @classmethod
    def get_units(cls, ndim):
        return symbols("{}")

    @staticmethod
    def calculate_property(
            area_array: np.ndarray,
            channel: np.ndarray,
            dimension: DimensionName,
            center_type: CenterType,
            bounds_info: BoundInfo,
            voxel_size: Spacing,
            _component_num: int,
            result_scalar: float,
            **kwargs):
        shift = bounds_info[_component_num].lower * voxel_size * result_scalar
        if center_type == CenterType.Mass_center:
            im = np.copy(channel)
            im[area_array == 0] = 0
            area_pos = np.array([af.density_mass_center(im, voxel_size) * result_scalar])
        else:
            area_pos = np.array([af.density_mass_center(area_array > 0, voxel_size) * result_scalar])
        print(area_pos, shift, voxel_size)
        result_center = area_pos[0] + shift

        return result_center[dimension.value]