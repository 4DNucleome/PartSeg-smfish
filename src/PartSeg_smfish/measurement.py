import numpy as np
from PartSegCore import autofit as af
from PartSegCore.analysis.measurement_base import (
    AreaType,
    Leaf,
    MeasurementMethodBase,
    PerComponent,
)
from PartSegCore.analysis.measurement_calculation import get_border
from sympy import symbols


class ComponentType(MeasurementMethodBase):
    text_info = "Component type", "If roi is in nucleus or in cytoplasm"

    @classmethod
    def get_fields(cls):
        return []

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

    @classmethod
    def get_fields(cls):
        return []

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

    @classmethod
    def get_fields(cls):
        return []

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
