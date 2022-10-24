from PartSegCore.analysis.measurement_base import (
    AreaType,
    Leaf,
    MeasurementMethodBase,
    PerComponent,
)


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
