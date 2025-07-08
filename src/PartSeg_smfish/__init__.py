import sys

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

__all__ = ()


def register():
    from PartSegCore.register import RegisterEnum, register as register_fun

    from . import measurement, segmentation

    register_fun(
        segmentation.SMSegmentationBase,
        RegisterEnum.roi_analysis_segmentation_algorithm,
    )
    register_fun(
        segmentation.LayerRangeThresholdFlow,
        RegisterEnum.roi_mask_segmentation_algorithm,
    )
    register_fun(
        segmentation.ThresholdFlowAlgorithmWithDilation,
        RegisterEnum.roi_mask_segmentation_algorithm,
    )
    register_fun(measurement.ComponentType, RegisterEnum.analysis_measurement)

    if getattr(sys, "frozen", False):
        from napari.plugins import plugin_manager

        plugin_manager.register(sys.modules[__name__])
