import sys

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

__all__ = ()


def register():
    from PartSegCore.register import RegisterEnum
    from PartSegCore.register import register as register_fun

    from PartSeg_smfish import measurement, segmentation

    register_fun(
        segmentation.SMSegmentationBase,
        RegisterEnum.roi_analysis_segmentation_algorithm,
    )
    register_fun(
        segmentation.LayerRangeThresholdFlow,
        RegisterEnum.roi_mask_segmentation_algorithm,
    )
    register_fun(measurement.ComponentType, RegisterEnum.analysis_measurement)

    if getattr(sys, "frozen", False):
        import napari

        napari.plugins.plugin_manager.register(sys.modules[__name__])
