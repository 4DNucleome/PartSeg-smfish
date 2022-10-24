import numpy as np
import pytest

from PartSeg_smfish.segmentation import (
    GaussBackgroundEstimate,
    GaussBackgroundEstimateParameters,
    LaplacianBackgroundEstimate,
    LaplacianBackgroundEstimateParameters,
)


@pytest.mark.parametrize("mask", (True, False))
def test_gauss_estimate(mask):
    param = GaussBackgroundEstimateParameters(
        background_estimate_radius=5,
        foreground_estimate_radius=2.5,
        estimate_mask=mask,
    )
    data = np.random.random((100, 100))
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[10:20, 10:20] = 1
    GaussBackgroundEstimate.spot_estimate(data, mask, (10, 10), param)


@pytest.mark.parametrize("mask", (True, False))
def test_laplacian_estimate(mask):
    param = LaplacianBackgroundEstimateParameters(
        laplacian_radius=1.3,
        estimate_mask=mask,
    )
    data = np.random.random((100, 100))
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[10:20, 10:20] = 1
    LaplacianBackgroundEstimate.spot_estimate(data, mask, (10, 10), param)
