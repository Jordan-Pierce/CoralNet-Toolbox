#!/usr/bin/env python

"""Tests for `coralnet_toolbox` package."""

import sys
import types

if "cv2" not in sys.modules:
	cv2_stub = types.SimpleNamespace(
		TERM_CRITERIA_EPS=1,
		TERM_CRITERIA_MAX_ITER=2,
		INTER_NEAREST=0,
		BORDER_CONSTANT=0,
		resize=lambda *args, **kwargs: args[0],
		remap=lambda *args, **kwargs: args[0],
	)
	sys.modules["cv2"] = cv2_stub

import numpy as np
import pytest

from coralnet_toolbox.Rasters.QtRaster import Raster


def test_estimate_batch_warp_bytes_scales_with_batch_and_shape():
	maps = [np.zeros((10, 20), dtype=np.int32) for _ in range(3)]

	estimated = Raster._estimate_batch_warp_bytes(maps)

	assert estimated > 0
	assert estimated == Raster._estimate_batch_warp_bytes(maps)


def test_warp_batch_cuda_rejects_mismatched_shapes_before_stacking(monkeypatch):
	monkeypatch.setattr(Raster, "_estimate_batch_warp_bytes", staticmethod(lambda maps, grid_gpu=None: 0))

	maps = [
		np.zeros((4, 4), dtype=np.int32),
		np.zeros((5, 4), dtype=np.int32),
	]

	with pytest.raises(ValueError, match="same shape"):
		Raster.warp_batch_cuda(maps, [-1, -1], grid_gpu=None, oob_mask=None)


def test_warp_batch_cuda_rejects_over_budget_batches(monkeypatch):
	monkeypatch.setattr(Raster, "_estimate_batch_warp_bytes", staticmethod(lambda maps, grid_gpu=None: 10**12))

	maps = [np.zeros((4, 4), dtype=np.int32)]

	with pytest.raises(ValueError, match="safe CUDA budget"):
		Raster.warp_batch_cuda(maps, [-1], grid_gpu=None, oob_mask=None)

