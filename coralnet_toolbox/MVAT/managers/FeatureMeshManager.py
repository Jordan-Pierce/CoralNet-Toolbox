"""
FeatureMeshManager — Tier 2 feature buffer orchestration for MVAT.

Responsibilities:
  - Precondition checking (feature + index maps present)
  - Worker launch + caching
  - QueryEngine instantiation
  - Viewer integration (array attachment, recolor triggers)
"""

from __future__ import annotations

import hashlib
from typing import Optional, List, Tuple, Dict, Any

import numpy as np

try:
    import torch
except ImportError:
    torch = None

from coralnet_toolbox.Features.QueryEngine import QueryEngine
from coralnet_toolbox.MVAT.core.FeatureBuffer import FeatureBuffer
from coralnet_toolbox.MVAT.workers.FeatureBakeWorker import FeatureBakeWorker
from coralnet_toolbox.MVAT.utils.FeatureBufferCodec import save_feature_buffer
from PyQt5.QtCore import QThread, Qt
from PyQt5.QtWidgets import (
    QApplication, QDialog, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QSpinBox,
    QPushButton, QGroupBox, QCheckBox, QDoubleSpinBox, QWidget, QFormLayout
)


class FeatureMeshManager:
    """
    Manages Tier-2 feature buffer baking, caching, and querying.

    Owns:
      - Current buffer (in memory)
      - QueryEngine instance
      - Worker thread
    """

    def __init__(self, mvat_manager):
        """
        Args:
            mvat_manager: MVATManager instance (access to cameras, cache_manager, etc.).
        """
        self.mvat_manager = mvat_manager
        self.viewer = mvat_manager.viewer
        self.main_window = mvat_manager.main_window
        self.cache_manager = mvat_manager.cache_manager

        self.buffer: Optional[FeatureBuffer] = None
        self.query_engine: Optional[QueryEngine] = None
        self.bake_worker_thread: Optional[QThread] = None
        # Bake parameters parked while the pre-bake index-map pass runs.
        self._pending_bake: Optional[dict] = None

        # Similarity-overlay engagement (the 3D analogue of the 2D tool's work
        # area): while engaged, the viewer draws the similarity overlay actor
        # (mesh/point cloud) or the splat display channel is mixed in, and the
        # shared annotation-window colormap dropdown + opacity slider drive it.
        # Mutually exclusive with the 2D tool's work-area engagement.
        self.overlay_engaged = False
        self._engaging = False  # reentrancy guard

        # Phase-2 GPU colormap shader (SimilarityShader). Bypasses VTK's per-change
        # color-buffer rebuild (the ~125 ms-at-4M / seconds-at-76M cost — confirmed
        # by profiling: a no-change render is ~5 ms, a post-recolor render ~125 ms)
        # by colormapping in a fragment shader from a small value texture, with
        # ScalarVisibilityOff so VTK never rebuilds the cell colors. If the shader
        # can't install, we fall back to direct RGB cell colors (slower but works).
        self.shader_enabled = True
        self.shader_state = None

        # Colormap for the binary similarity view (the multi-class view colors
        # by label palette instead). Changing it is a 256-entry LUT swap on the
        # mesh shader texture / splat display LUT — instant, no recompute.
        self._colormap_name = "plasma"

        self._weighting_config = {
            "use_angle": True,
            "use_inv_dist": True,
            "use_edge_guard": False,
        }
        # How the coarse patch features are sampled up to image resolution
        # during the bake ("nearest" | "bilinear"). Part of the cache key.
        self._interpolation = "nearest"

    def prepare(self, scope: str = "all") -> Tuple[List[Tuple[str, Any]], Dict[str, int]]:
        """
        Check preconditions: which cameras have both feature_map AND index_map loaded?

        Args:
            scope: "all" | "selected" | "visible".

        Returns:
            (eligible_cameras, stats): List of (path, Camera) with both maps.
                stats = {"with_feature_map": N, "with_index_map": M, "both": K, "missing_feature": ..., ...}
        """
        stats = {
            "with_feature_map": 0,
            "with_index_map": 0,
            "both": 0,
            "missing_feature": 0,
            "missing_index": 0,
        }

        eligible = []

        for path, camera in self._collect_scope_cameras(scope):
            raster = getattr(camera, "_raster", None)
            if raster is None:
                continue

            has_fm, has_im = self._camera_map_presence(raster)

            if has_fm:
                stats["with_feature_map"] += 1
            else:
                stats["missing_feature"] += 1

            if has_im:
                stats["with_index_map"] += 1
            else:
                stats["missing_index"] += 1

            if has_fm and has_im:
                stats["both"] += 1
                eligible.append((path, camera))

        return eligible, stats

    def _collect_scope_cameras(self, scope: str) -> List[Tuple[str, Any]]:
        """(path, Camera) list for a bake scope: "all" | "selected" | "visible"."""
        if scope == "selected":
            return [
                (p, self.mvat_manager.cameras[p])
                for p in getattr(self.mvat_manager, "selected_camera_paths", [])
                if p in self.mvat_manager.cameras
            ]
        if scope == "visible":
            try:
                visible = set(self.mvat_manager._get_visible_context_camera_paths())
            except Exception:
                visible = set()
            return [(p, c) for p, c in self.mvat_manager.cameras.items()
                    if p in visible]
        return list(self.mvat_manager.cameras.items())

    @staticmethod
    def _camera_map_presence(raster) -> Tuple[bool, bool]:
        """(has_feature_map, has_index_map) — existence checks ONLY.

        Never touch raster.has_feature_map() / raster.index_map here: those read
        the lazy properties, which pull every map off disk through the LRU
        (decompressing all N cameras' maps just to count them — the dialog-open
        lag). An index map counts as present when it is RESIDENT, on DISK, or
        RECOVERABLE via the aggressive-mode recompute provider — the last case
        is what lets scope="all" bake cameras that are not currently visible in
        the ContextMatrix (their dense map was dropped on unpin, but the bake
        worker's lazy ``raster.index_map`` read regenerates it on demand).
        """
        has_fm = (getattr(raster, "_feature_map", None) is not None
                  or bool(getattr(raster, "feature_map_path", None)))
        has_im = (getattr(raster, "_index_map", None) is not None
                  or bool(getattr(raster, "index_map_path", None))
                  or getattr(raster, "_index_map_provider", None) is not None)
        return has_fm, has_im

    def _collect_missing_index_cameras(self, scope: str) -> List[Any]:
        """Cameras in scope that HAVE a feature map but NO recoverable index map.

        These are the cameras the pre-bake pass must compute visibility for
        (typically: never shown in the ContextMatrix, so no index map was ever
        rendered)."""
        missing = []
        for path, camera in self._collect_scope_cameras(scope):
            raster = getattr(camera, "_raster", None)
            if raster is None:
                continue
            has_fm, has_im = self._camera_map_presence(raster)
            if has_fm and not has_im:
                missing.append(camera)
        return missing

    # Pre-bake index-map computation is chained in slices so the aggressive-mode
    # VisibilityWorker (which ships every camera's DENSE map in one finished
    # payload) never holds more than this many native-res maps at once.
    PREBAKE_SLICE = 16

    def bake(self, compressor_kind: str = "nn", compressor_dim: int = 32,
             scope: str = "all", interpolation: str = "nearest",
             nn_params: dict = None) -> None:
        """
        Bake entry point: complete missing index maps, then launch the worker.

        Cameras in scope that have a feature map but no index map (never shown
        in the ContextMatrix, so visibility was never rendered for them) first
        get their index maps computed via the existing visibility machinery, in
        chained slices of PREBAKE_SLICE with status-bar progress. Only then is
        the actual bake launched (_launch_bake). Trade-off, accepted: in
        aggressive (no-disk) mode each pre-computed map is dropped on unpin and
        regenerated lazily by the bake worker (~25-40 ms/camera) — two renders
        per missing camera, but RAM stays flat.

        Args:
            compressor_kind: "nn" | "pca".
            compressor_dim: Target D.
            scope: "all" | "selected" | "visible".
            interpolation: "nearest" | "bilinear" — how the coarse patch
                features are sampled up to image resolution.
            nn_params: optional dict of NN autoencoder hyperparameters
                (hidden_dim, epochs, lr, beta) when compressor_kind == "nn".
        """
        status_bar = self.mvat_manager.main_window.status_bar
        if self.bake_worker_thread is not None:
            status_bar.showMessage("Bake already running.", 4000)
            return
        if getattr(self.mvat_manager, "_is_computing_visibility", False):
            status_bar.showMessage(
                "Bake: a visibility computation is already running — retry shortly.",
                5000)
            return

        self._pending_bake = {
            "compressor_kind": compressor_kind,
            "compressor_dim": compressor_dim,
            "scope": scope,
            "interpolation": interpolation,
            "nn_params": nn_params,
        }

        primary_target = self.viewer.scene_context.get_primary_target()
        missing = self._collect_missing_index_cameras(scope)
        if not missing or primary_target is None:
            self._launch_bake()
            return

        status_bar.showMessage(
            f"Bake: computing index maps for {len(missing)} camera(s) first…", 0)
        slices = [missing[i:i + self.PREBAKE_SLICE]
                  for i in range(0, len(missing), self.PREBAKE_SLICE)]
        self._run_prebake_slice(slices, 0, primary_target)

    def _run_prebake_slice(self, slices: List[List[Any]], idx: int,
                           primary_target) -> None:
        """Compute index maps for slice ``idx``, then chain to the next / bake."""
        total = sum(len(s) for s in slices)
        done = sum(len(s) for s in slices[:idx])
        self.mvat_manager.main_window.status_bar.showMessage(
            f"Bake: index maps {done}/{total} cameras…", 0)

        def _on_slice_done(_ok: bool):
            # A failed slice is not fatal — its cameras simply stay missing and
            # are counted in the bake's skip report.
            if primary_target is not self.viewer.scene_context.get_primary_target():
                self.mvat_manager.main_window.status_bar.showMessage(
                    "Bake aborted: 3D model changed during index-map computation.",
                    5000)
                self._pending_bake = None
                return
            nxt = idx + 1
            if nxt < len(slices):
                self._run_prebake_slice(slices, nxt, primary_target)
            else:
                self._launch_bake()

        self.mvat_manager._compute_visibility_async(
            primary_target, slices[idx], on_complete=_on_slice_done)

    def _launch_bake(self) -> None:
        """Launch the bake worker for the stored pending-bake parameters."""
        params = self._pending_bake or {}
        self._pending_bake = None
        compressor_kind = params.get("compressor_kind", "nn")
        compressor_dim = params.get("compressor_dim", 32)
        scope = params.get("scope", "all")
        interpolation = params.get("interpolation", "nearest")
        nn_params = params.get("nn_params")

        self._interpolation = str(interpolation or "nearest").lower()
        eligible, stats = self.prepare(scope)

        if stats["both"] == 0:
            self.mvat_manager.main_window.status_bar.showMessage(
                "Bake failed: no cameras have both feature maps and index maps loaded.",
                5000
            )
            return

        # Check that all feature maps use the same model
        model_ids = set()
        for path, camera in eligible:
            mid = getattr(camera._raster, "feature_map_model_id", "unknown")
            model_ids.add(mid)

        if len(model_ids) > 1:
            self.mvat_manager.main_window.status_bar.showMessage(
                f"Bake failed: mixed feature models across cameras: {model_ids}",
                5000
            )
            return

        model_id = model_ids.pop() if model_ids else "unknown"

        # Baking replaces any existing feature mesh: clear the current buffer
        # (and detach it from the mesh / tool) before building the new one. Done
        # only after preconditions pass, so a no-op bake never wipes a good buffer.
        if self.buffer is not None:
            self.clear()

        # Create compressor
        if compressor_kind == "pca":
            from coralnet_toolbox.MVAT.core.FeatureBuffer import PCACompressor
            compressor = PCACompressor(compressor_dim)
        elif compressor_kind == "nn":
            from coralnet_toolbox.MVAT.core.FeatureBuffer import NNCompressor
            params = nn_params or {}
            compressor = NNCompressor(
                compressor_dim,
                hidden_dim=params.get("hidden_dim", 256),
                epochs=params.get("epochs", 30),
                lr=params.get("lr", 1e-3),
                beta=params.get("beta", 1.0),
            )
        else:
            raise ValueError(f"Unknown compressor: {compressor_kind}")

        # Show a busy cursor for the whole bake (compressor fit on the main
        # thread + the background scatter). Restored in _on_bake_finished /
        # _on_bake_error; restored here too if launch fails before then.
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            # Warp maps for distorted cameras must be FIRST-built on the main
            # thread (the bake worker's lazy index-map recompute would otherwise
            # first-touch them off-thread — see cache_index_maps_to_disk).
            for _p, camera in eligible:
                raster = getattr(camera, "_raster", None)
                if (raster is not None and getattr(camera, "is_distorted", False)
                        and getattr(raster, "intrinsics_undistorted", None) is not None):
                    try:
                        raster._ensure_warp_maps()
                    except Exception:
                        pass

            # Fit compressor on sampled patches
            self._fit_compressor(compressor, eligible)

            # Launch worker
            primary_target = self.viewer.scene_context.get_primary_target()
            N = primary_target.get_element_count()
        except Exception:
            QApplication.restoreOverrideCursor()
            raise

        worker = FeatureBakeWorker(
            eligible_cameras=eligible,
            primary_target=primary_target,
            compressor=compressor,
            weighting_config=self._weighting_config,
            element_count=N,
            cache_manager=self.cache_manager,
            interpolation=self._interpolation,
        )

        thread = QThread()
        self.bake_worker_thread = thread
        worker.moveToThread(thread)
        thread.started.connect(worker.run)

        worker.signals.finished.connect(
            lambda buf: self._on_bake_finished(buf, model_id, eligible)
        )
        worker.signals.error.connect(
            lambda err: self._on_bake_error(err)
        )
        worker.signals.progress.connect(
            lambda done, total, msg: (
                self.mvat_manager.main_window.status_bar.showMessage(
                    f"Baking… {msg}", 0
                )
            )
        )

        # Both finished and error must quit the thread so it can be cleaned up
        worker.signals.finished.connect(thread.quit)
        worker.signals.error.connect(thread.quit)
        worker.signals.finished.connect(worker.deleteLater)

        # Remove this entry from _active_workers when the thread finishes
        # (mirrors MVATManager._run_visibility_worker's lifecycle pattern).
        def _remove_worker(t=thread, w=worker):
            self.mvat_manager._active_workers = [
                (ot, ow) for ot, ow in self.mvat_manager._active_workers
                if ot is not t
            ]

        thread.finished.connect(_remove_worker)

        def _is_alive(t):
            try:
                return t.isRunning()
            except RuntimeError:
                return False

        self.mvat_manager._active_workers = [
            (t, w) for t, w in self.mvat_manager._active_workers if _is_alive(t)
        ]
        self.mvat_manager._active_workers.append((thread, worker))

        thread.start()

    def _fit_compressor(self, compressor, eligible: List[Tuple[str, Any]]) -> None:
        """Fit the compressor on a sample of patches from loaded feature maps."""
        # Sample patches
        sample_list = []
        max_sample = 200_000
        per_camera = max(100, max_sample // max(len(eligible), 1))

        for path, camera in eligible:
            try:
                fm = camera._raster.feature_map
                if fm is None:
                    continue
                fm = np.asarray(fm, dtype=np.float32)
                h, w, C = fm.shape
                total = h * w
                if total > per_camera:
                    indices = np.random.choice(total, per_camera, replace=False)
                    sample = fm.reshape(total, C)[indices]
                else:
                    sample = fm.reshape(total, C)
                sample_list.append(sample)
            except Exception as e:
                print(f"[FeatureMeshManager] Fit sample from {path} failed: {e}")
                continue

        if sample_list:
            sample_feats = np.vstack(sample_list)
            compressor.fit(sample_feats)

    def _on_bake_finished(self, buffer: FeatureBuffer, model_id: str,
                          eligible: List[Tuple[str, Any]]) -> None:
        """Handle successful bake completion.

        The worker thread is done by now, but there's still heavy main-thread
        post-processing: the QueryEngine GPU upload, the shader value/LUT
        textures, and the cache write to disk. Keep the busy cursor up until ALL
        of that finishes (restore in finally) — otherwise the cursor goes normal
        while the UI is still frozen.
        """
        try:
            self.buffer = buffer
            self.query_engine = QueryEngine(buffer.features, buffer.valid)
            self._build_shader_state(buffer)

            # NOTE: automated disk caching of the feature buffer is intentionally
            # disabled — the compressed write of the [N,D] buffer dominated the
            # post-bake stall. The cache methods (_cache_buffer / load_from_cache,
            # CacheManager.*_feature_buffer, FeatureBufferCodec) are kept dormant
            # for possible future use.

            # Attach arrays to the mesh
            primary_target = self.viewer.scene_context.get_primary_target()
            primary_target.attach_feature_arrays(buffer)

            # Sync the freshly built LUT sinks with the shared colormap choice.
            self._sync_colormap_from_ui()
            self.apply_colormap()

            # Repopulate the array dropdown
            self.viewer._update_array_selector()

            # Status — include how many cameras contributed and how many were
            # skipped (no feature/index map even after the pre-bake pass).
            n_valid = int(np.sum(buffer.valid))
            skipped = list(buffer.provenance.get("skipped_cameras", []))
            n_cams = int(buffer.provenance.get("num_cameras", 0))
            msg = (f"Baked {n_valid}/{buffer.features.shape[0]} elements from "
                   f"{n_cams - len(skipped)} camera(s).")
            if skipped:
                msg += f" Skipped {len(skipped)} (no feature/index map)."
            msg += " Query ready."
            self.mvat_manager.main_window.status_bar.showMessage(msg, 8000)
        finally:
            self.bake_worker_thread = None
            QApplication.restoreOverrideCursor()

    def _on_bake_error(self, error_msg: str) -> None:
        """Handle bake error."""
        QApplication.restoreOverrideCursor()
        self.mvat_manager.main_window.status_bar.showMessage(f"Bake error: {error_msg}", 5000)
        self.bake_worker_thread = None

    def _cache_buffer(self, buffer: FeatureBuffer, model_id: str,
                      eligible: List[Tuple[str, Any]]) -> None:
        """Save the buffer to the Tier-2 cache."""
        try:
            primary_target = self.viewer.scene_context.get_primary_target()
            mesh_path = getattr(primary_target, "path", "unknown")

            # Camera set hash
            camera_extrinsics = [
                camera.extrinsics.tobytes()
                for path, camera in eligible
            ]
            camera_set_hash = hashlib.md5(
                b"".join(sorted(camera_extrinsics))
            ).hexdigest().encode()

            # Weighting flags
            weighting_str = "_".join(
                f"{k}={int(v)}" for k, v in sorted(self._weighting_config.items())
            )
            weighting_str = f"{weighting_str}_interp={self._interpolation}"
            weighting_bytes = weighting_str.encode()

            element_type = primary_target.get_element_type()

            self.cache_manager.save_feature_buffer(
                mesh_path,
                camera_set_hash,
                model_id,
                buffer.provenance.get("compressor_kind", "nn"),
                buffer.provenance.get("compressor_dim", 32),
                weighting_bytes,
                buffer,
                element_type=element_type,
            )
        except Exception as e:
            print(f"[FeatureMeshManager] Cache save failed: {e}")

    def load_from_cache(self, model_id: str, compressor_kind: str = "nn",
                        compressor_dim: int = 32) -> bool:
        """
        Try to load a feature buffer from cache.

        Returns:
            True if successfully loaded, False otherwise.
        """
        try:
            primary_target = self.viewer.scene_context.get_primary_target()
            mesh_path = getattr(primary_target, "path", "unknown")

            # Reconstruct cache key (same as _cache_buffer)
            eligible, _ = self.prepare()
            camera_extrinsics = [
                camera.extrinsics.tobytes()
                for path, camera in eligible
            ]
            camera_set_hash = hashlib.md5(
                b"".join(sorted(camera_extrinsics))
            ).hexdigest().encode()

            weighting_str = "_".join(
                f"{k}={int(v)}" for k, v in sorted(self._weighting_config.items())
            )
            weighting_str = f"{weighting_str}_interp={self._interpolation}"
            weighting_bytes = weighting_str.encode()

            element_type = primary_target.get_element_type()

            buffer = self.cache_manager.load_feature_buffer(
                mesh_path,
                camera_set_hash,
                model_id,
                compressor_kind,
                compressor_dim,
                weighting_bytes,
                element_type=element_type,
            )

            if buffer is None:
                return False

            self.buffer = buffer
            self.query_engine = QueryEngine(buffer.features, buffer.valid)
            self._build_shader_state(buffer)
            primary_target.attach_feature_arrays(buffer)
            self._sync_colormap_from_ui()
            self.apply_colormap()
            self.viewer._update_array_selector()

            return True
        except Exception as e:
            print(f"[FeatureMeshManager] Cache load failed: {e}")
            return False

    def clear(self) -> None:
        """Clear the current buffer and detach from the mesh."""
        # Tear down the similarity overlay first (it references shader_state).
        self.disengage_overlay()

        self.buffer = None
        self.query_engine = None
        self.shader_state = None

        primary_target = self.viewer.scene_context.get_primary_target()
        if primary_target:
            primary_target.clear_feature_arrays()
            self.viewer._update_array_selector()

        # Disable (and exit) the Feature Select tool, since there's no buffer
        # left to query.
        feature_tool_action = getattr(self.main_window, 'feature_tool_action', None)
        if feature_tool_action is not None:
            if feature_tool_action.isChecked():
                feature_tool_action.setChecked(False)
                self.viewer.set_selected_3d_tool(None)
                
    def recolor_by_similarity(self, threshold: Optional[float] = None,
                              hover_id: Optional[int] = None) -> None:
        """
        Recolor the similarity overlay in place.

        ONE display representation for every product type: an [N] uint8 value
        colored through a 256-entry LUT. Meshes / point clouds write it into
        the SimilarityShader's disp texture (rendered by the engaged overlay
        actor); Gaussian splats write it into their display-channel SSBO
        (``set_similarity_scalars``). There is NO CPU color fallback — the
        similarity view requires the GPU LUT path (see engage_overlay).

        Args:
            threshold: When provided, render a live preview of the thresholded
                selection — only elements with raw similarity >= threshold (exactly
                what QueryEngine.select() would pick) are lit. When None, render
                the full similarity gradient.
            hover_id: optional transient prototype (element under the cursor) to
                fold into the query for live hover preview, without committing it.
        """
        if self.buffer is None or self.query_engine is None:
            return

        primary_target = self.viewer.scene_context.get_primary_target()
        if primary_target is None:
            return

        element_type = getattr(primary_target, "get_element_type", lambda: None)()

        # Mesh / point-cloud GPU colormap shader path: push the [N] uint8
        # display value into the shader's value texture (raw upload). The
        # overlay actor's fragment shader reads it — VTK never rebuilds colors.
        if element_type in ("face", "point"):
            if not self._shader_in_play():
                return
            try:
                disp = self.query_engine.display_scalars(threshold, hover_id=hover_id)
                self.shader_state.update_disp(disp)
            except Exception as e:
                self._disable_shader(f"disp update failed: {e}")
            return

        # Splat display channel: same scalars, product-side LUT.
        scalar_sink = getattr(primary_target, "set_similarity_scalars", None)
        if callable(scalar_sink):
            disp = self.query_engine.display_scalars(threshold, hover_id=hover_id)
            scalar_sink(disp)

    # Below-reject-floor elements in the multi-class view map to LUT row 0 —
    # a dark scrim so the unassigned region reads as dimmed, not exposed.
    MULTICLASS_SCRIM_RGB = (45, 45, 45)

    def recolor_by_class(self, prototypes_by_class, class_colors,
                         reject_threshold: Optional[float] = None,
                         hover_key=None, hover_id: Optional[int] = None) -> list:
        """Recolor the primary target by nearest-prototype class (multi-class).

        Reuses the exact binary-similarity display plumbing: the per-element
        value becomes a class index (0 = below the reject floor / uncovered)
        and the 256-entry LUT becomes the label palette, so the per-tick cost
        is identical to the gradient view (C matvecs + an [N]-byte upload).

        Args:
            prototypes_by_class: ``label_id -> [element_ids]`` committed clicks.
            class_colors: ``label_id -> (r, g, b)`` for the palette rows.
            reject_threshold: raw-cosine floor; below it elements show the scrim.
            hover_key / hover_id: transient hover prototype folded into
                ``hover_key``'s class for the live preview.

        Returns:
            keys: class keys in LUT-row order (row k+1 == keys[k]).
        """
        if self.buffer is None or self.query_engine is None:
            return []
        primary_target = self.viewer.scene_context.get_primary_target()
        if primary_target is None:
            return []

        disp, keys = self.query_engine.class_display_scalars(
            prototypes_by_class, reject_threshold, hover_key=hover_key,
            hover_id=hover_id)

        # Build and push the label palette (row 0 = scrim, row k+1 = class k).
        lut = np.zeros((256, 3), dtype=np.uint8)
        lut[0] = self.MULTICLASS_SCRIM_RGB
        for k, key in enumerate(keys):
            lut[k + 1] = class_colors.get(key, (255, 255, 255))
        self._push_display_lut(lut, primary_target)

        element_type = getattr(primary_target, "get_element_type", lambda: None)()
        if element_type in ("face", "point"):
            if self._shader_in_play():
                try:
                    self.shader_state.update_disp(disp)
                except Exception as e:
                    self._disable_shader(f"class disp update failed: {e}")
            return keys

        scalar_sink = getattr(primary_target, "set_similarity_scalars", None)
        if callable(scalar_sink):
            scalar_sink(disp)
        return keys

    # ------------------------------------------------------------------ #
    # On-demand colormap (binary similarity view)
    # ------------------------------------------------------------------ #
    def set_colormap(self, colormap_name: str) -> None:
        """Switch the similarity colormap live (mesh LUT texture + splat LUT +
        CPU fallback). A 256-entry write — the display values stay put, so the
        view recolors instantly without recomputing anything."""
        name = str(colormap_name or "plasma").lower()
        self._colormap_name = name
        if self.query_engine is not None:
            self.query_engine.set_colormap(name)
        self.apply_colormap()

    def _sync_colormap_from_ui(self) -> None:
        """Seed the colormap from the shared annotation-window dropdown.

        The 3D similarity LUT mirrors the 2D overlay colormap dropdown (bridged
        live via AnnotationWindow.overlayColormapChanged in QtMainWindow); this
        picks up its current value when a buffer is freshly baked/loaded.
        'None' has no 3D meaning — keep the current colormap then.
        """
        try:
            name = self.main_window.annotation_window.colormap_dropdown.currentText()
        except Exception:
            return
        if name and name != "None":
            self._colormap_name = name.lower()
            if self.query_engine is not None:
                self.query_engine.set_colormap(self._colormap_name)

    def apply_colormap(self) -> None:
        """(Re)push the current colormap into every LUT sink (binary mode)."""
        try:
            from coralnet_toolbox.MVAT.shaders.SimilarityShader import colormap_lut
            lut = colormap_lut(self._colormap_name)
        except Exception as e:
            print(f"[FeatureMeshManager] colormap '{self._colormap_name}' unavailable: {e}")
            return
        self._push_display_lut(lut)

    def _push_display_lut(self, lut: np.ndarray, primary_target=None) -> None:
        """Write a [K<=256, 3|4] uint8 LUT into the mesh shader texture and the
        primary product's display-channel LUT (splats), whichever exist."""
        if self.shader_state is not None:
            try:
                self.shader_state.set_lut(lut)
            except Exception:
                pass
        if primary_target is None:
            primary_target = self.viewer.scene_context.get_primary_target()
        sink = getattr(primary_target, "apply_feature_colormap", None)
        if callable(sink):
            sink(lut)

    # ------------------------------------------------------------------ #
    # Phase-2 shader plumbing
    # ------------------------------------------------------------------ #
    def _shader_in_play(self) -> bool:
        """True when the GPU shader path is enabled and its artifacts are built."""
        return bool(self.shader_enabled and self.shader_state is not None)

    def _disable_shader(self, reason: str) -> None:
        """Disable the GPU colormap shader for this session.

        There is no CPU fallback anymore — the similarity view requires the
        shader, so a failure also tears down any engaged overlay and tells the
        user why the heatmap disappeared.
        """
        print(f"[FeatureMeshManager] similarity shader disabled: {reason}")
        self.shader_enabled = False
        self.shader_state = None
        if self.overlay_engaged:
            self.disengage_overlay()
            try:
                self.main_window.status_bar.showMessage(
                    "Feature Select: GPU similarity shader failed — view disabled.",
                    6000)
            except Exception:
                pass

    def _build_shader_state(self, buffer: FeatureBuffer) -> None:
        """Build the GPU colormap textures once per bake / cache-load (best-effort)."""
        self.shader_state = None
        if not self.shader_enabled:
            return
        try:
            from coralnet_toolbox.MVAT.shaders.SimilarityShader import build_state
            self.shader_state = build_state(int(buffer.features.shape[0]))
        except Exception as e:
            # Don't kill the whole feature — just note we'll use uint8.
            print(f"[FeatureMeshManager] shader artifacts unavailable: {e}")
            self.shader_state = None

    # ------------------------------------------------------------------ #
    # Similarity-overlay engagement (3D analogue of the 2D work area)
    # ------------------------------------------------------------------ #
    def engage_overlay(self, mode: str = "binary") -> bool:
        """Engage the similarity overlay (Space in the 3D viewer).

        Mirrors the 2D tool's work-area engagement: creates the overlay actor
        (mesh/point cloud) or lights the splat display channel, and hands the
        shared annotation-window colormap dropdown + opacity slider to the 3D
        view (2D `_engage_colormap_controls` parity). Mutually exclusive with
        the 2D tool's work area — engaging here cancels it.

        Returns True when engaged (already-engaged counts as success).
        """
        if self._engaging:
            return self.overlay_engaged
        if self.overlay_engaged:
            return True
        if self.buffer is None or self.query_engine is None:
            try:
                self.main_window.status_bar.showMessage(
                    "Feature Select: bake mesh features first (Bake Mesh Features).",
                    5000)
            except Exception:
                pass
            return False

        primary_target = self.viewer.scene_context.get_primary_target()
        if primary_target is None:
            return False
        element_type = getattr(primary_target, "get_element_type", lambda: None)()
        if element_type in ("face", "point") and not self._shader_in_play():
            try:
                self.main_window.status_bar.showMessage(
                    "Feature Select: similarity view unavailable — the GPU "
                    "colormap shader failed to initialize.", 6000)
            except Exception:
                pass
            return False

        self._engaging = True
        try:
            # Either/or with the 2D tool: cancel its work area (and release the
            # shared controls) BEFORE we take them over.
            self._cancel_2d_engagement()

            self.overlay_engaged = True

            # Take over the shared colormap controls (2D-engage parity). The
            # dropdown is parked on the (empty) 2D feature overlay so changing
            # it can never re-show the hidden Z overlay.
            aw = getattr(self.main_window, "annotation_window", None)
            slider_opacity = 0.5
            if aw is not None:
                try:
                    aw._z_overlay.hide()
                    aw.set_active_colormap_overlay('feature')
                    if hasattr(aw, 'z_dynamic_button'):
                        aw.z_dynamic_button.setChecked(False)
                        aw.z_dynamic_button.setEnabled(False)
                    aw.colormap_dropdown.setEnabled(True)
                    aw.colormap_opacity_slider.setEnabled(True)
                    slider_opacity = aw.colormap_opacity_slider.value() / 255.0
                except Exception:
                    pass

            # Adopt whatever real colormap the dropdown already shows, then
            # point it at the mode's target ('None' in multiclass; the current
            # colormap in binary — first engage defaults to Plasma). Finally
            # push the LUT explicitly, covering the no-signal case where the
            # dropdown text didn't change.
            self._sync_colormap_from_ui()
            self.apply_mode_colormap(mode)
            self.apply_colormap()

            # Light the sinks. The label overlays are re-synced AFTER the
            # similarity actor is added so painted labels draw on top of the
            # heatmap in the translucent pass (2D stacking parity); their
            # opacity stays on the label transparency slider.
            if element_type in ("face", "point"):
                sync = getattr(self.viewer, "_sync_similarity_overlay_actor", None)
                if callable(sync):
                    sync()
                resync_labels = getattr(self.viewer, "_sync_all_label_overlay_actors", None)
                if callable(resync_labels):
                    resync_labels()
            else:
                engage_sink = getattr(primary_target, "set_display_engaged", None)
                if callable(engage_sink):
                    engage_sink(True, mix01=slider_opacity)

            if not self.overlay_engaged:
                # A shader failure during the actor sync disengaged us.
                return False

            # Baseline / committed view.
            self.recolor_by_similarity()
            try:
                self.viewer.plotter.render()
            except Exception:
                pass
            return True
        finally:
            self._engaging = False

    def disengage_overlay(self, release_controls: bool = True) -> None:
        """Tear the similarity overlay down (Space/Backspace with no prompts,
        tool deactivate, buffer clear, or the 2D tool engaging).

        ``release_controls=False`` skips returning the shared colormap controls
        (used when the 2D tool is about to immediately take them over).
        """
        if not self.overlay_engaged:
            return
        self.overlay_engaged = False

        remove = getattr(self.viewer, "_remove_similarity_overlay_actor", None)
        if callable(remove):
            try:
                remove()
            except Exception:
                pass
        resync_labels = getattr(self.viewer, "_sync_all_label_overlay_actors", None)
        if callable(resync_labels):
            try:
                resync_labels()
            except Exception:
                pass

        primary_target = self.viewer.scene_context.get_primary_target()
        disengage_sink = getattr(primary_target, "set_display_engaged", None)
        if callable(disengage_sink):
            try:
                disengage_sink(False)
            except Exception:
                pass

        if release_controls:
            # Mirror the 2D tool's _release_colormap_controls: back to the Z
            # overlay (left hidden by design), dropdown to 'None', controls
            # disabled when the image has no depth data.
            aw = getattr(self.main_window, "annotation_window", None)
            if aw is not None:
                try:
                    aw.set_active_colormap_overlay('z')
                    if aw.colormap_dropdown.currentText() != 'None':
                        aw.colormap_dropdown.setCurrentText('None')
                    else:
                        aw.update_overlay_colormap('None')
                    if getattr(aw, 'z_data_raw', None) is None:
                        aw.enable_z_visualization_controls(False)
                except Exception:
                    pass

        try:
            self.viewer.plotter.render()
        except Exception:
            pass

    def set_overlay_opacity(self, opacity01: float) -> None:
        """Live opacity for the engaged overlay (shared colormap slider bridge).

        Mesh/point cloud: the overlay actor's opacity. Splat: the display-
        channel mix over the SH shading. No-op when not engaged.
        """
        if not self.overlay_engaged:
            return
        opacity01 = float(max(0.0, min(1.0, opacity01)))

        actor = getattr(self.viewer, "_similarity_overlay_actor", None)
        if actor is not None:
            try:
                actor.GetProperty().SetOpacity(opacity01)
            except Exception:
                pass
        else:
            primary_target = self.viewer.scene_context.get_primary_target()
            ga = getattr(primary_target, "gaussian_actor", None)
            if ga is not None:
                try:
                    ga.set_display_mix(opacity01)
                except Exception:
                    pass
        try:
            self.viewer.plotter.render()
        except Exception:
            pass

    def apply_mode_colormap(self, mode: str) -> None:
        """Point the shared dropdown at 'None' (multiclass) or the colormap
        (binary) — the 3D analogue of the 2D tool's _apply_mode_colormap.

        In multiclass the LUT holds the label palette, so the dropdown reads
        'None'; back in binary it shows the similarity colormap again. No-op
        when the overlay isn't engaged.
        """
        if not self.overlay_engaged:
            return
        aw = getattr(self.main_window, "annotation_window", None)
        if aw is None:
            return
        target = 'None' if mode == "multiclass" else self._colormap_name.capitalize()
        try:
            if aw.colormap_dropdown.currentText() != target:
                aw.colormap_dropdown.setCurrentText(target)
        except Exception:
            pass

    def _cancel_2d_engagement(self) -> None:
        """Cancel the 2D FeatureSelectTool's work area + release its controls
        (either/or exclusivity — the 3D overlay is about to engage)."""
        try:
            tool2d = self.main_window.annotation_window.tools.get('feature_select')
        except Exception:
            tool2d = None
        if tool2d is None:
            return
        try:
            if getattr(tool2d, 'working_area', None) is not None:
                tool2d.cancel_working_area()
            release = getattr(tool2d, '_release_colormap_controls', None)
            if callable(release):
                release()
        except Exception:
            pass


class BakeFeatureDialog(QDialog):
    """Simple dialog to configure and launch a feature buffer bake."""

    def __init__(self, feature_mesh_manager, parent=None):
        super().__init__(parent)
        self.feature_mesh_manager = feature_mesh_manager
        self.setWindowTitle("Bake Mesh Features")
        self.setMinimumWidth(400)

        layout = QVBoxLayout()

        # Precondition check. Missing index maps are no longer blocking — the
        # bake computes them automatically in a pre-pass — so only cameras
        # without FEATURE maps are truly out.
        eligible, stats = feature_mesh_manager.prepare()
        precond_label = QLabel(
            f"Cameras: {stats['both']}/{stats['with_feature_map']} have both feature & index maps\n"
            f"Missing feature maps: {stats['missing_feature']}\n"
            f"Missing index maps: {stats['missing_index']} (computed automatically at bake)"
        )
        layout.addWidget(precond_label)

        if stats["with_feature_map"] == 0:
            reject_label = QLabel("Cannot proceed: no cameras have feature maps loaded.")
            layout.addWidget(reject_label)
            self.setMinimumHeight(200)
            cancel_btn = QPushButton("Close")
            cancel_btn.clicked.connect(self.reject)
            layout.addWidget(cancel_btn)
            self.setLayout(layout)
            return

        # Compressor config
        config_group = QGroupBox("Compressor")
        config_layout = QVBoxLayout()

        comp_layout = QHBoxLayout()
        comp_layout.addWidget(QLabel("Type:"))
        self.compressor_combo = QComboBox()
        self.compressor_combo.addItems(["nn", "pca"])
        self.compressor_combo.setItemData(
            0, "EXPERIMENTAL (default): trains a scene-specific autoencoder. Slower bake "
               "(trains on the main thread, see console log); may separate look-alike "
               "classes better than PCA at small dimensions.", Qt.ToolTipRole
        )
        self.compressor_combo.setToolTip("Compression method: nn (autoencoder) or pca (dimensionality reduction)")
        comp_layout.addWidget(self.compressor_combo)
        config_layout.addLayout(comp_layout)

        dim_layout = QHBoxLayout()
        dim_label = QLabel("Dimension:")
        dim_tooltip = (
            "Compressed feature size per element (default 32).\n"
            "Higher = separates look-alike classes better but uses more memory; "
            "lower = lighter but blurs fine distinctions."
        )
        dim_label.setToolTip(dim_tooltip)
        dim_layout.addWidget(dim_label)
        self.dim_spinbox = QSpinBox()
        self.dim_spinbox.setMinimum(1)
        self.dim_spinbox.setMaximum(512)
        self.dim_spinbox.setValue(32)
        self.dim_spinbox.setToolTip(dim_tooltip)
        dim_layout.addWidget(self.dim_spinbox)
        config_layout.addLayout(dim_layout)

        # Interpolation: how coarse patch features are upsampled to image res.
        interp_layout = QHBoxLayout()
        interp_label = QLabel("Interpolation:")
        interp_tooltip = (
            "How the coarse patch-grid features are sampled up to full image "
            "resolution when projecting onto the mesh.\n\n"
            "• Nearest — each surface point takes its closest patch's feature. "
            "Crisp boundaries, fastest, but blocky.\n"
            "• Bilinear — blends the 4 neighboring patches for smoother feature "
            "transitions across the surface (slightly slower).\n\n"
            "Element IDs are always matched nearest — only the features are "
            "interpolated."
        )
        interp_label.setToolTip(interp_tooltip)
        interp_layout.addWidget(interp_label)
        self.interp_combo = QComboBox()
        self.interp_combo.addItems(["nearest", "bilinear"])
        self.interp_combo.setToolTip(interp_tooltip)
        interp_layout.addWidget(self.interp_combo)
        config_layout.addLayout(interp_layout)

        # NN autoencoder training parameters (only shown when "nn" is selected).
        self.nn_params_widget = QWidget()
        nn_form = QFormLayout(self.nn_params_widget)
        nn_form.setContentsMargins(0, 0, 0, 0)

        self.nn_hidden_spinbox = QSpinBox()
        self.nn_hidden_spinbox.setRange(8, 4096)
        self.nn_hidden_spinbox.setValue(256)
        self.nn_hidden_spinbox.setToolTip(
            "Hidden layer width H of the encoder/decoder. Larger = more capacity.")
        nn_form.addRow("Hidden dim:", self.nn_hidden_spinbox)

        self.nn_epochs_spinbox = QSpinBox()
        self.nn_epochs_spinbox.setRange(1, 1000)
        self.nn_epochs_spinbox.setValue(30)
        self.nn_epochs_spinbox.setToolTip("Number of training epochs over the sampled patches.")
        nn_form.addRow("Epochs:", self.nn_epochs_spinbox)

        self.nn_lr_spinbox = QDoubleSpinBox()
        self.nn_lr_spinbox.setDecimals(5)
        self.nn_lr_spinbox.setRange(1e-5, 1.0)
        self.nn_lr_spinbox.setSingleStep(1e-4)
        self.nn_lr_spinbox.setValue(1e-3)
        self.nn_lr_spinbox.setToolTip("Adam learning rate.")
        nn_form.addRow("Learning rate:", self.nn_lr_spinbox)

        self.nn_beta_spinbox = QDoubleSpinBox()
        self.nn_beta_spinbox.setDecimals(2)
        self.nn_beta_spinbox.setRange(0.0, 100.0)
        self.nn_beta_spinbox.setSingleStep(0.1)
        self.nn_beta_spinbox.setValue(1.0)
        self.nn_beta_spinbox.setToolTip(
            "Weight of the cosine-preservation loss. Higher = prioritize keeping the "
            "compressed space angle-faithful (better for click-to-select queries).")
        nn_form.addRow("Cosine loss β:", self.nn_beta_spinbox)

        config_layout.addWidget(self.nn_params_widget)

        # Toggle NN params visibility with the compressor selection.
        self.compressor_combo.currentTextChanged.connect(self._on_compressor_changed)

        config_group.setLayout(config_layout)
        layout.addWidget(config_group)

        # Apply the initial visibility for the default selection.
        self._on_compressor_changed(self.compressor_combo.currentText())

        # Scope
        scope_group = QGroupBox("Camera Scope")
        scope_layout = QVBoxLayout()
        self.scope_combo = QComboBox()
        self.scope_combo.addItems(["all", "selected", "visible"])
        scope_layout.addWidget(self.scope_combo)
        scope_group.setLayout(scope_layout)
        layout.addWidget(scope_group)

        # Weighting options
        weight_group = QGroupBox("Confidence Weighting")
        weight_layout = QVBoxLayout()
        self.angle_check = QCheckBox("View Angle")
        self.angle_check.setChecked(True)
        weight_layout.addWidget(self.angle_check)

        self.dist_check = QCheckBox("Inverse Distance")
        self.dist_check.setChecked(True)
        weight_layout.addWidget(self.dist_check)

        self.edge_check = QCheckBox("Edge Guard")
        self.edge_check.setChecked(False)
        weight_layout.addWidget(self.edge_check)

        weight_group.setLayout(weight_layout)
        layout.addWidget(weight_group)

        # Buttons
        button_layout = QHBoxLayout()
        bake_btn = QPushButton("Bake")
        bake_btn.clicked.connect(self._on_bake_clicked)
        bake_btn.setToolTip("Compress and bake features onto the mesh with the specified parameters.")
        button_layout.addWidget(bake_btn)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        cancel_btn.setToolTip("Close without baking.")
        button_layout.addWidget(cancel_btn)

        layout.addLayout(button_layout)
        self.setLayout(layout)

    def _on_compressor_changed(self, kind: str):
        """Show NN training params only when the NN compressor is selected."""
        self.nn_params_widget.setVisible(kind == "nn")

    def _on_bake_clicked(self):
        """Launch the bake with current settings."""
        compressor_kind = self.compressor_combo.currentText()
        compressor_dim = self.dim_spinbox.value()
        scope = self.scope_combo.currentText()
        interpolation = self.interp_combo.currentText()

        nn_params = None
        if compressor_kind == "nn":
            nn_params = {
                "hidden_dim": self.nn_hidden_spinbox.value(),
                "epochs": self.nn_epochs_spinbox.value(),
                "lr": self.nn_lr_spinbox.value(),
                "beta": self.nn_beta_spinbox.value(),
            }

        # Update weighting config
        self.feature_mesh_manager._weighting_config = {
            "use_angle": self.angle_check.isChecked(),
            "use_inv_dist": self.dist_check.isChecked(),
            "use_edge_guard": self.edge_check.isChecked(),
        }

        # Launch the bake
        self.feature_mesh_manager.bake(
            compressor_kind=compressor_kind,
            compressor_dim=compressor_dim,
            scope=scope,
            interpolation=interpolation,
            nn_params=nn_params,
        )

        self.accept()
