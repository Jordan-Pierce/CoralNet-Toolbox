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

        # Phase-2 GPU colormap shader (SimilarityShader). Bypasses VTK's per-change
        # color-buffer rebuild (the ~125 ms-at-4M / seconds-at-76M cost — confirmed
        # by profiling: a no-change render is ~5 ms, a post-recolor render ~125 ms)
        # by colormapping in a fragment shader from a small value texture, with
        # ScalarVisibilityOff so VTK never rebuilds the cell colors. If the shader
        # can't install, we fall back to direct RGB cell colors (slower but works).
        self.shader_enabled = True
        self.shader_state = None

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

        # Collect cameras
        if scope == "selected":
            cameras_to_check = [
                (p, self.mvat_manager.cameras[p])
                for p in getattr(self.mvat_manager, "selected_camera_paths", [])
                if p in self.mvat_manager.cameras
            ]
        elif scope == "visible":
            cameras_to_check = list(self.mvat_manager.cameras.items())  # TODO: filter by visibility
        else:  # "all"
            cameras_to_check = list(self.mvat_manager.cameras.items())

        for path, camera in cameras_to_check:
            raster = getattr(camera, "_raster", None)
            if raster is None:
                continue

            # Existence check ONLY — do NOT touch raster.has_feature_map() /
            # raster.index_map here: those read the lazy properties, which pull
            # every map off disk through the LRU (decompressing all N cameras'
            # maps just to count them — the dialog-open lag). The in-memory
            # array OR a configured disk path is sufficient to mark eligibility;
            # the bake worker loads the actual data later, on its own thread.
            has_fm = (getattr(raster, "_feature_map", None) is not None
                      or bool(getattr(raster, "feature_map_path", None)))
            has_im = (getattr(raster, "_index_map", None) is not None
                      or bool(getattr(raster, "index_map_path", None)))

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

    def bake(self, compressor_kind: str = "nn", compressor_dim: int = 32,
             scope: str = "all", interpolation: str = "nearest",
             nn_params: dict = None) -> None:
        """
        Launch the bake worker on a background thread.

        Args:
            compressor_kind: "nn" | "pca".
            compressor_dim: Target D.
            scope: "all" | "selected" | "visible".
            interpolation: "nearest" | "bilinear" — how the coarse patch
                features are sampled up to image resolution.
            nn_params: optional dict of NN autoencoder hyperparameters
                (hidden_dim, epochs, lr, beta) when compressor_kind == "nn".
        """
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

            # Repopulate the array dropdown
            self.viewer._update_array_selector()

            # Status
            n_valid = int(np.sum(buffer.valid))
            msg = f"Baked {n_valid}/{buffer.features.shape[0]} elements. Query ready."
            self.mvat_manager.main_window.status_bar.showMessage(msg, 5000)
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
            self.viewer._update_array_selector()

            return True
        except Exception as e:
            print(f"[FeatureMeshManager] Cache load failed: {e}")
            return False

    def clear(self) -> None:
        """Clear the current buffer and detach from the mesh."""
        self.buffer = None
        self.query_engine = None
        self.shader_state = None

        primary_target = self.viewer.scene_context.get_primary_target()
        if primary_target:
            # Tear the shader off the live mesh actor before the array detaches.
            actor = self._get_mesh_actor(primary_target)
            if actor is not None:
                self.uninstall_shader(actor)
            primary_target.clear_feature_arrays()
            self.viewer._update_array_selector()

        # Disable (and exit) the Feature Select tool, since there's no buffer
        # left to query.
        feature_tool_action = getattr(self.main_window, 'feature_tool_action', None)
        if feature_tool_action is not None:
            if feature_tool_action.isChecked():
                feature_tool_action.setChecked(False)
                self.viewer.set_selected_3d_tool(None)
                
    def recolor_by_similarity(self, sim: Optional[np.ndarray] = None,
                              threshold: Optional[float] = None,
                              hover_id: Optional[int] = None) -> None:
        """
        Recolor the primary target by similarity in place.

        Works across product types: meshes write a per-face "Similarity" RGB
        array (or drive the GPU colormap shader), point clouds write a per-point
        "Similarity" RGB array, and Gaussian splats push the colours straight to
        the splat SH. The per-element display colours are identical in every
        case — only the sink differs (``product.set_similarity_colors``).

        Args:
            sim: [N] float32 similarity scores, or None to use query_engine.similarity().
            threshold: When provided, render a live preview of the thresholded
                selection — only elements with raw similarity >= threshold (exactly
                what QueryEngine.select() would pick) are lit. When None, render
                the full similarity gradient.
            hover_id: optional transient prototype (element under the cursor) to
                fold into the query for live hover preview, without committing it.
        """
        if self.buffer is None or (sim is None and self.query_engine is None):
            return

        primary_target = self.viewer.scene_context.get_primary_target()
        if primary_target is None:
            return

        element_type = getattr(primary_target, "get_element_type", lambda: None)()
        # Only meshes (gl_PrimitiveID) and point clouds (injected gl_VertexID) are
        # rendered through a VTK mapper the colormap shader can replace. Gaussian
        # splats are drawn by their own ModernGL geometry shader (no VTK mapper to
        # patch — maybe_install_shader is never even called for them), so they MUST
        # take the direct-RGB path below, which pushes the colours into the splat
        # SH via set_similarity_colors. Routing splats through the shader path
        # silently no-ops (the disp texture nothing reads gets updated), which is
        # why similarity never appeared on splats.
        is_shader_capable = element_type in ("face", "point")

        # Both meshes and point clouds use the GPU colormap shader path:
        # push the [N] uint8 display value into the shader's value texture (raw
        # upload). ScalarVisibilityOff means VTK never rebuilds the color
        # buffer — the render stays ~5 ms. Splats take the direct-RGB path below.
        if sim is None and is_shader_capable and self._shader_in_play():
            try:
                disp = self.query_engine.display_scalars(threshold, hover_id=hover_id)
                self.shader_state.update_disp(disp)
            except Exception as e:
                self._disable_shader(f"disp update failed: {e}")
            else:
                return
            # On failure the shader is disabled → fall through to the RGB path.

        # Direct-RGB path: used by Gaussian Splats (straight to SH) or as a fallback.
        if sim is None:
            colors = self.query_engine.display_colors(threshold, hover_id=hover_id)
        else:
            self.query_engine._ensure_colormap_lut()
            disp = QueryEngine._disp_unit01_numpy(np.asarray(sim, dtype=np.float32),
                                                  threshold)
            ci = np.clip(np.rint(disp * 255.0), 0, 255).astype(np.int64)
            colors = self.query_engine._cmap_np[ci]

        # Dispatch to the product's similarity sink (mesh cell_data / point_data
        # in place, or the splat SH push).
        sink = getattr(primary_target, "set_similarity_colors", None)
        if callable(sink):
            sink(colors)

    # ------------------------------------------------------------------ #
    # Phase-2 shader plumbing
    # ------------------------------------------------------------------ #
    def _shader_in_play(self) -> bool:
        """True when the GPU shader path is enabled and its artifacts are built."""
        return bool(self.shader_enabled and self.shader_state is not None)

    def _disable_shader(self, reason: str) -> None:
        """Permanently fall back to the uint8 path for this session."""
        print(f"[FeatureMeshManager] shader disabled → uint8 fallback: {reason}")
        self.shader_enabled = False
        self.shader_state = None

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

    def _get_mesh_actor(self, primary_target):
        """Resolve the live VTK actor for the primary mesh product, if any."""
        try:
            product_actors = getattr(self.viewer, "_product_actors", {})
            return product_actors.get(getattr(primary_target, "product_id", None))
        except Exception:
            return None

    def maybe_install_shader(self, actor, product) -> None:
        """
        Install the similarity shader on a freshly built actor.

        Called from the viewer after add_mesh()/add_points() whenever the
        Similarity array is active. No-op unless the shader is in play. Any
        failure flips to the uint8 fallback for the rest of the session.
        """
        if actor is None or not self._shader_in_play():
            return

        element_type = getattr(product, "get_element_type", lambda: None)()
        # Splats are excluded: they have no VTK mapper for the colormap shader to
        # replace (they recolor via the SH direct-RGB path), and render_scene
        # short-circuits them before this is ever called.
        if element_type not in ("face", "point"):
            return

        sa = getattr(product, "selected_array", None)
        if sa != "Similarity":
            return
        try:
            from coralnet_toolbox.MVAT.shaders.SimilarityShader import (
                install_similarity_shader,
            )
            install_similarity_shader(actor, self.shader_state, element_type=element_type)
        except Exception as e:
            self._disable_shader(f"install failed: {e}")

    def uninstall_shader(self, actor) -> None:
        """Remove the shader from a mesh actor (tool deactivate / array switch)."""
        if actor is None:
            return
        try:
            from coralnet_toolbox.MVAT.shaders.SimilarityShader import (
                uninstall_similarity_shader,
            )
            uninstall_similarity_shader(actor)
        except Exception as e:
            print(f"[FeatureMeshManager] shader uninstall failed: {e}")


class BakeFeatureDialog(QDialog):
    """Simple dialog to configure and launch a feature buffer bake."""

    def __init__(self, feature_mesh_manager, parent=None):
        super().__init__(parent)
        self.feature_mesh_manager = feature_mesh_manager
        self.setWindowTitle("Bake Mesh Features")
        self.setMinimumWidth(400)

        layout = QVBoxLayout()

        # Precondition check
        eligible, stats = feature_mesh_manager.prepare()
        precond_label = QLabel(
            f"Cameras: {stats['both']}/{stats['with_feature_map']} have both feature & index maps\n"
            f"Missing feature maps: {stats['missing_feature']}\n"
            f"Missing index maps: {stats['missing_index']}"
        )
        layout.addWidget(precond_label)

        if stats["both"] == 0:
            reject_label = QLabel("Cannot proceed: no cameras have both maps loaded.")
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
