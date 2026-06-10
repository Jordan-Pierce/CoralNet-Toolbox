# MVAT Index-Map Generation Performance Plan — Executable Steps

Goal: reduce wall-clock time of index-map generation (especially Native budget).
Measured baseline (91 cameras, Native): **6.31s total** = 3.02s cache writes (zlib)
+ 1.85s render/readback loop + ~1.4s orchestration overhead. GPU rasterize itself
is ~0.01ms/cam; the bottlenecks are compression, synchronous readback, and
serialization of render → save phases. Target: ~1.5–2s on the same dataset.

## Ground rules for the executing agent

1. **One step = one commit.** Complete a step, run its Verify block, commit with the
   given message, then move on. Never combine steps.
2. After every step, from the repo root:
   ```
   python -m py_compile <each file edited in the step>
   ```
   A step is only done when py_compile exits 0 and the step's grep checks pass.
3. Do NOT reformat or "clean up" code you are not told to touch.
4. Work on the current branch (`dev_mvat`).
5. Read the target method fully before editing. If a described anchor doesn't
   exist, STOP and re-read — do not improvise.

Files referenced (paths relative to repo root):

- `coralnet_toolbox/MVAT/managers/VisibilityManager.py`   (VM)
- `coralnet_toolbox/MVAT/workers/VisibilityWorker.py`     (VW)
- `coralnet_toolbox/MVAT/utils/IndexMapCodec.py`          (CODEC)
- `coralnet_toolbox/MVAT/managers/CacheManager.py`        (CM)
- `coralnet_toolbox/MVAT/shaders/gpu_interop.py`          (GPU)
- `requirements.txt`

---

## Step 1 — Fix the always-on debug logging + handler leak (VM)

**Why:** `compute_batch_visibility_moderngl` adds a NEW `logging.FileHandler` on
*every call* (one per chunk — handlers accumulate, each log line is written N
times), and `os.environ.setdefault('VISIBILITY_DEBUG', '1')` forces DEBUG logging
ON by default. This burns several formatted-string + file writes per camera in
the hot loop.

**Edits** in `compute_batch_visibility_moderngl` (near the top of the method):

1. Change `os.environ.setdefault('VISIBILITY_DEBUG', '1')` → default `'0'`.
2. Only create/attach the `FileHandler` when `VISIBILITY_DEBUG == '1'`, AND only
   if no handler is already attached for that file (module-level guard flag).
   When debug is off, do not attach any handler and do not call
   `logger.setLevel(logging.DEBUG)`.
3. The per-camera `_stats` min/max bookkeeping and all
   `if logger.isEnabledFor(logging.DEBUG)` blocks stay as-is (they become cheap
   once DEBUG is off).

**Verify:** py_compile; `grep -n "VISIBILITY_DEBUG', '0'" coralnet_toolbox/MVAT/managers/VisibilityManager.py`
returns one hit; launching the worker twice must not duplicate lines in
`visibility_timing_debug.log` when debug is enabled.

**Commit:** `perf(MVAT): make visibility debug logging opt-in, fix handler leak`

---

## Step 2 — Readback micro-optimizations (VM)

**Why:** Per camera the CPU path does `ctx.finish()` (2–6ms redundant stall —
`fbo.read` already synchronizes), then `fbo.read()` (allocates a bytes object),
then `np.frombuffer(raw) - 1` (allocates a second full-res array). ~6–17ms/cam
of avoidable memory traffic.

**Edits** in the per-camera loop of `compute_batch_visibility_moderngl`:

1. Delete the explicit `ctx.finish()` in the "GPU sync" block (keep the timing
   variable, set `t_gpu_sync = 0.0`).
2. Replace the read + decode:
   ```python
   raw = fbo_to_read.read(components=1, dtype='i4')
   ...
   crop_index_map = np.frombuffer(raw, dtype=np.int32).reshape(crop_h, crop_w) - 1
   ```
   with a direct read into the final retained array, decoded in place:
   ```python
   crop_index_map = np.empty((crop_h, crop_w), dtype=np.int32)
   fbo_to_read.read_into(crop_index_map, components=1, dtype='i4')
   np.subtract(crop_index_map, 1, out=crop_index_map)
   ```
   (The array is retained in the result dict, so one allocation per camera is
   required; the bytes copy and the subtract temporary are not.)

**Verify:** py_compile; `grep -n "read_into(crop_index_map" coralnet_toolbox/MVAT/managers/VisibilityManager.py`
returns one hit; `grep -n "frombuffer(raw" ...VisibilityManager.py` returns nothing
in the index-map path (the depth read may still use frombuffer).

**Commit:** `perf(MVAT): zero-copy FBO readback with in-place ID decode`

---

## Step 3 — Replace zlib cache codec with zstd (CODEC, CM, requirements.txt)

**Why:** `np.savez_compressed` is single-shot zlib (~30–50 MB/s): **~0.75s per
native map**, 3.02s of the 6.31s run. zstd level 3 is ~10–20× faster at equal or
better ratio. All archive I/O already goes through `IndexMapCodec`, so the change
is contained (consumers: CM and `coralnet_toolbox/Rasters/QtRaster.py`, both via
the codec functions).

**Edits:**

1. `requirements.txt`: add `zstandard`.
2. CODEC `save_index_map_archive`: when `compress=True`, write the new format:
   serialize the existing `payload` dict with `np.savez` (UNcompressed) into an
   `io.BytesIO`, then `zstandard.ZstdCompressor(level=3).compress(buf.getvalue())`,
   write to `temp_path`, `os.replace` as today. When `compress=False`, keep
   plain `np.savez` to disk unchanged. Keep atomic temp-file + replace behavior.
3. CODEC `load_index_map_archive`: sniff the first 4 bytes of the file. If they
   are the zstd magic `b"\x28\xb5\x2f\xfd"`, decompress to `BytesIO` and
   `np.load` from it; otherwise `np.load` the path directly (legacy `.npz` files
   keep working). The rest of the function is unchanged.
4. If `zstandard` fails to import, fall back to today's
   `np.savez_compressed` / direct `np.load` (wrap import in try/except at module
   top, `HAS_ZSTD` flag).
5. CM: no path changes needed (the file is still named `.npz`; only its bytes
   differ, and the loader sniffs). Do NOT change `get_cache_path`.

**Verify:** py_compile both files; round-trip test from repo root:
```
python -c "import numpy as np, tempfile, os; from coralnet_toolbox.MVAT.utils.IndexMapCodec import save_index_map_archive, load_index_map_archive; p=os.path.join(tempfile.gettempdir(),'t.npz'); m=np.random.randint(-1,5000,(1024,768)).astype(np.int32); save_index_map_archive(p,m,np.unique(m[m>=0])); r=load_index_map_archive(p); assert np.array_equal(r['index_map'],m); print('OK')"
```

**Commit:** `perf(MVAT): zstd index-map cache codec (~10-20x faster writes), legacy npz fallback`

---

## Step 4 — Overlap cache writes with rendering (VW)

**Why:** The chunk loop is serial: render chunk → **block** in
`save_to_disk_task` → next chunk. Compression releases the GIL, so saves can run
concurrently with the next chunk's render/readback, turning render+save time
from a sum into a max. Also: the "measured first camera" is saved inline and
single-threaded (~0.75s serial) inside `_process_result_pipeline`, and every
chunk ends with a `gc.collect()` that can cost 100s of ms on a multi-GB heap.

**Edits** in `VisibilityWorker.run`:

1. Create ONE persistent `ThreadPoolExecutor(max_workers=self.n_workers)` before
   the chunk loop; collect all save futures in a list.
2. Change `save_to_disk_task` to SUBMIT each camera's save to that pool and
   return immediately (keep the per-save logging inside the submitted fn). Do
   not wait per chunk. To bound RAM, before submitting a new chunk's saves, if
   the number of not-yet-done futures exceeds `4 * self.n_workers`, wait until
   it drops (`concurrent.futures.wait(..., return_when=FIRST_COMPLETED)` loop).
3. In `_process_result_pipeline`, route the first measured camera's
   `save_visibility` call through the same pool instead of calling it inline.
4. After the chunk loop (inside the existing `try`, before the `finally`), wait
   for all futures, then log the existing cache summary using the completion
   time of the last future.
5. Delete the per-chunk `gc.collect()` in "--- E. FLUSH SYSTEM RAM ---" (keep
   the `del chunk_results` / `del batch_results`). NOTE: saves now hold
   references to each result's `index_map` until written, so peak RAM ≈ chunk +
   in-flight saves; the future cap in (2) bounds this.

**Verify:** py_compile; `grep -n "FIRST_COMPLETED" coralnet_toolbox/MVAT/workers/VisibilityWorker.py`
returns a hit; `grep -n "gc.collect()" ...VisibilityWorker.py` no longer matches
inside the chunk loop (the one in `_measure_actual_camera_footprint` may stay).

**Commit:** `perf(MVAT): overlap index-map cache writes with rendering via persistent pool`

---

## Step 5 — Double-buffered PBO readback pipeline (VM)

**Why:** Even after Step 2, `read_into` stalls the CPU until the GPU finishes the
DMA (~8–14ms/cam). Reading into a moderngl `Buffer` (a PBO) is asynchronous: we
can kick off camera *i*'s transfer, then decode camera *i−1* while it flies.

**Edits** in `compute_batch_visibility_moderngl` (apply ONLY when
`compute_depth_map` is False — the batch path; when True, keep the Step-2
synchronous path):

1. After the context/fbo setup, allocate two persistent PBOs sized for the
   largest camera in the batch:
   `max_bytes = 4 * max(w*h for (_,_,_,w,h) in camera_params_list)` (note: crops
   are ≤ render size ≤ native size), `pbos = [ctx.buffer(reserve=max_bytes) for _ in range(2)]`,
   stored in `mgl_context` so repeated calls reuse them; release in the worker's
   existing `finally` block alongside the FBO cache.
2. Restructure the loop into a 1-deep pipeline:
   - After rendering camera *i*: `fbo.read_into(pbos[i % 2], components=1, dtype='i4')`
     and push `(i, crop geometry, fbo metadata, result placeholders)` onto a
     1-slot `pending` variable.
   - Before issuing camera *i*'s read, if `pending` holds camera *i−1*: drain it —
     `crop_index_map = np.empty((ph, pw), np.int32)`;
     `pbos[(i-1) % 2].read_into(crop_index_map, size=ph*pw*4)`;
     `np.subtract(crop_index_map, 1, out=crop_index_map)`; then run the existing
     paste/upsample/normalize/append logic for that camera.
   - Cameras that take the OFF_SCREEN early-continue or the CUDA-GL
     (`gpu_index_positions`) path must drain `pending` first to preserve result
     ordering — **results must stay in camera order**; build `results` as a
     pre-sized list and assign by index rather than `append`.
   - After the loop, drain the final `pending` entry.
3. Keep all existing timing accumulators compiling (set drained-stage timings on
   the camera being drained; exact attribution between overlapping stages does
   not need to be precise).

**Verify:** py_compile; `grep -n "reserve=max_bytes\|pending" coralnet_toolbox/MVAT/managers/VisibilityManager.py`
shows the pipeline; functional check — run any existing MVAT visibility flow (or
`compute_visibility_from_scene` on a small mesh) and confirm index maps are
identical to pre-change output for ≥3 cameras (`np.array_equal`).

**Commit:** `perf(MVAT): pipeline FBO readback through double-buffered PBOs`

---

## Step 6 — Distorted-camera path: stop re-measuring and re-registering (VW, GPU)

**Why:** Two per-iteration costs in the distortion path:
(a) `_pbo_cuda_readback` creates, CUDA-registers, maps, unmaps, unregisters and
deletes a PBO **per camera** (`cudaGraphicsGLRegisterBuffer` is ~ms each);
(b) the VRAM-budget probe (sync + `empty_cache` + `reset_peak_memory_stats` +
4-map test warp) reruns for every chunk × distortion group.

**Edits:**

1. GPU `_pbo_cuda_readback`: accept an optional `cache` dict (stored in
   `mgl_context`). Key by buffer size; keep `(pbo, registered_resource)` alive
   across calls and only create/register on first use for that size. Map/unmap
   per call (cheap); unregister/delete everything in a new
   `release_pbo_cache(gl, cudart, cache)` helper that VM's cleanup path calls.
2. VW distortion block: hoist the per-group VRAM measurement out of the chunk
   loop — cache `actual_vram_per_map` per group key
   (`dist_coeffs_bytes` / callable id) in a dict created before the chunk loop,
   measure on first encounter only, reuse afterwards (re-derive
   `vram_batch_size` from current `mem_get_info()` each chunk, that part is
   cheap).

**Verify:** py_compile both; `grep -n "release_pbo_cache" coralnet_toolbox/MVAT/shaders/gpu_interop.py coralnet_toolbox/MVAT/managers/VisibilityManager.py`
hits both files; `grep -n "actual_vram_per_map" coralnet_toolbox/MVAT/workers/VisibilityWorker.py`
shows the cached lookup.

**Commit:** `perf(MVAT): cache CUDA-GL PBO registration and per-group VRAM probe`

---

## Benchmarking (after Steps 3, 4, and 6)

Set `VISIBILITY_DEBUG=1`, run the visibility computation on the same dataset at
Native budget, and compare in the log: `[VISIBILITY PROCESSING WALL TIME]`,
`Cached N/N maps to disk in X s`, and the moderngl batch `Total Time`. Record
the three numbers in the commit message of the step that produced them.

## Out of scope (future ideas, do NOT implement now)

- Render-on-demand with RAM LRU instead of a disk cache (rendering is now ~20ms,
  cheaper than a cache load; would need a persistent GL context + warp-on-load).
- For budgeted (non-native) runs: cache the render-res map + `scale_factor` and
  NEAREST-upsample at load instead of caching the upsampled native map.
- Row-delta transform before zstd (better ratio; adds `cumsum` on load).
