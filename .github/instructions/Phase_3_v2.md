# Phase 3: The Conveyor Belt Navigation Engine

## Summary

The `ContextMatrixWidget` (Phase 2) displays N nearest cameras in a grid. Phase 3 adds keyboard-driven navigation through the full ranked list — a "sliding window" that scrubs forward/backward through neighbors without touching the mouse. The user's workflow: annotate in the main window, press `Ctrl+Right` to cycle context views through progressively farther cameras, `Ctrl+Down` to page through them N at a time.

---

## Current Architecture (What Exists)

### Conveyor Belt Data Source
- `MVATManager._reorder_cameras(reference_path)` computes a proximity-scored, sorted list of camera paths.
- Currently feeds `camera_grid.set_camera_order(ordered_paths)` which reorders thumbnails.
- Phase 2 added: `context_matrix.set_camera_order(context_paths, active_path)` which stores `self.ordered_context_paths`.

### Context Matrix State (from Phase 2)
- `self.ordered_context_paths: List[str]` — sorted camera paths (excluding active).
- `self.current_rank_offset: int` — starting index into the list (default 0).
- `self._get_visible_capacity() -> int` — returns `rows * cols`.
- `self._refresh_visible_canvases()` — slices `ordered_context_paths[offset:offset+N]` and loads them.

### Global Event Filter (`QtEventFilter.py`)
- Currently intercepts:
  - `Ctrl+Left/Right` → `annotation_window.cycle_annotations(±1)` (annotation cycling)
  - `Ctrl+Up/Down` → `label_window.cycle_labels(±1)` (label cycling)
  - `Alt+Up/Down` → `image_window.cycle_previous/next_image()` (image cycling)
- **Conflict:** `Ctrl+Left/Right` is already used for annotation cycling. We need different hotkeys or a context-aware dispatch.

---

## Target Architecture

### Hotkey Scheme (Revised to Avoid Conflicts)

| Hotkey | Current Use | New Use |
|--------|-------------|---------|
| `Ctrl+Left/Right` | Cycle annotations | Cycle annotations (unchanged) |
| `Ctrl+Up/Down` | Cycle labels | Cycle labels (unchanged) |
| `Alt+Up/Down` | Cycle images | Cycle images (unchanged) |
| **`Ctrl+Shift+Left/Right`** | (unassigned) | **Micro-step conveyor ±1** |
| **`Ctrl+Shift+Up/Down`** | (unassigned) | **Macro-step conveyor ±N** |

> **Rationale:** `Ctrl+Shift` combos are free and ergonomically close to the existing `Ctrl+Arrow` hotkeys. The shift modifier signals "context view" vs "annotation window" action.

---

## Implementation Steps

### Step 1: Add Conveyor State Management to `ContextMatrixWidget`

Extend the state machine from Phase 2:

```python
# In ContextMatrixWidget.__init__:
self.ordered_context_paths: List[str] = []
self.current_rank_offset: int = 0
```

### Step 2: Implement `shift_conveyor(delta)`

The core navigation method:

```python
def shift_conveyor(self, delta: int):
    """Shift the conveyor belt by delta positions.

    Args:
        delta: Number of positions to shift. Positive = forward (farther cameras),
               negative = backward (nearer cameras).
    """
    if not self.ordered_context_paths:
        return

    capacity = self._get_visible_capacity()
    total = len(self.ordered_context_paths)

    # Calculate new offset with clamping
    new_offset = self.current_rank_offset + delta
    new_offset = max(0, min(new_offset, max(0, total - capacity)))

    if new_offset == self.current_rank_offset:
        return  # No change (hit boundary)

    self.current_rank_offset = new_offset

    # Refresh the displayed canvases
    self._refresh_visible_canvases()

    # Update the rank indicator
    self._update_rank_label()
```

### Step 3: The Rank Indicator HUD

Update the label in the toolbar:

```python
def _update_rank_label(self):
    """Update the rank position indicator in the toolbar."""
    if not self.ordered_context_paths:
        self._rank_label.setText("—")
        return

    capacity = self._get_visible_capacity()
    total = len(self.ordered_context_paths)
    start = self.current_rank_offset + 1  # 1-based for display
    end = min(self.current_rank_offset + capacity, total)

    self._rank_label.setText(f"Neighbors {start}–{end} of {total}")
    self.rankIndicatorUpdated.emit(start, end, total)
```

### Step 4: Handle Layout Resizes Gracefully

When the user changes the layout (e.g., from 1×3 to 2×2), the capacity changes. The conveyor must adjust:

```python
def _rebuild_layout(self, rows, cols):
    """Rearrange the grid with the specified rows and columns."""
    # ... existing code from Phase 2 ...

    # After capacity change, re-clamp and refresh
    self.shift_conveyor(0)  # delta=0 triggers clamping and refresh
```

This is elegant: `shift_conveyor(0)` triggers the clamping logic which ensures the offset is valid for the new capacity, then refreshes the canvases.

### Step 5: Hotkey Integration in GlobalEventFilter

Extend `QtEventFilter.py` to intercept `Ctrl+Shift+Arrow` combinations:

```python
# In GlobalEventFilter.eventFilter(), inside the Ctrl+modifier block:

# After existing Ctrl+Left/Right (annotation cycling):
if event.modifiers() == (Qt.ControlModifier | Qt.ShiftModifier):
    # Context Matrix Conveyor Belt
    if event.key() == Qt.Key_Right:
        if hasattr(self.main_window, 'context_matrix'):
            self.main_window.context_matrix.shift_conveyor(1)
        return True
    if event.key() == Qt.Key_Left:
        if hasattr(self.main_window, 'context_matrix'):
            self.main_window.context_matrix.shift_conveyor(-1)
        return True
    if event.key() == Qt.Key_Down:
        if hasattr(self.main_window, 'context_matrix'):
            N = self.main_window.context_matrix._get_visible_capacity()
            self.main_window.context_matrix.shift_conveyor(N)
        return True
    if event.key() == Qt.Key_Up:
        if hasattr(self.main_window, 'context_matrix'):
            N = self.main_window.context_matrix._get_visible_capacity()
            self.main_window.context_matrix.shift_conveyor(-N)
        return True
```

**Important:** The `Ctrl+Shift` check must come **before** the plain `Ctrl` check in the event filter, because `event.modifiers() & Qt.ControlModifier` is true for both `Ctrl` and `Ctrl+Shift`.

### Step 6: Data Pipeline from MVATManager

When the active camera changes, `MVATManager._reorder_cameras()` already computes the sorted proximity list. We extend it to feed the context matrix:

```python
# In MVATManager._reorder_cameras():
def _reorder_cameras(self, reference_path, hide_distant_cameras=True):
    reference_camera = self.cameras.get(reference_path)
    if not reference_camera:
        return

    if reference_camera.is_orthographic:
        return

    camera_scores = []
    for path, camera in self.cameras.items():
        if path == reference_path:
            score = float('inf')
        else:
            score = self._calculate_camera_proximity_score(reference_camera, camera)

        if hide_distant_cameras and score == 0.0 and path != reference_path:
            continue
        camera_scores.append((path, score))

    camera_scores.sort(key=lambda x: x[1], reverse=True)
    ordered_paths = [p for p, s in camera_scores]

    # Feed CameraGrid (existing)
    self.camera_grid.set_camera_order(ordered_paths)

    # Feed ContextMatrixWidget (NEW)
    if hasattr(self, 'context_matrix') and self.context_matrix is not None:
        context_paths = [p for p in ordered_paths if p != reference_path]
        self.context_matrix.set_camera_order(context_paths, reference_path)
```

### Step 7: `set_camera_order` in ContextMatrixWidget

```python
def set_camera_order(self, ordered_paths: List[str], active_path: str):
    """Set the ranked list of context cameras.

    Args:
        ordered_paths: Camera paths sorted by proximity (nearest first).
                       Should already exclude the active camera.
        active_path: The currently active camera (for reference only).
    """
    # Filter out active path (defensive — caller should already do this)
    self.ordered_context_paths = [p for p in ordered_paths if p != active_path]
    self._active_camera_path = active_path

    # Reset conveyor to beginning
    self.current_rank_offset = 0

    # Display the first batch
    self._refresh_visible_canvases()
    self._update_rank_label()
```

### Step 8: Optimized Image Swapping

When shifting by 1 (micro-step), only 1 canvas needs to change. The others just need to be rearranged. However, since `BaseCanvas.load_visuals()` already checks `if canvas.current_image_path == path: continue`, the existing `update_context_cameras()` naturally skips reloading unchanged canvases.

For truly optimal 1-shift performance (shuffling existing canvases):

```python
def _refresh_visible_canvases(self):
    """Repaint canvases based on current conveyor state."""
    capacity = self._get_visible_capacity()
    if not self.ordered_context_paths:
        for i in range(capacity):
            self.canvas_pool[i].clear_scene()
        return

    target_paths = self.ordered_context_paths[
        self.current_rank_offset : self.current_rank_offset + capacity
    ]
    self.update_context_cameras(target_paths)
```

The `update_context_cameras` path-check optimization means:
- Shift by 1: ~1 canvas loads a new image, N-1 canvases are skipped (path unchanged).
- Shift by N: all N canvases load new images.

If even the 1-canvas load is too slow (large rasters), Phase 5's "thumbnail proxy" optimization will handle it.

---

## Edge Cases & Risks

- **Empty camera list:** If no cameras are loaded, `ordered_context_paths` is empty. All canvases show placeholder, hotkeys are no-ops. ✓

- **Fewer cameras than capacity:** If there are 2 cameras but 4 canvas slots, 2 slots show images and 2 show placeholder. Clamping prevents offset from going negative. ✓

- **Active camera changes rapidly:** If the user cycles images quickly (`Alt+Up/Down`), `_reorder_cameras` is called each time, resetting `current_rank_offset` to 0. This is correct — the neighbor list changes with each active camera.

- **Hotkey conflict with text fields:** The GlobalEventFilter already checks `isinstance(QApplication.focusWidget(), QLineEdit)` for delete keys. `Ctrl+Shift+Arrow` is unlikely to conflict with text editing, but we should add the same guard for safety.

- **Performance:** Loading N images on each shift. Mitigated by the path-check skip. For datasets with >100 cameras, the proximity computation in `_reorder_cameras` runs in O(N) which is fast.

---

## Validation Criteria

After Phase 3 is complete:

1. With 3 canvases visible (e.g., 1×3 layout), pressing `Ctrl+Shift+Right` shifts the conveyor by 1: the leftmost canvas drops off, the other two shift left, and a new camera appears on the right.
2. Pressing `Ctrl+Shift+Down` pages the entire grid (e.g., from cameras [1,2,3] to [4,5,6]).
3. The rank indicator shows "Neighbors 1–3 of 45" and updates on each shift.
4. Hitting the boundary (start or end of list) is a no-op (no crash, no wrap-around).
5. Changing the layout from 1×3 to 2×2 preserves the approximate position in the conveyor (clamped if needed).
6. Selecting a new active camera resets the conveyor to offset 0.
