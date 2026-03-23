# Plan Summary v2: Multi-View Context Canvas Architecture

## Overview

Transform the MVAT (Multi-View Annotation Tool) context viewing experience from a scrollable grid of static thumbnails into an interactive, synchronized multi-viewport workspace. The user will be able to inspect the same 3D scene from multiple camera angles simultaneously, with spatial markers, synchronized zoom/pan, and keyboard-driven navigation.

## Phase Dependencies

```
Phase 1 ─────► Phase 2 ─────► Phase 3
(BaseCanvas)   (Matrix UI)    (Conveyor)
    │              │              │
    ▼              ▼              ▼
Phase 4 ◄──── Phase 5 ◄───── (all above)
(Markers)     (Sync Engine)
    │
    ▼
Phase 6
(Annotations)
```

**Phases 1 and 6 can be started in parallel** because Phase 1 is a UI refactor (extracting BaseCanvas from AnnotationWindow) and Phase 6's first step is a data refactor (extracting AnnotationManager from AnnotationWindow). They touch different parts of the monolith.

---

## Phase 1: Extract `BaseCanvas` from `AnnotationWindow`

**Goal:** Create a lightweight, reusable `QGraphicsView` subclass that handles image display, zoom, pan, Z-channel visualization, and marker slots — without any annotation or tool logic.

**What changes:**
- New file: `QtBaseCanvas.py` (~400 lines)
- `AnnotationWindow` changes from `QGraphicsView` → `BaseCanvas` inheritance
- Navigation (zoom/pan) moves to BaseCanvas with native mouse events
- Z-channel visualization moves entirely to BaseCanvas
- Viewport control API added: `center_on_pixel()`, `set_zoom_level()`, `snap_to_target()`

**What doesn't change:**
- All tool dispatch stays in AnnotationWindow
- All annotation management stays in AnnotationWindow
- PanTool/ZoomTool continue to work (AnnotationWindow overrides route to them)
- Application behavior is identical post-refactor

**Key risk:** Z-channel methods reference `self.main_window`. BaseCanvas must not know about MainWindow. Solution: parameterize these methods and override in AnnotationWindow.

---

## Phase 2: Build the "Security Camera" Matrix

**Goal:** Create `ContextMatrixWidget` — a new dock containing a configurable grid (1×1 to 3×3) of `BaseCanvas` viewports for displaying nearby cameras.

**What's built:**
- New file: `MVAT/ui/QtContextMatrix.py`
- Object pool of pre-created BaseCanvas instances (up to 9)
- Layout engine: dropdown for preset configurations (1×1, 1×2, 2×1, 2×2, 1×3, 3×1)
- Auto-flow: layout adapts to dock aspect ratio on resize
- Data feed: receives ordered camera paths from MVATManager
- "Promote to Main" via double-click
- New dock: "Context" in MainWindow's PyQtAds layout

**Coexistence:** The existing `CameraGrid` remains unchanged for selection/highlighting. `ContextMatrixWidget` is an additional view.

---

## Phase 3: Conveyor Belt Navigation Engine

**Goal:** Add keyboard-driven scrolling through the ranked camera list — micro-step (±1 camera) and macro-step (±N cameras, page).

**Hotkeys:**
- `Ctrl+Shift+Right/Left`: micro-step ±1
- `Ctrl+Shift+Down/Up`: macro-step ±N (page)

**Conveyor state:** `current_rank_offset` into `ordered_context_paths`. Clamped to bounds. On layout resize, offset is re-clamped. On active camera change, offset resets to 0.

**HUD:** Rank indicator label "Neighbors 1–3 of 45" in toolbar.

**Optimization:** Path-check skip — when shifting by 1, only 1 canvas reloads, the rest are unchanged.

---

## Phase 4: Dual-Marker System

**Goal:** Project mouse cursor position and focal points from the main AnnotationWindow into context canvases as visual markers.

**Two marker types per BaseCanvas:**
- **Static Focal Marker** (crosshair): set by double-click or 3D viewer focal point. Persists until next focal point.
- **Dynamic Hover Marker** (circle): follows mouse movement in main window, real-time. Clears on mouse leave.

**Both use `ItemIgnoresTransformations`** for constant screen-pixel size regardless of zoom.

**Data flow:** Extends existing `MousePositionBridge` which already calculates ray projections via `Camera.project()` and `CameraRay`. Adds parallel routing to `ContextMatrixWidget.update_dynamic_markers()` alongside existing `CameraGrid.update_markers()`.

---

## Phase 5: Target-Locked Sync Engine

**Goal:** When the user zooms/pans in the main AnnotationWindow, all context canvases automatically zoom and center on the same 3D world point from their respective camera angles.

**Signal flow:** `AnnotationWindow.viewNavigated` → `MVATManager._on_main_view_navigated()` → unproject center pixel to 3D → project into each context camera → `ContextMatrixWidget.sync_to_targets()` → each `BaseCanvas.snap_to_target()`.

**Soft override:** User can manually pan/zoom a context canvas. Next main window navigation snaps it back. Zero state flags needed.

**Conveyor synergy:** New cameras from conveyor belt shifts load already synced to the current main view target.

**Performance:** Thumbnail proxy for initial load, full-res swap on zoom beyond threshold. 30ms throttle on sync updates.

**Toggle:** Sync button in Context Matrix toolbar (enabled by default).

---

## Phase 6: Annotation Visualization in Context Views

**Goal:** Display native annotations on context images and (eventually) project active drawing as "ghost" shapes.

**Architecture change:** Extract annotation data store into `AnnotationManager` — a central model accessible by any view.

**Migration strategy:** Property aliases in AnnotationWindow delegate to AnnotationManager. All existing code works unchanged. Gradual migration of methods.

**Context view features:**
- Read-only annotation overlays (non-interactive `QGraphicsItem`s)
- Live updates on create/delete/modify via manager signals
- Selection highlighting in context views
- Ghost projection (deferred to post-v1 — complex, requires per-vertex 3D projection during drawing)

---

## Files to Create/Modify

### New Files
| File | Phase | Purpose |
|------|-------|---------|
| `coralnet_toolbox/QtBaseCanvas.py` | 1 | Reusable viewport class |
| `coralnet_toolbox/MVAT/ui/QtContextMatrix.py` | 2 | Context matrix widget |
| `coralnet_toolbox/QtAnnotationManager.py` | 6 | Central annotation data store |

### Modified Files
| File | Phases | Changes |
|------|--------|---------|
| `QtAnnotationWindow.py` | 1, 6 | Inherit BaseCanvas; delegate data to AnnotationManager |
| `MVAT/managers/MVATManager.py` | 2–5 | Feed context matrix; sync engine; marker routing |
| `QtMainWindow.py` | 2, 6 | Create ContextMatrixWidget dock; create AnnotationManager |
| `QtEventFilter.py` | 3 | Add Ctrl+Shift+Arrow hotkeys |
| `MVAT/__init__.py` | 2 | Export ContextMatrixWidget |
| `Layout/QtDockWrapper.py` | — | No changes needed |

### Untouched Files (Verified)
- `MVAT/core/Camera.py` — projection math already works
- `MVAT/core/Ray.py` — ray math already works
- `MVAT/core/Marker.py` — stays for AnnotationWindow focal marker
- `MVAT/ui/QtCameraGrid.py` — untouched, coexists with new Context dock
- `MVAT/ui/QtMVATViewer.py` — untouched, 3D viewer
- `Tools/*.py` — untouched
- `Annotations/*.py` — untouched (graphics item creation stays per-annotation)
- `Rasters/*.py` — untouched

---

## Risk Summary

| Risk | Mitigation |
|------|-----------|
| BaseCanvas extraction breaks AnnotationWindow | Property/method inheritance; run full test suite after each migration |
| Z-channel `main_window` coupling | Parameterize methods; AnnotationWindow overrides inject references |
| Memory from N full-res images | Thumbnail proxy with on-demand full-res swap |
| Hotkey conflicts | Use `Ctrl+Shift+Arrow` (currently unassigned) |
| PyQtAds dock events | Use `resizeEvent` heuristic instead of dock location signals |
| AnnotationManager migration | Property aliases first; gradual method migration |
| Scene recreation destroys markers | `_on_scene_cleared()` hook re-initializes markers |
