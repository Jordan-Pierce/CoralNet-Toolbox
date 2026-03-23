Here is the detailed execution plan for the final core piece, **Phase 5: The "Target-Locked" Sync Engine**.

This is the feature that bridges the gap between a "neat UI" and a "professional workflow." It requires translating 2D navigation in one image into 3D spatial navigation across multiple different camera angles.

### Step 1: The Sync Trigger & The Toggle UI

First, we need to know when the user moves the main view, and we need to give them the ability to turn this feature off.

**1. The UI Toggle:**

* Add a "Sync" or "Link" toggle button to the top toolbar of the `ContextMatrixWidget` (perhaps a chain-link icon).
* By default, it is checked (Target-Lock Enabled).

**2. The Trigger:**

* In your new `AnnotationWindow` (which now inherits from `BaseCanvas`), ensure that every time a pan completes or a zoom occurs, it emits the `viewNavigated(center_x, center_y, zoom_factor)` signal.
* `center_x` and `center_y` represent the exact pixel currently in the dead-center of the `AnnotationWindow`'s viewport.

### Step 2: The Spatial Math Pipeline

When the `viewNavigated` signal fires, we cannot simply tell the context views to "go to pixel 500, 500"—because the cameras are looking from entirely different angles. Pixel 500, 500 in Camera A might be sky in Camera B. We must route the navigation through the 3D world.

**The Logic (handled in `MVATManager` or a dedicated Sync Controller):**

1. **Check Toggle:** If the Target-Lock toggle is OFF, ignore the signal entirely.
2. **2D to 3D:** Take the `(center_x, center_y)` from the main camera. Use the image's Z-channel (or the scene's median depth) to cast a ray and find the exact 3D world coordinate the user is looking at.
3. **3D to 2D:** Loop through the currently visible cameras in the `ContextMatrixWidget`. Project that 3D coordinate into each context camera's pixel space to find their respective `(target_x, target_y)`.
4. **Command the Matrix:** Send a dictionary of these targets and the `zoom_factor` to the `ContextMatrixWidget`.

### Step 3: The Canvas Execution (`snap_to_target`)

The `ContextMatrixWidget` receives the targets and commands its individual canvases to move.

**Inside `BaseCanvas`:**

* Create a method: `snap_to_target(target_x, target_y, relative_zoom)`.
* **Zooming:** The canvas updates its transformation matrix to match the `relative_zoom`. (If the main window is zoomed in 3x, the context window scales itself 3x).
* **Centering:** The canvas uses a built-in Qt function (like `centerOn(QPointF(target_x, target_y))`) to instantly adjust its scrollbars so the target pixel is perfectly centered in its viewport.

### Step 4: The "Soft Override" (Independent Navigation)

You explicitly requested that if a user pans or zooms an *individual* canvas, it updates independently, but snaps back to the main target when the user moves the `AnnotationWindow` again.

**The brilliant part of this architecture is that this requires zero extra logic.**

* Because you extracted `BaseCanvas` in Phase 1, every context view inherently knows how to handle its own mouse wheel and right-click drag.
* If a user goes to Context View 2 and pans around, it just moves its own scrollbars. No "lock" is broken; no complex boolean states are flipped.
* The *very next time* the user pans or zooms in the main `AnnotationWindow`, the `viewNavigated` signal fires again, the 3D math runs, and `snap_to_target` is called on all context views. Context View 2 is instantly commanded to center back on the main target. It creates a flawless, intuitive "temporary override."

### Step 5: Conveyor Belt Synergy

This step ties Phase 3 (the hotkeys) and Phase 5 (target-lock) together.

When the user presses `Ctrl + Right` to shift the conveyor belt, a new camera slides into the matrix.

* Before the `ContextMatrixWidget` displays this newly loaded camera, it automatically calls the 3D math pipeline to find the *current* main window target.
* It calls `snap_to_target` on the new canvas *before* it becomes visible.
* **Result:** As the user rapidly hotkeys through the dataset, every new camera that slides onto the screen is already perfectly zoomed and centered on the coral head they are inspecting.

---

### The Complete Picture

If we execute these 5 phases, you will have transformed the tool.

1. **Phase 1** gives you robust, reusable mini-canvases.
2. **Phase 2** gives you a responsive, "security camera" layout that adapts to your monitors.
3. **Phase 3** gives you lightning-fast, hotkey-driven navigation through the dataset without UI clutter.
4. **Phase 4** paints high-performance markers to give you perfect spatial awareness.
5. **Phase 5** links the cameras mathematically so they act as a unified inspection array.