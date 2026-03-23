Here is the detailed execution plan for **Phase 3: The Conveyor Belt Navigation Engine**.

The goal of this phase is to turn the static grid of canvases into a "sliding window" that scrubs through the dataset perfectly ordered by physical proximity to the main camera, using only hotkeys.

### Step 1: The Data Pipeline (Filtering & Storing the Sorted List)

Before we can slide a window across a list of cameras, the `ContextMatrixWidget` needs to own that list.

Your `MVATManager` already calculates proximity scores and generates an ordered list of cameras (in `_reorder_cameras`). We will feed this directly into the new widget.

**The Logic:**

* Add a property `self.ordered_context_paths = []` to `ContextMatrixWidget`.
* Create a method `set_camera_order(ordered_paths, active_path)`.
* **Crucial UX Detail:** Inside this method, *filter out the `active_path*`. The user is already looking at the active camera in the giant main `AnnotationWindow`. The context matrix should only show the neighbors (Rank 1, Rank 2, Rank 3...), skipping Rank 0 (the active camera itself).
* Store the filtered list in `self.ordered_context_paths`.
* Reset the conveyor belt by setting `self.current_rank_offset = 0`.
* Trigger an immediate visual update to paint the first batch of images.

### Step 2: State Management & The Shift Engine

The widget needs to know its own capacity and where it currently sits in the list.

**The Logic:**

* **Capacity:** The widget's capacity $N$ is simply the number of currently visible canvases in your layout (e.g., if you are in a $2 \times 2$ layout, capacity is 4).
* **The Offset Tracker:** Maintain an integer `self.current_rank_offset`.
* **Create the `shift_conveyor(delta)` Method:**
1. Calculate `new_offset = self.current_rank_offset + delta`.
2. **Bounds Checking:** Clamp the `new_offset` so it cannot go below `0`, and cannot go higher than the length of the camera list minus your capacity. (e.g., If you have 10 cameras and show 3 at a time, the maximum offset is 7).
3. If the offset hasn't changed (e.g., the user hit the end of the list), do nothing and return.
4. Update `self.current_rank_offset = new_offset`.
5. Slice the master list to get the new visible targets: `target_paths = self.ordered_context_paths[self.current_rank_offset : self.current_rank_offset + capacity]`.
6. Pass `target_paths` to the `update_context_cameras` method you built in Phase 2.



### Step 3: Hotkey Integration (Micro and Macro Steps)

Now we wire the hotkeys directly into the `shift_conveyor` method.

Because the user's mouse and focus will likely be in the main `AnnotationWindow` when they want to cycle the context cameras, you should capture these hotkeys globally.

**The Logic:**

* In your `QtMainWindow.py`, locate the `GlobalEventFilter` or the `keyPressEvent` handler.
* Intercept the specific key combinations and route them to the `ContextMatrixWidget`:
* **Micro-Step Forward (`Ctrl + Right Arrow`):** Call `context_matrix.shift_conveyor(1)`. The matrix visually slides one image to the left. The oldest context image drops off, and the next nearest neighbor slides in.
* **Micro-Step Backward (`Ctrl + Left Arrow`):** Call `context_matrix.shift_conveyor(-1)`.
* **Macro-Step / Page Forward (`Ctrl + Down Arrow`):** Call `context_matrix.shift_conveyor(N)` (where $N$ is the widget's current capacity). This instantly swaps the entire grid to the next "tier" of neighbors.
* **Macro-Step / Page Backward (`Ctrl + Up Arrow`):** Call `context_matrix.shift_conveyor(-N)`.



### Step 4: The Visual Indicator (The HUD Label)

Since the user is navigating without a scrollbar, they need a subtle visual cue to know where they are in the dataset.

**The Logic:**

* Add a sleek, semi-transparent `QLabel` overlay to the top-right corner of the `ContextMatrixWidget` (or integrate it into its toolbar).
* Whenever `shift_conveyor` completes, update the text of this label.
* **Format:** `"Showing nearest neighbors: {start} - {end} of {total}"` (e.g., *"Showing nearest neighbors: 1 - 3 of 45"*).
* If the user presses `Ctrl + Right`, it instantly updates to *"Showing nearest neighbors: 2 - 4 of 45"*.

### Step 5: Handling Layout Resizes Seamlessly

If the user is scrolled deep into the conveyor belt (e.g., showing cameras 10, 11, and 12 in a $1 \times 3$ strip) and they dock the widget differently (forcing it to snap to a $2 \times 2$ grid), the system must handle the capacity change gracefully.

**The Logic:**

* In your `_rebuild_layout` method (from Phase 2), after the new layout is set and the capacity $N$ is updated, simply call `shift_conveyor(0)`.
* Because `shift_conveyor` includes the clamping logic (Step 2), it will automatically adjust the offset if the new capacity causes an out-of-bounds error, and immediately repaint the grid with the correct number of canvases starting from their current rank.

---

### Checkpoint / Validation

By the end of Phase 3, you have achieved the "High-Speed Targeting" workflow.

1. You select a camera in the main window.
2. The context dock instantly populates with the 3 nearest physical neighbors.
3. You press `Ctrl + Right` and watch the 1st neighbor vanish, the 2nd and 3rd shift over, and the 4th nearest neighbor appear.
4. The transition is instant and entirely driven by the keyboard, keeping the user's mouse free to annotate.

Shall we move on to **Phase 4: Implementing the Dual-Marker System** so we can actually see where the mouse is pointing in these context views?