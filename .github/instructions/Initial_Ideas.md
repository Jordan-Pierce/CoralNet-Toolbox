### Ideas (still formulating, not concrete yet):

1. The "Auto-Matrix" (Security Camera Layout)Instead of a scrollable area with fixed-size thumbnails, abandon the scrollbar entirely when in this mode.The Concept: The widget automatically subdivides its available dock space into a perfect, non-scrolling grid (e.g., 1x1, 1x2, 2x2) based exactly on the user's chosen $N$.How it looks: If the user sets $N=3$, the dock splits into three large, equal-sized panes that stretch to fill the entire dock.Why it works: Scrolling is dead. You maximize every single pixel of screen real estate for exactly the cameras that matter. Because the images are much larger, the user can actually see the projected mouse markers accurately without squinting at tiny thumbnails.

If the user needs to zoom and pan, we are no longer building a "Grid of Thumbnails"—we are building a "Multi-Viewport Workspace", very similar to what you see in professional CAD, 3D modeling, or medical imaging software.To make this feel seamless and powerful without becoming tedious to manage, here are the concepts for how we can redesign this interaction:1. The Core Shift: From Thumbnails to "Mini-Canvases"Architecturally, the static image widgets must be replaced with lightweight viewports (similar to a stripped-down version of your main AnnotationWindow).The Look: They still fit into the "Auto-Matrix" layout (e.g., a 2x2 grid filling the dock), but they no longer look like static pictures. As the user moves their mouse over one, a subtle overlay appears (perhaps a small "100%" or "Fit" button in the corner), indicating it is an interactive space.The Feel: Standard interactions apply. Scroll wheel zooms; middle-click or right-click-drag pans. The projected cross-camera marker stays perfectly pinned to its correct pixel coordinate, regardless of how the user manipulates the view.2. "Target-Locked" Synchronized Zooming (The Killer Feature)If a user has 3 context images open, manually zooming and panning all three to find the exact rock or coral head they are looking at in the main window is incredibly tedious.The Concept: When the user zooms in on the primary AnnotationWindow, the $N$ context viewports automatically zoom and pan to match.How it works: The system uses the 3D ray of the mouse cursor (or the center of the main view) as an anchor. The context views automatically center themselves on that exact projected 3D point and match the relative scale of the main window.Why it works: The user never has to touch the context windows. They zoom into a coral head in the main view, and all 3 context views instantly snap to close-ups of that exact same coral head from their respective angles.3. The "Promote to Main" Interaction (The Swap)If the user is panning around a context image, they are doing so because they are looking for better visual information. Often, they will realize that the context image actually provides a better angle for the annotation they want to make.The Concept: Make it effortless to swap a context image into the main AnnotationWindow.How it works: A simple, intuitive action—like double-clicking the context viewport, or clicking a small "Promote" icon in its corner. The current main image swaps down into the context matrix, and the chosen context image takes over the giant main screen, preserving its current zoom and pan state.Why it works: It creates a fluid, frictionless workflow where the user is "driving" the camera matrix, hopping from one optimal view to the next without navigating menus.4. "Ghost" Annotations in Context ViewsIf the user is zooming and panning in these views, they are highly engaged with the pixel data.The Concept: Don't just project the mouse cursor—project the actual annotations being drawn.How it looks: If the user is drawing a polygon in the main window, the vertices and lines project into the context viewports in real-time as a semi-transparent "ghost" polygon.Why it works: If a user is annotating a complex 3D structure (like branching coral), they can look at the zoomed-in context view while drawing in the main view to ensure they aren't accidentally including background pixels that just happen to look attached from the main camera's perspective.5. Performance & Detail on DemandSince you are dealing with high-res rasters, loading $N$ full-resolution interactive canvases simultaneously could be heavy.The Concept: The Mini-Canvases use the low-res thumbnail proxy by default.How it works: As long as the context view is zoomed out (Fit to View), it uses the cheap, cached thumbnail. The absolute microsecond the user scrolls the wheel to zoom in (or the Auto-Sync triggers a zoom), it instantly seamlessly swaps to the full-resolution raster data for that specific viewport.Summary of the Ideal FlowImagine this: The user sets $N=2$. The dock splits into two large panels. The user zooms into a patch of sand in the main window; the two side panels instantly zoom into that exact same patch of sand from different angles. The user realizes the second panel has better lighting. They double-click it. It instantly becomes the main window. They draw a polygon, and as they click, they watch the polygon shape conform perfectly in the other views.

If a user has an ultra-wide monitor, a $1 \times 3$ (horizontal row) might be perfect. If they have the dock pinned to the right side of their screen, a $3 \times 1$ (vertical column) is mandatory.Here is how you can handle flexible matrix layouts cleanly and intuitively:

1. Aspect-Ratio Aware "Auto-Flow" (The Smart Default)The absolute best user experience feels like magic: the system just knows what to do based on the shape of the window.The Concept: The matrix automatically reflows itself based on the bounding box of its container (the Qt Dock).How it works: * If the user pins the dock to the Right or Left of the main window, the dock naturally becomes tall and skinny. The grid detects this aspect ratio and snaps to a $3 \times 1$ vertical stack.If the user moves the dock to the Bottom, it becomes short and wide. The grid instantly reflows into a $1 \times 3$ horizontal strip.If they float the dock and size it into a large square, it calculates the most efficient use of space and snaps to a $2 \times 2$ grid.Why it works: The user never has to open a menu to fix their layout. They just arrange their workspace, and the software actively conforms to fit it perfectly.2. The "Grid Chooser" Override (For Power Users)While auto-flow is great, professional tools should always offer a manual override for edge cases.The Concept: A lightweight, visual layout selector in the toolbar of the multi-view dock.How it works: You place a small "Layout" icon in the dock's header. Clicking it drops down a visual menu—very similar to how viewport layouts work in 3D animation software (like Blender or Maya) or video conferencing apps. It shows little wireframe icons for:$[ \boxbox ]$ ($1 \times 2$)$[ \boxminus ]$ ($2 \times 1$)$[ \boxplus ]$ ($2 \times 2$)$[ \shortmid\shortmid\shortmid ]$ ($1 \times 3$)$[ \equiv ]$ ($3 \times 1$)Why it works: It gives ultimate control. If a user wants a $3 \times 1$ horizontal strip hovering on a vertical monitor for some specific reason, they can force it.3. Handling the "Letterbox" Problem (Target-Lock Synergy)There is a catch when forcing a specific grid: the aspect ratio of the cells might clash with the aspect ratio of the images. For example, forcing landscape photos into a $3 \times 1$ tall, skinny column will result in massive black bars (letterboxing) if you just fit the image to the frame.The Concept: Use the "Target Lock" feature to solve the aspect ratio clash!How it works: When the user is zoomed out, you accept the letterboxing so they can see the whole image. But the moment they trigger Target Lock (zooming in on the main window), the context views don't just zoom—they Fill their respective grid cells.Why it works: Because the views are centered on the specific 3D coordinate the user cares about, it doesn't matter if the edges of the image are cropped off by a skinny vertical cell. The user's target is perfectly centered, utilizing 100% of the available pixel space in that cell.4. The "Tear-Away" Canvas (Multi-Monitor Nirvana)If a user is asking for specific matrix sizes, they might be trying to span across monitors.The Concept: Allow the individual mini-canvases to be dragged completely out of the matrix.How it works: The user clicks the header of one of the 3 context views and drags it. It detaches from the grid and becomes its own floating, borderless window. They can drag it to a second monitor and maximize it. The "Target Lock" sync remains unbroken.Why it works: It accommodates the most extreme power-user setups without cluttering the UI for single-monitor laptop users.

To keep the UI completely pristine while allowing the user to dig deeper into the $N$-nearest list, we should elevate the navigation from the individual canvas level to the matrix level.Here are three cleaner, modern ways to handle this without cluttering the mini-canvases:1. The Global "Conveyor Belt" (Zero-UI Scrolling)Treat the matrix of mini-canvases as a viewing window over a much longer ranked strip of cameras.How it works: The user moves their mouse over the gap between the canvases (or holds a modifier key like Shift while scrolling). Scrolling down shifts the entire matrix by one rank.The Flow: If the matrix shows cameras Ranked [1, 2, 3], one tick of the scroll wheel shifts it to [2, 3, 4], then [3, 4, 5].Why it's clean: There are literally zero buttons on the screen. It relies on a natural, intuitive gesture that keeps the strict spatial ranking perfectly intact.

Using a dedicated AnnotationManager class is the absolute best approach here. If we dump all this logic into MainWindow, it will become massively bloated. By creating an AnnotationManager, we establish a strict Model-View-Controller (MVC) architecture.

The AnnotationManager becomes the "Model" (the single source of truth), and your AnnotationWindow and BaseCanvas instances become "Views" that simply listen to the manager and draw what it tells them to.

Here is the exact blueprint for decoupling the storage into an AnnotationManager.

1. Define the AnnotationManager (The Single Source of Truth)
This class will inherit from QObject so it can broadcast Qt Signals whenever data changes.

Data Structures to Move Here (from AnnotationWindow):

self.annotations_dict (Dictionary of UUID -> Annotation)

self.image_annotations_dict (Dictionary of image_path -> List of Annotations)

self.selected_annotations (List of currently selected annotations)

Core Signals it Needs to Emit:

annotationAdded(annotation)

annotationsAdded(list_of_annotations)

annotationRemoved(annotation_id, image_path)

annotationModified(annotation_id) (Fired when geometry, label, or scale changes)

selectionChanged(list_of_selected_ids)

Core Methods it Needs:

add_annotation(annotation, record_action=True)

delete_annotation(annotation_id)

get_image_annotations(image_path)

select_annotation(annotation_id, multi_select=False)

clear_selection()

2. Rewiring the Flow of Data
Right now, the drawing tools inside AnnotationWindow create an annotation and immediately draw it to the scene. We need to insert the Manager into this loop.

The New Flow for Creating an Annotation:

The user draws a polygon in the AnnotationWindow.

The active Tool finishes the shape and creates a PolygonAnnotation object in memory.

Instead of drawing it, the tool calls MainWindow.annotation_manager.add_annotation(new_poly).

The AnnotationManager saves it to its dictionaries and broadcasts the annotationAdded signal.

The Views React:

The AnnotationWindow hears the signal. It checks if the annotation belongs to its current image. If yes, it generates an editable QGraphicsItem and adds it to its scene.

The Context BaseCanvas instances hear the signal. If the annotation belongs to their currently assigned image, they generate a read-only QGraphicsItem and add it to their local scenes.

3. Modifying BaseCanvas to be "Data-Driven"
Since BaseCanvas instances (in your security camera layout) will be dynamically loading images as they scroll along the conveyor belt, they need to be able to fetch their data instantly.

When a BaseCanvas is assigned a new image_path, it calls MainWindow.annotation_manager.get_image_annotations(image_path).

It loops through that list and creates lightweight, read-only graphics items.

Because it is connected to the AnnotationManager's signals, if the user modifies an annotation in the main window, the BaseCanvas instantly receives the annotationModified signal and redraws that specific polygon to match the new coordinates.

4. Handling the Undo/Redo Stack (ActionStack)
Currently, AnnotationWindow owns the ActionStack.

Move it to the Manager: The AnnotationManager should own the ActionStack.

When a user hits Ctrl+Z (Undo), the AnnotationManager pops the action, reverts the data state, and emits annotationRemoved or annotationModified. The canvases simply listen to the signals and update visually, completely unaware that an "Undo" command was what triggered the change.