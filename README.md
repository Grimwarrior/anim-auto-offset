# Animation Auto Offset

This extension adds a new animation editing mode, in which any change you make to a transform or other animatable property gets applied as an offset to the existing keyframes.

This means you can pose a character or object with the normal move/rotate/scale tools and have that adjustment baked into the animation as a relative offset, without manually editing keyframes in the Graph Editor. Change a bone by a few degrees and every affected keyframe moves by the same amount, preserving the original motion.

<img src="img/showcase.gif" style='height: 100%; width: 100%; object-fit: contain'/>

## Usage

1. Disable Auto Keying (see the note below).
2. Enable **Offset Editing** using the toggle in the Graph Editor or Dope Sheet header.
3. Transform the object or bone as usual. The keyframes are updated as soon as you make the change.

By default every keyframe of the edited property is offset. The **Affect keyframes** setting in the popover menu changes that:

- **All** — every keyframe of the edited property.
- **Selected** — only the keyframes currently selected in the Graph Editor or Dope Sheet.
- **Current Frame** — only the keyframe on the current frame, no selection needed. If there is none, one is inserted (on every animated channel of the edited property) and receives your change, so it works like Auto Keying built into Offset Editing. Turn off **Insert key if missing** to only offset keys that already exist. Combine it with proportional falloff (below) to blend the pose at the playhead into the neighbouring keyframes. Cancelling the transform (`Esc`) removes the key again.

### Side of the current frame

**From the current frame** limits the affected keyframes to one side of the current frame, whatever **Affect keyframes** is set to:

- **Both** — keyframes before and after (default).
- **Forward** — only keyframes on and after the current frame. With **All** this offsets everything from here to the end of the animation; with **Current Frame** and a falloff only the following keyframes blend in.
- **Backward** — only keyframes on and before the current frame.

The keyframe on the current frame itself is always included.

Locked and muted channels are never affected. Enable **Affect only selected channels** to additionally restrict the offset to the F-Curves selected in the editor. Only continuous (float) properties are offset; booleans, integers and enums are left alone.

Besides objects and bones, animated shape keys, materials, camera and light data are supported.

### Custom properties

Animated custom properties (`obj["foo"]`, `pose.bones["Bone"]["ik_fk"]`) are offset as well. Blender does not reliably report changes of those, so they are polled about 10 times a second for the active and selected objects. Custom properties that are driven are ignored.

### Cancelling and dragging

While you drag a value (transform in the viewport, a slider, ...) the keyframes are not nudged by every small change. The add-on remembers how the keyframes looked when the drag started and always writes *that plus the total change so far*. This means there is no drift, and cancelling the drag with `Esc` (or dragging back to where you started) restores the keyframes exactly. The remembered state is dropped after a short pause, on frame change, undo/redo and when you change a setting.

## Rotations

Rotations are not offset by adding numbers to the Euler/Quaternion channels. The change you make is converted to a rotation which is then applied to every keyframe, so the motion is preserved (and quaternions stay normalized). This requires all rotation channels to be keyed on the same frames, otherwise it falls back to a per-channel offset.

In the popover, **Rotation offset** selects the space of that rotation:

- **Parent** — keyframes are rotated by the same rotation in the space of the parent (`new = delta @ key`). This is what you usually want when posing.
- **Local** — keyframes are rotated by the same rotation in their own local space (`new = key @ delta`).

## Proportional falloff (time)

Like Blender's proportional editing, but in time. When only some keyframes are affected, the offset can end abruptly at the edge. Turn on **Proportional Falloff (Time)** (the button in the header next to the Offset Editing toggle, or in the popover) to let the offset also reach the keyframes around them and fade out over a **radius**. With **Current Frame** this is the pose at the playhead pulling its neighbours along.

In the Offset Editing popover:

- **Radius** — how far the offset reaches on each side of the affected keyframes before it has faded out (the first keyframe past the radius gets no offset).
- **Keys / Frames** — the unit of the radius. *Keys* counts keyframes, *Frames* measures the distance in frames regardless of how many keyframes are in there (this is what proportional editing in the timeline does).
- **Adjust radius** (the drag button next to the radius) — click and drag left/right to change the radius interactively. Release the left mouse button to confirm, or press `Esc` / right-click to cancel and restore the previous value. (Blender's own transforms can't be changed while they run, so there is no mouse wheel adjustment.)
- **Falloff** — the shape: Smooth, Root, Constant, Linear, Sine, Quadratic, Cubic, Quartic, Quintic, Exponential or Circular. These match the falloffs of proportional editing: *Smooth*, *Root*, *Constant* and *Linear* are the same, *Sharp* is Quadratic (In) and *Sphere* is Circular (Out).
- **Mode** — the easing direction for the shapes that have one: **In** (slow start, fast end), **Out** (fast start, slow end), or **In & Out** (S-curve).

It needs a range of keyframes to fade out from, so it works with **Selected** and **Current Frame** (not with **All**). In **Selected** mode the selected keyframes must be consecutive on every animated channel, otherwise the settings are disabled and the full offset is applied to the selected keyframes. Combine it with **From the current frame** to let it fade out in one direction only.

## Indicator

While Offset Editing is on, the 3D Viewport shows a border and a label with the current settings (and that it is paused when Auto Keying is on, see below). The Graph Editor and Dope Sheet highlight the keyframes that will be affected and the falloff radius around them. With **All** they only mark the side of the current frame when it is limited to one. Turn this off with **Show indicator** in the popover.

## Auto keying

Offset Editing has no effect while Auto Keying is enabled. To avoid toggling Auto Keying by hand, enable **Auto keying override** in the add-on preferences: turning on Offset Editing will then disable Auto Keying automatically, and turning Auto Keying back on will disable Offset Editing.
