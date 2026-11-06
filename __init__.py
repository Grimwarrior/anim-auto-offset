import bpy # type: ignore
from bpy.props import BoolProperty, IntProperty, EnumProperty # type: ignore
from bpy.app.handlers import persistent # type: ignore
from mathutils import Vector, Quaternion, Euler # type: ignore
import numpy as np # type: ignore
import math
import re
import time
import traceback
from collections import namedtuple
import blf # type: ignore
import gpu
from gpu_extras.batch import batch_for_shader

from . import __package__

class AnimAutoOffsetPreferences(bpy.types.AddonPreferences):
    bl_idname = __package__

    auto_key_override : BoolProperty (
            name="Auto keying override",
            description="When enabled 'Offset editing' mode will switch off 'Auto keying' and vice versa",
            default=False
        ) # # type: ignore

    def draw(self, context):
        self.layout.prop(self, 'auto_key_override')

def get_is_auto_key_override(context):
    addon = context.preferences.addons.get(__package__)
    return addon is not None and addon.preferences.auto_key_override

def _get_obj_fcurves(obj):
    """F-Curves of the ID's active action/slot, None if there are none.
    Works for any ID with animation_data (objects, shape keys, materials, cameras, lights...)"""
    anim_data = obj.animation_data
    if anim_data is None:
        return None
    action = anim_data.action
    action_slot = anim_data.action_slot
    if not action or not action_slot:
        return None
    # TODO: update to support multiple layers when they are actually introduced
    for layer in action.layers:
        for strip in layer.strips:
            if strip.type != 'KEYFRAME':
                continue
            channelbag = strip.channelbag(action_slot)
            if channelbag is not None:
                return channelbag.fcurves
    return None

# user-facing settings snapshot, read once per update
OffsetSettings = namedtuple('OffsetSettings', [
    'target', 'side', 'frame', 'insert_missing', 'only_selected_channels',
    'easing_window', 'easing_unit', 'easing_function', 'easing_mode',
    'rotation_space',
])

# which side of the current frame is affected
SIDE_BOTH = 'BOTH'
SIDE_AFTER = 'AFTER'
SIDE_BEFORE = 'BEFORE'

SIDE_LABELS = {
    SIDE_BOTH: 'both sides',
    SIDE_AFTER: 'forward',
    SIDE_BEFORE: 'backward',
}

# which keyframes the offset is applied to
TARGET_ALL = 'ALL'
TARGET_SELECTED = 'SELECTED'
TARGET_CURRENT = 'CURRENT'

TARGET_LABELS = {
    TARGET_ALL: 'All keys',
    TARGET_SELECTED: 'Selected keys',
    TARGET_CURRENT: 'Current frame',
}

# unit of the easing window
UNIT_KEYS = 'KEYS'
UNIT_FRAMES = 'FRAMES'

# keyframes closer than this (in frames) to the current frame count as "on" it
FRAME_TOLERANCE = 1e-3

def _get_settings(scene):
    return OffsetSettings(
        target=scene.anim_offset_target,
        side=scene.anim_offset_side,
        frame=float(scene.frame_current),
        insert_missing=scene.anim_offset_insert_missing,
        only_selected_channels=scene.anim_offset_mode_only_selected_channels,
        # proportional editing off = no falloff window
        easing_window=scene.anim_offset_easing_window if scene.anim_offset_proportional else 0,
        easing_unit=scene.anim_offset_easing_unit,
        easing_function=scene.anim_offset_easing_function,
        easing_mode=scene.anim_offset_easing_mode,
        rotation_space=scene.anim_offset_rotation_space,
    )

def _same_for_all_modes(func):
    return {'in': func, 'out': func, 'in_out': func}

EASINGS = {
    # same shapes as the falloffs of Blender's proportional editing
    'constant': _same_for_all_modes(lambda t: 1.0),
    'smooth': _same_for_all_modes(lambda t: t * t * (3 - 2 * t)),
    'root': _same_for_all_modes(lambda t: math.sqrt(t)),
    'linear': {
        'in':     lambda t: t,
        'out':    lambda t: t,
        'in_out': lambda t: t,
    },
    'sine': {
        'in':     lambda t: 1 - math.cos(t * math.pi / 2),
        'out':    lambda t: math.sin(t * math.pi / 2),
        'in_out': lambda t: -(math.cos(t * math.pi) - 1) / 2,
    },
    'quadratic': {
        'in':     lambda t: t * t,
        'out':    lambda t: 1 - (1 - t) * (1 - t),
        'in_out': lambda t: 2 * t * t if t < 0.5 else 1 - (-2 * t + 2) ** 2 / 2,
    },
    'cubic': {
        'in':     lambda t: t * t * t,
        'out':    lambda t: 1 - (1 - t) ** 3,
        'in_out': lambda t: 4 * t * t * t if t < 0.5 else 1 - (-2 * t + 2) ** 3 / 2,
    },
    'quartic': {
        'in':     lambda t: t * t * t * t,
        'out':    lambda t: 1 - (1 - t) ** 4,
        'in_out': lambda t: 8 * t * t * t * t if t < 0.5 else 1 - (-2 * t + 2) ** 4 / 2,
    },
    'quintic': {
        'in':     lambda t: t * t * t * t * t,
        'out':    lambda t: 1 - (1 - t) ** 5,
        'in_out': lambda t: 16 * t * t * t * t * t if t < 0.5 else 1 - (-2 * t + 2) ** 5 / 2,
    },
    'exponential': {
        'in':     lambda t: 0.0 if t == 0 else math.pow(2, 10 * (t - 1)),
        'out':    lambda t: 1.0 if t == 1 else 1 - math.pow(2, -10 * t),
        'in_out': lambda t: 0.0 if t == 0 else 1.0 if t == 1 else math.pow(2, 10 * (2 * t - 1)) / 2 if t < 0.5 else 1 - math.pow(2, -10 * (2 * t - 1)) / 2,
    },
    'circular': {
        'in':     lambda t: 1 - math.sqrt(1 - t * t),
        'out':    lambda t: math.sqrt(1 - (1 - t) * (1 - t)),
        'in_out': lambda t: (1 - math.sqrt(1 - (2 * t) ** 2)) / 2 if t < 0.5 else (math.sqrt(1 - (-2 * t + 2) ** 2) + 1) / 2,
    },
}

def _get_keyframe_selection(fcurve):
    """(control points, left handles, right handles) selection as bool arrays"""
    kps = fcurve.keyframe_points
    size = len(kps)
    result = []
    for attr in ('select_control_point', 'select_left_handle', 'select_right_handle'):
        selected = np.empty(size, dtype=bool)
        kps.foreach_get(attr, selected)
        result.append(selected)
    return result

def _get_keyframe_frames(fcurve):
    kps = fcurve.keyframe_points
    co = np.empty(len(kps) * 2)
    kps.foreach_get('co', co)
    return co[0::2]

def _is_consecutive(indices):
    return len(indices) == 1 or bool(np.all(np.diff(indices) == 1))

def _get_easing_weights(fcurve, selected, easing_window, easing_function, easing_mode,
                        easing_unit=UNIT_KEYS):
    """Per keyframe weights: 1 inside the selected range, fading to 0 over
    `easing_window` keyframes (or frames) on each side. None if easing does not apply."""
    if easing_window <= 0:
        return None

    selected_indices = np.where(selected)[0]
    if len(selected_indices) == 0 or not _is_consecutive(selected_indices):
        return None

    size = len(fcurve.keyframe_points)
    first_sel = int(selected_indices[0])
    last_sel = int(selected_indices[-1])
    frames = _get_keyframe_frames(fcurve)

    weights = np.zeros(size)
    weights[first_sel:last_sel + 1] = 1.0

    easing_fun = EASINGS[easing_function][easing_mode]

    def _ease(t):
        if t <= 0:
            return 0.0
        if t >= 1:
            return 1.0
        return easing_fun(t)

    if easing_unit == UNIT_FRAMES:
        # the window is a distance in frames from the edge of the selected range
        near = frames[first_sel]
        for i in range(first_sel - 1, -1, -1):
            dist = near - frames[i]
            if dist >= easing_window:
                break
            if dist > 0:
                weights[i] = _ease(1.0 - dist / easing_window)

        near = frames[last_sel]
        for i in range(last_sel + 1, size):
            dist = frames[i] - near
            if dist >= easing_window:
                break
            if dist > 0:
                weights[i] = _ease(1.0 - dist / easing_window)
        return weights

    # The weight fades between the selected boundary (1) and the first keyframe
    # *outside* of the window (0), so that every keyframe inside the window gets
    # a non-zero weight. t is computed from actual frame positions, so uneven
    # keyframe spacing is respected.

    # BEFORE selected range
    count = min(int(easing_window), first_sel)
    if count > 0:
        near = frames[first_sel]
        win_start = first_sel - count
        if win_start - 1 >= 0:
            outer = frames[win_start - 1]
        else: # no keyframe past the window, extrapolate using average spacing
            outer = frames[win_start] - (near - frames[win_start]) / count
        span = near - outer
        if span > 0:
            for i in range(win_start, first_sel):
                weights[i] = _ease((frames[i] - outer) / span)

    # AFTER selected range
    count = min(int(easing_window), size - 1 - last_sel)
    if count > 0:
        near = frames[last_sel]
        win_end = last_sel + count # inclusive
        if win_end + 1 < size:
            outer = frames[win_end + 1]
        else:
            outer = frames[win_end] + (frames[win_end] - near) / count
        span = outer - near
        if span > 0:
            for i in range(last_sel + 1, win_end + 1):
                weights[i] = _ease((outer - frames[i]) / span)

    return weights

def _get_target_selection(fcurve, settings):
    """(control points, left handles, right handles) bool arrays of the keyframes the
    offset is aimed at (before easing)"""
    if settings.target == TARGET_CURRENT:
        sel_cp = np.abs(_get_keyframe_frames(fcurve) - settings.frame) < FRAME_TOLERANCE
        return sel_cp, sel_cp, sel_cp
    return _get_keyframe_selection(fcurve)

def _get_side_mask(fcurve, settings):
    """1 for keyframes on the affected side of the current frame (the one on the frame itself
    always counts), 0 for the others. None if both sides are affected"""
    if settings.side == SIDE_BOTH:
        return None
    frames = _get_keyframe_frames(fcurve)
    if settings.side == SIDE_AFTER:
        return (frames >= settings.frame - FRAME_TOLERANCE).astype(float)
    return (frames <= settings.frame + FRAME_TOLERANCE).astype(float)

def _get_keyframe_weights(fcurve, settings):
    """(control point, left handle, right handle) weights in 0..1 for each keyframe"""
    cp, left, right = _get_unsided_keyframe_weights(fcurve, settings)
    mask = _get_side_mask(fcurve, settings)
    if mask is not None: # handles follow their keyframe
        cp, left, right = cp * mask, left * mask, right * mask
    return cp, left, right

def _get_unsided_keyframe_weights(fcurve, settings):
    size = len(fcurve.keyframe_points)
    if settings.target == TARGET_ALL:
        ones = np.ones(size)
        return ones, ones, ones

    sel_cp, sel_left, sel_right = _get_target_selection(fcurve, settings)
    eased = _get_easing_weights(fcurve, sel_cp, settings.easing_window,
                                settings.easing_function, settings.easing_mode,
                                settings.easing_unit)
    if eased is not None:
        return eased, eased, eased

    # handles follow their keyframe, or can be moved on their own when selected
    cp = sel_cp.astype(float)
    return (cp,
            np.maximum(cp, sel_left.astype(float)),
            np.maximum(cp, sel_right.astype(float)))

# ---------------------------------------------------------------------------
# rotations
# ---------------------------------------------------------------------------
# Adding a delta to Euler/Quaternion components is not a rotation offset
# (quaternions would denormalize, Eulers only work for tiny changes). Instead
# the delta is converted to a rotation which is then applied to every keyframe.

_EULER_ORDERS = {'XYZ', 'XZY', 'YXZ', 'YZX', 'ZXY', 'ZYX'}

def _get_rotation_kind(data_path):
    prop = data_path.rsplit('.', 1)[-1]
    if prop.startswith('delta_'):
        prop = prop[len('delta_'):]
    if prop == 'rotation_euler':
        return 'EULER'
    if prop == 'rotation_quaternion':
        return 'QUATERNION'
    return None

def _get_euler_order(id_data, data_path):
    try:
        owner = id_data.path_resolve(data_path.rsplit('.', 1)[0]) if '.' in data_path else id_data
        mode = owner.rotation_mode
    except (KeyError, AttributeError, IndexError, ValueError):
        return 'XYZ'
    return mode if mode in _EULER_ORDERS else 'XYZ'

def _to_quaternion(kind, values, order):
    if kind == 'EULER':
        return Euler(tuple(values), order).to_quaternion()
    q = Quaternion(tuple(values))
    if q.magnitude == 0:
        return None
    q.normalize()
    return q

def _wrap_angle_delta(delta):
    return (delta + math.pi) % (2 * math.pi) - math.pi

# ---------------------------------------------------------------------------
# F-Curve data helpers
# ---------------------------------------------------------------------------

def _read_arrays(fcurve):
    """(co, handle_left, handle_right) as flat float arrays [x0, y0, x1, y1, ...]"""
    kps = fcurve.keyframe_points
    size = len(kps)
    out = []
    for attr in ('co', 'handle_left', 'handle_right'):
        arr = np.empty(size * 2)
        kps.foreach_get(attr, arr)
        out.append(arr)
    return out

def _write_arrays(fcurve, co, handle_left, handle_right):
    kps = fcurve.keyframe_points
    kps.foreach_set('co', co)
    kps.foreach_set('handle_left', handle_left)
    kps.foreach_set('handle_right', handle_right)
    fcurve.update() # re-calculates auto handles

def _has_key_on_frame(fcurve, frame):
    frames = _get_keyframe_frames(fcurve)
    return len(frames) > 0 and bool(np.any(np.abs(frames - frame) < FRAME_TOLERANCE))

def _insert_key(fcurve, frame):
    fcurve.keyframe_points.insert(frame, fcurve.evaluate(frame))
    fcurve.update()

def _remove_key(fcurve, frame):
    frames = _get_keyframe_frames(fcurve)
    indices = np.where(np.abs(frames - frame) < FRAME_TOLERANCE)[0]
    if len(indices):
        fcurve.keyframe_points.remove(fcurve.keyframe_points[int(indices[0])])
        fcurve.update()

# ---------------------------------------------------------------------------
# offset session
# ---------------------------------------------------------------------------
# A transform (drag in the viewport, slider...) produces a lot of small updates.
# Applying every one of them as a delta to the keyframes accumulates rounding errors
# and can not be undone, e.g. by cancelling the transform. Instead, the first update
# of a property takes a snapshot of its keyframes ("session") and every following
# update writes `snapshot + total change since the first update`. This way:
#  - there is no drift, the result only depends on the current value
#  - cancelling a transform (value goes back to the start) restores the keyframes
#    exactly, and removes keyframes which were inserted by the session
# The session is dropped when it goes idle, on frame change, undo/redo, or when
# settings change.

SESSION_IDLE_SECONDS = 0.75
_ZERO_EPS = 1e-7

class _Channel:
    """Snapshot of a single F-Curve"""
    __slots__ = ('co', 'left', 'right', 'w_cp', 'w_left', 'w_right')

class OffsetSession:
    def __init__(self, id_data, data_path, fcurves, start, settings):
        self.settings = settings
        self.start = start
        self.is_vector = isinstance(start, Vector)
        self.kind = _get_rotation_kind(data_path) if self.is_vector else None
        self.order = _get_euler_order(id_data, data_path) if self.kind == 'EULER' else None
        self.indices = sorted(fcurves.keys())
        self.orig = dict()     # index -> arrays *before* a keyframe got inserted
        self.inserted = dict() # index -> frame of the inserted keyframe
        self.removed = False   # inserted keyframes are removed because the change was reverted
        self.channels = dict()
        self.rot = None
        # for the per channel Euler fallback, angle changes are accumulated so that 359 -> 1 is +2 and not -358
        self.accumulated = start * 0.0 if self.is_vector else 0.0
        self.last = start

        if settings.target == TARGET_CURRENT and settings.insert_missing:
            for i, fcurve in fcurves.items():
                if len(fcurve.keyframe_points) == 0 or _has_key_on_frame(fcurve, settings.frame):
                    continue
                self.orig[i] = _read_arrays(fcurve)
                _insert_key(fcurve, settings.frame)
                self.inserted[i] = settings.frame

        for i, fcurve in fcurves.items():
            ch = _Channel()
            ch.co, ch.left, ch.right = _read_arrays(fcurve)
            ch.w_cp, ch.w_left, ch.w_right = _get_keyframe_weights(fcurve, settings)
            self.channels[i] = ch

        self._setup_rotation()

    def _setup_rotation(self):
        if self.kind is None or not self.is_vector:
            return
        n = 3 if self.kind == 'EULER' else 4
        if len(self.start) != n or any(i not in self.channels for i in range(n)):
            return
        frames = [self.channels[i].co[0::2] for i in range(n)]
        # rotations can only be offset as a whole if all channels are keyed on the same frames
        if any(len(f) != len(frames[0]) or not np.allclose(f, frames[0]) for f in frames):
            return
        chans = [self.channels[i] for i in range(n)]
        values = np.stack([c.co[1::2] for c in chans], axis=1)
        cp_w = np.max([c.w_cp for c in chans], axis=0)
        self.rot = (values,
                    cp_w,
                    np.max([c.w_left for c in chans], axis=0) > 0,
                    np.max([c.w_right for c in chans], axis=0) > 0,
                    np.nonzero(cp_w > 0)[0])

    def is_valid_for(self, fcurves, settings):
        if settings != self.settings or sorted(fcurves.keys()) != self.indices:
            return False
        if self.removed:
            return True
        for i, fcurve in fcurves.items():
            if len(fcurve.keyframe_points) * 2 != len(self.channels[i].co):
                return False
        return True

    # -- cancel handling -------------------------------------------------
    def _remove_inserted(self, fcurves):
        for i, frame in self.inserted.items():
            fcurve = fcurves[i]
            _remove_key(fcurve, frame)
            _write_arrays(fcurve, *self.orig[i])
        self.removed = True

    def _reinsert(self, fcurves):
        for i, frame in self.inserted.items():
            _insert_key(fcurves[i], frame)
        self.removed = False

    # -- applying ----------------------------------------------------------
    def apply(self, fcurves, new):
        """Write keyframes for the new property value. False if the F-Curves no longer match the session"""
        if self.is_vector:
            reverted = float(np.linalg.norm(np.array(new) - np.array(self.start))) < _ZERO_EPS
        else:
            reverted = abs(new - self.start) < _ZERO_EPS

        if self.inserted:
            if reverted:
                if not self.removed:
                    self._remove_inserted(fcurves)
                return True
            if self.removed:
                self._reinsert(fcurves)

        if not self.is_valid_for(fcurves, self.settings):
            return False

        if reverted: # exactly the keyframes the session started with
            self.last = new
            self.accumulated = self.start * 0.0 if self.is_vector else 0.0
            for i, fcurve in fcurves.items():
                ch = self.channels[i]
                _write_arrays(fcurve, ch.co, ch.left, ch.right)
            return True

        if self.rot is not None:
            self._apply_rotation(fcurves, new)
        else:
            self._apply_additive(fcurves, new)
        return True

    def _apply_additive(self, fcurves, new):
        if self.kind == 'EULER' and self.is_vector:
            # angle changes are accumulated so that 359 -> 1 degrees is +2 and not -358
            step = Vector(_wrap_angle_delta(d) for d in (new - self.last))
            self.accumulated = self.accumulated + step
            total = self.accumulated
        else:
            total = new - self.start # exact, so that going back to the start value restores the keyframes
        self.last = new

        for i, fcurve in fcurves.items():
            ch = self.channels[i]
            delta = float(total[i] if self.is_vector else total)
            co, left, right = ch.co.copy(), ch.left.copy(), ch.right.copy()
            co[1::2] += delta * ch.w_cp
            left[1::2] += delta * ch.w_left
            right[1::2] += delta * ch.w_right
            _write_arrays(fcurve, co, left, right)

    def _apply_rotation(self, fcurves, new):
        values, cp_w, left_move, right_move, weighted = self.rot
        start_q = _to_quaternion(self.kind, self.start, self.order)
        new_q = _to_quaternion(self.kind, new, self.order)
        if start_q is None or new_q is None:
            return

        parent_space = self.settings.rotation_space == 'PARENT'
        # parent space: new = delta @ old, local space: new = old @ delta
        delta_q = new_q @ start_q.inverted() if parent_space else start_q.inverted() @ new_q

        identity = Quaternion()
        new_values = values.copy()
        for k in weighted:
            weight = float(cp_w[k])
            if self.kind == 'EULER':
                key_euler = Euler(values[k].tolist(), self.order)
                key_q = key_euler.to_quaternion()
            else:
                key_q = _to_quaternion(self.kind, values[k].tolist(), self.order)
                if key_q is None:
                    continue

            d = delta_q if weight >= 1.0 else identity.slerp(delta_q, weight)
            result_q = d @ key_q if parent_space else key_q @ d

            if self.kind == 'EULER':
                new_values[k] = tuple(result_q.to_euler(self.order, key_euler)) # compatible with the old one, no flips
            else:
                if result_q.dot(key_q) < 0: # keep the sign continuous
                    result_q.negate()
                new_values[k] = tuple(result_q)

        key_deltas = new_values - values
        for i in range(values.shape[1]):
            ch = self.channels[i]
            co, left, right = ch.co.copy(), ch.left.copy(), ch.right.copy()
            co[1::2] += key_deltas[:, i]
            left[1::2] += key_deltas[:, i] * left_move
            right[1::2] += key_deltas[:, i] * right_move
            _write_arrays(fcurves[i], co, left, right)

_sessions: dict = {} # (ID pointer, data path) -> OffsetSession
_last_session_activity = 0.0

def end_sessions():
    _sessions.clear()

def _session_idle_timer():
    if not _sessions:
        return None
    remaining = SESSION_IDLE_SECONDS - (time.monotonic() - _last_session_activity)
    if remaining <= 0:
        end_sessions()
        return None
    return remaining

def _touch_sessions():
    global _last_session_activity
    _last_session_activity = time.monotonic()
    if not bpy.app.timers.is_registered(_session_idle_timer):
        bpy.app.timers.register(_session_idle_timer, first_interval=SESSION_IDLE_SECONDS)

def apply_change(id_data, data_path, fcurves, old, new, settings):
    """`old`: value of the property before it was changed, `new`: its current value"""
    key = (id_data.as_pointer(), data_path)
    session = _sessions.get(key)
    if session is not None and not session.is_valid_for(fcurves, settings):
        session = None

    if session is None:
        session = OffsetSession(id_data, data_path, fcurves, old, settings)
        _sessions[key] = session

    if not session.apply(fcurves, new):
        # keyframes changed under our feet (e.g. edited manually), start over from the current state
        session = OffsetSession(id_data, data_path, fcurves, old, settings)
        _sessions[key] = session
        session.apply(fcurves, new)

    _touch_sessions()

# ---------------------------------------------------------------------------
# change tracking
# ---------------------------------------------------------------------------

def read_numeric_property(id_data, data_path):
    """Value of a continuous (float, or float array) property as float or Vector.
    None for anything else (bool, int, enum, string, unresolvable path), those must never be offset"""
    try:
        data = id_data.path_resolve(data_path)
    except (KeyError, AttributeError, IndexError, ValueError, TypeError):
        return None # stale F-Curve: its property no longer exists on the ID
    if isinstance(data, (bool, int, str)):
        return None
    if isinstance(data, float):
        return data
    try:
        items = list(data)
    except TypeError:
        return None
    if not items or not all(isinstance(x, float) for x in items):
        return None
    return Vector(items)

# custom (ID) properties, e.g. obj["foo"] or pose.bones["Bone"]["foo"]. Setting them does
# not reliably trigger a depsgraph update so they are polled instead.
_CUSTOM_PROPERTY_PATH = re.compile(r'\[(?:"(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\')\]$')

def is_custom_property_path(data_path):
    return _CUSTOM_PROPERTY_PATH.search(data_path) is not None

def has_anim_attr_changed(obj, pre_update, attr, sub_attr):
    attr_val = getattr(obj.animation_data, attr)
    if attr_val:
        if pre_update[attr] != getattr(attr_val, sub_attr):
            return True
    elif pre_update[attr]:
        return True
    return False

def save_anim_attr(obj, pre_update, attr, sub_attr):
    attr_val = getattr(obj.animation_data, attr)
    pre_update[attr] = getattr(attr_val, sub_attr) if attr_val else ''

_pre_update_registry: dict[int, dict] = {}

def _is_editable_fcurve(fcurve, settings):
    if fcurve.lock or fcurve.mute:
        return False
    if settings.only_selected_channels and not fcurve.select:
        return False
    return True

_ANIMATION_MATCH_EPS = 1e-4

def _is_driven_by_animation(fcurves, old, new, frame):
    """True if the property changed because its own F-Curves say so: every changed channel
    already evaluates to its new value on `frame`. That is what editing keyframes in the
    Graph Editor / Dope Sheet looks like (the animation system writes the evaluated value
    back to the property). Offsetting then would rewrite the keyframes under the running
    transform, double the edit and corrupt the F-Curve data the transform points into."""
    if isinstance(new, Vector):
        changed = [(i, new[i]) for i in range(len(new)) if new[i] != old[i]]
    else:
        changed = [(0, new)]
    if not changed:
        return False
    for i, value in changed:
        fcurve = fcurves.get(i)
        if fcurve is None or len(fcurve.keyframe_points) == 0:
            return False
        if abs(fcurve.evaluate(frame) - value) > _ANIMATION_MATCH_EPS * max(1.0, abs(value)):
            return False
    return True

def get_fcurves_changes(obj, settings):
    if obj.animation_data is None:
        return

    pre_update = _pre_update_registry.pop(obj.as_pointer(), None)
    if pre_update is None:
        return

    # skip on action or action_slot change
    if has_anim_attr_changed(obj, pre_update, 'action', 'name'):
        return
    if has_anim_attr_changed(obj, pre_update, 'action_slot', 'identifier'):
        return

    obj_fcurves = _get_obj_fcurves(obj)
    if obj_fcurves is None:
        return

    fcurve_map = dict()
    for fcurve in obj_fcurves:
        if not _is_editable_fcurve(fcurve, settings):
            continue
        fcurve_map.setdefault(fcurve.data_path, dict())[fcurve.array_index] = fcurve

    for data_path, old in pre_update['fcurves'].items():
        fcurves = fcurve_map.get(data_path)
        if not fcurves:
            continue
        new = read_numeric_property(obj, data_path)
        if new is None or isinstance(new, Vector) != isinstance(old, Vector):
            continue
        if isinstance(new, Vector) and len(new) != len(old):
            continue
        if new == old: # no change
            continue
        if _is_driven_by_animation(fcurves, old, new, settings.frame):
            continue
        yield (data_path, fcurves, old, new)

def save_fcurves_data(obj, depsgraph):
    eval_obj = obj.evaluated_get(depsgraph)
    if eval_obj.animation_data is None:
        return

    obj_fcurves = _get_obj_fcurves(eval_obj)
    if obj_fcurves is None:
        return

    fcurves_pre_update = dict()
    for fcurve in obj_fcurves:
        if fcurve.data_path in fcurves_pre_update or is_custom_property_path(fcurve.data_path):
            continue
        data = read_numeric_property(eval_obj, fcurve.data_path)
        if data is not None:
            fcurves_pre_update[fcurve.data_path] = data

    pre_update = dict()
    pre_update['fcurves'] = fcurves_pre_update
    save_anim_attr(eval_obj, pre_update, 'action', 'name')
    save_anim_attr(eval_obj, pre_update, 'action_slot', 'identifier')
    _pre_update_registry[obj.as_pointer()] = pre_update

# last seen values of custom properties: (ID pointer, data path) -> float | Vector
_custom_last: dict = {}
CUSTOM_POLL_INTERVAL = 0.1

def get_custom_property_changes(obj, settings):
    obj_fcurves = _get_obj_fcurves(obj)
    if not obj_fcurves:
        return

    driven = {d.data_path for d in obj.animation_data.drivers}
    by_path = dict()
    for fcurve in obj_fcurves:
        if is_custom_property_path(fcurve.data_path) and fcurve.data_path not in driven:
            by_path.setdefault(fcurve.data_path, dict())[fcurve.array_index] = fcurve

    pointer = obj.as_pointer()
    for data_path, fcurves in by_path.items():
        key = (pointer, data_path)
        new = read_numeric_property(obj, data_path)
        if new is None:
            _custom_last.pop(key, None)
            continue
        last = _custom_last.get(key)
        _custom_last[key] = new
        if last is None or isinstance(new, Vector) != isinstance(last, Vector):
            continue
        if isinstance(new, Vector) and len(new) != len(last):
            continue
        if new == last:
            continue
        editable = {i: fc for i, fc in fcurves.items() if _is_editable_fcurve(fc, settings)}
        if editable and not _is_driven_by_animation(editable, last, new, settings.frame):
            yield (data_path, editable, last, new)

def _get_custom_property_candidates(view_layer):
    active = view_layer.objects.active
    for obj in view_layer.objects:
        if obj.animation_data is not None and (obj == active or obj.select_get(view_layer=view_layer)):
            yield obj

def poll_custom_properties(scene, view_layer=None):
    """Offset keyframes for custom properties that changed since the last poll.
    Returns True if any keyframes were changed"""
    if not scene.use_anim_offset_mode or g_is_undo_redo_in_progress:
        return False
    if scene.tool_settings.use_keyframe_insert_auto:
        return False

    settings = _get_settings(scene)
    changed = False
    for obj in _get_custom_property_candidates(view_layer or bpy.context.view_layer):
        for change in get_custom_property_changes(obj, settings):
            apply_change(obj, *change, settings)
            changed = True
    return changed

def _custom_poll_timer():
    try:
        context = bpy.context
        screen = getattr(context, 'screen', None)
        if screen is not None and screen.is_animation_playing:
            return CUSTOM_POLL_INTERVAL
        scene = context.scene
        if scene is not None and scene.use_anim_offset_mode:
            if poll_custom_properties(scene):
                _redraw_editors()
    except Exception:
        traceback.print_exc()
    return CUSTOM_POLL_INTERVAL

def reset_runtime_state():
    """Forget everything tracked between updates"""
    end_sessions()
    _custom_last.clear()
    _pre_update_registry.clear()

g_is_undo_redo_in_progress = False

def is_valid_update(update):
    if update.id.id_type == 'OBJECT':
        return (update.is_updated_geometry or # this fires for e.g. pose bone transform update
                update.is_updated_transform)
    # other animatable data (shape keys, materials, camera, light...), for these
    # the "shading" flag is the only one set for e.g. material property edits
    return (update.is_updated_geometry or
            update.is_updated_transform or
            update.is_updated_shading)

def _get_updated_ids(depsgraph):
    for update in depsgraph.updates:
        if not is_valid_update(update):
            continue
        id_data = update.id.original
        # not every ID type is animatable (e.g. Action)
        if getattr(id_data, 'animation_data', None) is None:
            continue
        yield id_data

# In depsgraph_update_pre, depsgraph.updates lists the IDs tagged for the coming evaluation (that
# is how the edited ID is found before its new value is applied). But the depsgraph is only
# rebuilt during that evaluation, so if IDs were deleted since the last update (Outliner delete,
# purge, a script removing objects) their entries still point to freed memory, and reading them
# crashes Blender. Deletions are detected by comparing the number of data-blocks of each type
# with the last update; in that case the pre-update snapshot is skipped once. That update is the
# deletion itself, so there is no user edit to offset in it.
_ID_COLLECTION_NAMES = None
_id_counts_at_last_update = None

def _id_counts():
    global _ID_COLLECTION_NAMES
    if _ID_COLLECTION_NAMES is None:
        _ID_COLLECTION_NAMES = tuple(
            prop.identifier for prop in bpy.types.BlendData.bl_rna.properties
            if prop.type == 'COLLECTION' and prop.identifier not in {'window_managers', 'workspaces', 'screens'})
    return tuple(len(getattr(bpy.data, name)) for name in _ID_COLLECTION_NAMES)

def _remember_id_counts():
    global _id_counts_at_last_update
    _id_counts_at_last_update = _id_counts()

def _ids_deleted_since_last_update():
    if _id_counts_at_last_update is None:
        return True
    return any(now < before for now, before in zip(_id_counts(), _id_counts_at_last_update))

@persistent
def post_depsgraph_update(scene):
    _remember_id_counts()
    if not scene.use_anim_offset_mode:
        return
    if g_is_undo_redo_in_progress:
        return
    if scene.tool_settings.use_keyframe_insert_auto:
        return

    settings = _get_settings(scene)
    depsgraph = bpy.context.view_layer.depsgraph
    for id_data in _get_updated_ids(depsgraph):
        for change in get_fcurves_changes(id_data, settings):
            apply_change(id_data, *change, settings)

    _pre_update_registry.clear()

@persistent
def pre_depsgraph_update(scene):
    if not scene.use_anim_offset_mode:
        return
    if g_is_undo_redo_in_progress:
        return
    if _ids_deleted_since_last_update():
        return      # depsgraph.updates may point to freed IDs: see _id_counts_at_last_update
    depsgraph = bpy.context.view_layer.depsgraph
    for id_data in _get_updated_ids(depsgraph):
        save_fcurves_data(id_data, depsgraph)

@persistent
def on_frame_change(scene, depsgraph=None):
    # animation overwrites property values, this is not a user edit
    reset_runtime_state()

@persistent
def post_frame_change(scene, depsgraph=None):
    _remember_id_counts()

@persistent
def pre_redo_undo(scene):
    global g_is_undo_redo_in_progress
    g_is_undo_redo_in_progress = True

@persistent
def post_redo_undo(scene):
    global g_is_undo_redo_in_progress
    # trigger depsgraph update now (it would be done after post_redo/undo callback)
    # but this time skip fcurves update (as it has already been done in depsgraph update before post_redo/undo callback)
    bpy.context.evaluated_depsgraph_get()
    g_is_undo_redo_in_progress = False
    reset_runtime_state()

_msgbus_owner = object()

def _register_message_bus() -> None:
    bpy.msgbus.subscribe_rna(
        key=(bpy.types.ToolSettings, "use_keyframe_insert_auto"),
        owner=_msgbus_owner,
        args=(),
        notify=_on_auto_key_change,
        options={"PERSISTENT"},
    )

def _unregister_message_bus() -> None:
    bpy.msgbus.clear_by_owner(_msgbus_owner)

def _migrate_legacy_settings() -> None:
    """'Affect only selected keyframes' checkbox (<= 1.5) became the 'Affect keyframes' enum"""
    try:
        scenes = list(bpy.data.scenes)
    except (AttributeError, RuntimeError): # bpy.data is restricted while add-ons register on startup
        return
    for scene in scenes:
        if scene.get('anim_offset_mode_only_selected') is not None:
            if scene['anim_offset_mode_only_selected']:
                scene.anim_offset_target = TARGET_SELECTED
            del scene['anim_offset_mode_only_selected']
        # an easing window (<= 1.7) is what proportional falloff now is, and it was always on
        if scene.get('anim_offset_proportional') is None and scene.anim_offset_easing_window > 0:
            scene.anim_offset_proportional = True

@persistent
def post_load(none, other_none) -> None:
    _register_message_bus()
    _migrate_legacy_settings()
    reset_runtime_state()
    _remember_id_counts()

def _redraw_editors(context=None):
    context = context or bpy.context
    screen = getattr(context, 'screen', None)
    if screen is None:
        return
    for area in screen.areas:
        if area.type in {'VIEW_3D', 'GRAPH_EDITOR', 'DOPESHEET_EDITOR'}:
            area.tag_redraw()

class DOPESHEET_PT_anim_offset_mode(bpy.types.Panel):
    bl_idname = "DOPESHEET_PT_anim_offset_mode"
    bl_label = "Offset Editing"
    bl_space_type = 'DOPESHEET_EDITOR'
    bl_region_type = 'HEADER'

    @staticmethod
    def _selected_keyframes_are_sequential(obj):
        fcurves = _get_obj_fcurves(obj)
        if not fcurves:
            return False

        is_seq = False
        for fcurve in fcurves:
            size = len(fcurve.keyframe_points)
            if size == 0:
                continue
            if not fcurve.select:
                continue

            selected = np.empty(size, dtype=bool)
            fcurve.keyframe_points.foreach_get('select_control_point', selected)
            indices = np.where(selected)[0]

            if len(indices) == 0:
                continue

            if not _is_consecutive(indices):
                return False

            is_seq = True

        return is_seq

    @staticmethod
    def _selected_keyframes_are_non_consecutive(obj):
        fcurves = _get_obj_fcurves(obj)
        if not fcurves:
            return False

        for fcurve in fcurves:
            size = len(fcurve.keyframe_points)
            if size == 0:
                continue
            if not fcurve.select:
                continue

            selected = np.empty(size, dtype=bool)
            fcurve.keyframe_points.foreach_get('select_control_point', selected)
            indices = np.where(selected)[0]

            if len(indices) > 1 and not _is_consecutive(indices):
                return True

        return False

    def draw(self, context):
        scene = context.scene
        layout = self.layout
        col = layout.column(align=True)
        col.active = scene.use_anim_offset_mode
        col.label(text='Affect keyframes:')
        col.row(align=True).prop(scene, 'anim_offset_target', expand=True)
        col.label(text='From the current frame:')
        col.row(align=True).prop(scene, 'anim_offset_side', expand=True)
        if scene.anim_offset_target == TARGET_CURRENT:
            col.prop(scene, 'anim_offset_insert_missing')
        col.prop(scene, 'anim_offset_mode_only_selected_channels')

        target = scene.anim_offset_target
        only_sel = target != TARGET_ALL # easing applies to a range of keyframes
        is_sequential = True
        obj = context.active_object
        obj_is_valid = obj and obj.animation_data

        if target == TARGET_SELECTED and obj_is_valid:
            is_sequential = self._selected_keyframes_are_sequential(obj)

        proportional = scene.anim_offset_proportional
        layout.prop(scene, 'anim_offset_proportional', text='Proportional Falloff (Time)',
                    icon='PROP_ON' if proportional else 'PROP_OFF', toggle=True)

        col = layout.column(align=True)
        col.active = proportional and only_sel and is_sequential
        row = col.row(align=True)
        row.prop(scene, 'anim_offset_easing_window', text='Radius')
        _col = row.column()
        _col.operator('anim.adjust_easing_window', text='', icon='DRIVER_DISTANCE')
        _col.enabled = only_sel and is_sequential
        col.row(align=True).prop(scene, 'anim_offset_easing_unit', expand=True)

        if (target == TARGET_SELECTED and not is_sequential and proportional
            and scene.anim_offset_easing_window > 0
            and obj_is_valid and self._selected_keyframes_are_non_consecutive(obj)):
            warn = col.column(align=True)
            warn.alert = True
            warn.label(text='Selected keyframes are not', icon='ERROR')
            warn.label(text='in consecutive order', icon='BLANK1')

        row = col.row(align=False)
        row.active = proportional and only_sel and is_sequential and scene.anim_offset_easing_window > 0
        _col = row.column()
        _col.label(text='Falloff:')
        _col.prop_enum(scene, 'anim_offset_easing_function', 'smooth', icon='SMOOTHCURVE')
        _col.prop_enum(scene, 'anim_offset_easing_function', 'root', icon='ROOTCURVE')
        _col.prop_enum(scene, 'anim_offset_easing_function', 'constant', icon='NOCURVE')
        _col.prop_enum(scene, 'anim_offset_easing_function', 'linear', icon='IPO_LINEAR')
        _col.prop_enum(scene, 'anim_offset_easing_function', 'sine', icon='IPO_SINE')
        _col.prop_enum(scene, 'anim_offset_easing_function', 'quadratic', icon='IPO_QUAD')
        _col.prop_enum(scene, 'anim_offset_easing_function', 'cubic', icon='IPO_CUBIC')
        _col.prop_enum(scene, 'anim_offset_easing_function', 'quartic', icon='IPO_QUART')
        _col.prop_enum(scene, 'anim_offset_easing_function', 'quintic', icon='IPO_QUINT')
        _col.prop_enum(scene, 'anim_offset_easing_function', 'exponential', icon='IPO_EXPO')
        _col.prop_enum(scene, 'anim_offset_easing_function', 'circular', icon='IPO_CIRC')
        _col = row.column()
        _col.label(text='Mode:')
        _col.prop_enum(scene, 'anim_offset_easing_mode', 'in', icon='IPO_EASE_IN')
        _col.prop_enum(scene, 'anim_offset_easing_mode', 'out', icon='IPO_EASE_OUT')
        _col.prop_enum(scene, 'anim_offset_easing_mode', 'in_out', icon='IPO_EASE_IN_OUT')

        col = layout.column(align=True)
        col.active = scene.use_anim_offset_mode
        col.label(text='Rotation offset:')
        col.row(align=True).prop(scene, 'anim_offset_rotation_space', expand=True)

        layout.prop(scene, 'anim_offset_show_indicator')

# ---------------------------------------------------------------------------
# overlays
# ---------------------------------------------------------------------------

INDICATOR_COLOR_ACTIVE = (0.25, 0.75, 0.25, 0.9)
INDICATOR_COLOR_IDLE = (0.6, 0.6, 0.6, 0.6)

def _get_overlay_ranges(context, settings):
    """(first selected frame, last selected frame, first window frame, last window frame)
    over the F-Curves of the active object that are affected, None if there is nothing to show"""
    obj = context.active_object
    if not obj or not obj.animation_data:
        return None
    fcurves = _get_obj_fcurves(obj)
    if not fcurves:
        return None

    window = settings.easing_window
    by_frames = settings.easing_unit == UNIT_FRAMES

    if settings.target == TARGET_CURRENT and settings.insert_missing and window <= 0:
        # a key is inserted on the frame if there is none, no need to look at the keyframes
        if any(len(fc.keyframe_points) > 0 and _is_editable_fcurve(fc, settings) for fc in fcurves):
            return settings.frame, settings.frame, settings.frame, settings.frame
        return None

    min_sel = min_win = float('inf')
    max_sel = max_win = float('-inf')
    found = False

    for fcurve in fcurves:
        size = len(fcurve.keyframe_points)
        if size == 0:
            continue
        if settings.target == TARGET_CURRENT:
            if not _is_editable_fcurve(fcurve, settings):
                continue
        elif not fcurve.select:
            continue

        frames = _get_keyframe_frames(fcurve)
        if settings.target == TARGET_CURRENT:
            indices = np.where(np.abs(frames - settings.frame) < FRAME_TOLERANCE)[0]
            if len(indices) == 0:
                if not settings.insert_missing:
                    continue
                # no keyframe here (yet), it will be inserted between its neighbours
                insert_idx = int(np.searchsorted(frames, settings.frame))
                first_frame = last_frame = settings.frame
                if by_frames:
                    win_start, win_end = first_frame - window, last_frame + window
                else:
                    ws_idx = max(0, insert_idx - window)
                    we_idx = min(size - 1, insert_idx + window - 1)
                    win_start = frames[ws_idx] if ws_idx < insert_idx else settings.frame
                    win_end = frames[we_idx] if we_idx >= insert_idx else settings.frame
            else:
                first_idx, last_idx = int(indices[0]), int(indices[-1])
                first_frame, last_frame = frames[first_idx], frames[last_idx]
                if by_frames:
                    win_start, win_end = first_frame - window, last_frame + window
                else:
                    win_start = frames[max(0, first_idx - window)]
                    win_end = frames[min(size - 1, last_idx + window)]
        else:
            selected = np.empty(size, dtype=bool)
            fcurve.keyframe_points.foreach_get('select_control_point', selected)
            indices = np.where(selected)[0]
            if len(indices) == 0 or not _is_consecutive(indices):
                continue
            first_idx, last_idx = int(indices[0]), int(indices[-1])
            first_frame, last_frame = frames[first_idx], frames[last_idx]
            if by_frames:
                win_start, win_end = first_frame - window, last_frame + window
            else:
                win_start = frames[max(0, first_idx - window)]
                win_end = frames[min(size - 1, last_idx + window)]

        found = True
        min_sel, max_sel = min(min_sel, first_frame), max(max_sel, last_frame)
        min_win, max_win = min(min_win, win_start), max(max_win, win_end)

    if not found:
        return None

    # only one side of the current frame is affected
    if settings.side == SIDE_AFTER:
        min_sel, min_win = max(min_sel, settings.frame), max(min_win, settings.frame)
    elif settings.side == SIDE_BEFORE:
        max_sel, max_win = min(max_sel, settings.frame), min(max_win, settings.frame)
    if min_sel > max_sel:
        return None
    return min_sel, max_sel, min_win, max_win

OVERLAY_CACHE_SECONDS = 0.25
_overlay_cache = {'key': None, 'time': 0.0, 'value': None}

def _get_overlay_ranges_cached(context, settings):
    """Scanning all F-Curves of a big rig takes long, and this runs on every redraw of several editors"""
    obj = context.active_object
    key = (obj.as_pointer() if obj else 0, settings)
    now = time.monotonic()
    cache = _overlay_cache
    if cache['key'] == key and now - cache['time'] < OVERLAY_CACHE_SECONDS:
        return cache['value']
    value = _get_overlay_ranges(context, settings)
    cache['key'], cache['time'], cache['value'] = key, now, value
    return value

def _draw_rects(rects, color):
    shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    gpu.state.blend_set('ALPHA')
    shader.uniform_float("color", color)
    for x0, y0, x1, y1 in rects:
        if x1 <= x0 or y1 <= y0:
            continue
        verts = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        batch_for_shader(shader, 'TRI_FAN', {"pos": verts}).draw(shader)
    gpu.state.blend_set('NONE')

def draw_range_overlay():
    """Graph Editor / Dope Sheet: highlights the keyframes affected by the offset and the easing window"""
    context = bpy.context
    scene = context.scene
    if not scene.use_anim_offset_mode or not scene.anim_offset_show_indicator:
        return
    region = context.region
    if region is None or context.screen.is_animation_playing:
        return

    if scene.anim_offset_target == TARGET_ALL:
        if scene.anim_offset_side == SIDE_BOTH:
            return
        # everything on one side of the current frame
        x = region.view2d.view_to_region(float(scene.frame_current), 0.0, clip=False)[0]
        if scene.anim_offset_side == SIDE_AFTER:
            _draw_rects([(x, 0, region.width, region.height)], (0.25, 0.75, 0.25, 0.12))
        else:
            _draw_rects([(0, 0, x, region.height)], (0.25, 0.75, 0.25, 0.12))
        _draw_rects([(x - 2, 0, x + 2, region.height)], (0.25, 0.75, 0.25, 0.30))
        return

    ranges = _get_overlay_ranges_cached(context, _get_settings(scene))
    if ranges is None:
        return
    min_sel, max_sel, min_win, max_win = ranges

    view2d = region.view2d
    x_win_start = view2d.view_to_region(min_win, 0.0, clip=False)[0]
    x_win_end = view2d.view_to_region(max_win, 0.0, clip=False)[0]
    x_sel_start = view2d.view_to_region(min_sel, 0.0, clip=False)[0]
    x_sel_end = view2d.view_to_region(max_sel, 0.0, clip=False)[0]
    if x_sel_end - x_sel_start < 4: # a single frame, keep it visible
        x_sel_start -= 2
        x_sel_end += 2

    y0, y1 = 0, region.height
    _draw_rects([(x_win_start, y0, x_sel_start, y1), (x_sel_end, y0, x_win_end, y1)],
                (0.25, 0.75, 0.25, 0.12))
    _draw_rects([(x_sel_start, y0, x_sel_end, y1)], (0.25, 0.75, 0.25, 0.30))

def draw_viewport_indicator():
    """3D Viewport: border + label while Offset Editing is on"""
    context = bpy.context
    scene = context.scene
    if not scene.use_anim_offset_mode or not scene.anim_offset_show_indicator:
        return
    region = context.region
    if region is None:
        return

    active = not scene.tool_settings.use_keyframe_insert_auto
    color = INDICATOR_COLOR_ACTIVE if active else INDICATOR_COLOR_IDLE
    w, h, t = region.width, region.height, 3
    _draw_rects([(0, 0, w, t), (0, h - t, w, h), (0, 0, t, h), (w - t, 0, w, h)], color)

    text = 'OFFSET EDITING  |  ' + TARGET_LABELS[scene.anim_offset_target]
    if scene.anim_offset_side != SIDE_BOTH:
        text += f'  |  {SIDE_LABELS[scene.anim_offset_side]}'
    if (scene.anim_offset_target != TARGET_ALL and scene.anim_offset_proportional
            and scene.anim_offset_easing_window > 0):
        unit = 'frames' if scene.anim_offset_easing_unit == UNIT_FRAMES else 'keys'
        text += f'  |  proportional {scene.anim_offset_easing_window} {unit}'
    if not active:
        text += '  (paused: Auto Keying is on)'

    font_id = 0
    blf.size(font_id, 14.0)
    width, _height = blf.dimensions(font_id, text)
    x, y = (w - width) / 2, 36
    blf.color(font_id, 0.0, 0.0, 0.0, 0.8)
    blf.position(font_id, x + 1, y - 1, 0)
    blf.draw(font_id, text)
    blf.color(font_id, *color[:3], 1.0)
    blf.position(font_id, x, y, 0)
    blf.draw(font_id, text)

_draw_handlers = []

def _register_draw_handlers():
    for space, func in ((bpy.types.SpaceView3D, draw_viewport_indicator),
                        (bpy.types.SpaceGraphEditor, draw_range_overlay),
                        (bpy.types.SpaceDopeSheetEditor, draw_range_overlay)):
        _draw_handlers.append((space, space.draw_handler_add(func, (), 'WINDOW', 'POST_PIXEL')))

def _unregister_draw_handlers():
    for space, handler in _draw_handlers:
        space.draw_handler_remove(handler, 'WINDOW')
    _draw_handlers.clear()

class ANIM_OT_adjust_easing_window(bpy.types.Operator):
    bl_idname = "anim.adjust_easing_window"
    bl_label = "Adjust Proportional Radius"
    bl_description = "Click and drag left/right to adjust the radius of the proportional falloff"
    bl_options = {'BLOCKING', 'GRAB_CURSOR_X'}

    @classmethod
    def poll(cls, context):
        return (context.scene.use_anim_offset_mode and
                context.scene.anim_offset_target != TARGET_ALL)

    def invoke(self, context, event):
        self._init_mouse_x = event.mouse_x
        self._init_value = context.scene.anim_offset_easing_window
        context.window_manager.modal_handler_add(self)
        context.window.cursor_modal_set('MOVE_X')
        return {'RUNNING_MODAL'}

    def _cleanup(self, context):
        context.window.cursor_modal_restore()
        _redraw_editors(context)

    def modal(self, context, event):
        if event.type == 'ESC' or event.type == 'RIGHTMOUSE':
            self.cancel(context)
            return {'CANCELLED'}
        if event.type == 'LEFTMOUSE' and event.value == 'RELEASE':
            self.finish(context)
            return {'FINISHED'}

        if event.type == 'MOUSEMOVE':
            delta = int((event.mouse_x - self._init_mouse_x) / 10)
            new_value = max(0, self._init_value + delta)
            context.scene.anim_offset_easing_window = new_value
            _redraw_editors(context)

        return {'RUNNING_MODAL'}

    def finish(self, context):
        self._cleanup(context)

    def cancel(self, context):
        context.scene.anim_offset_easing_window = self._init_value
        self._cleanup(context)

g_auto_key_change_active = False
g_anim_offset_mode_change_active = False

def _on_anim_offset_mode_change(scene, context):
    reset_runtime_state()
    _redraw_editors(context)

    if not get_is_auto_key_override(context):
        return
    if g_auto_key_change_active:
        return

    # XXX: not cleared because message_bus will trigger the _on_auto_key_change() callback AFTER this one
    global g_anim_offset_mode_change_active
    g_anim_offset_mode_change_active = True

    tool_settings = scene.tool_settings
    if scene.use_anim_offset_mode:
        scene.use_keyframe_insert_auto_old = tool_settings.use_keyframe_insert_auto # save
        tool_settings.use_keyframe_insert_auto = False
    else:
        tool_settings.use_keyframe_insert_auto = scene.use_keyframe_insert_auto_old # restore

def _on_anim_offset_setting_change(scene, context):
    end_sessions()
    _redraw_editors(context)

def _on_auto_key_change():
    if not get_is_auto_key_override(bpy.context):
        return

    global g_anim_offset_mode_change_active
    if g_anim_offset_mode_change_active:
        g_anim_offset_mode_change_active = False
        return

    global g_auto_key_change_active
    g_auto_key_change_active = True

    tool_settings = bpy.context.tool_settings
    scene = bpy.context.scene
    if tool_settings.use_keyframe_insert_auto:
        scene.use_anim_offset_mode = False

    g_auto_key_change_active = False

def draw_header(self, context):
    # st = context.space_data
    # if st.mode == 'TIMELINE': # don't show in timeline
    #     return

    row = self.layout.row(align=True)
    row.prop(context.scene, "use_anim_offset_mode", icon='CON_TRANSLIKE', toggle=True, text='')
    row.active = not context.scene.tool_settings.use_keyframe_insert_auto
    sub = row.row(align=True)
    sub.prop(context.scene, "anim_offset_proportional", text='', toggle=True,
             icon='PROP_ON' if context.scene.anim_offset_proportional else 'PROP_OFF')
    sub.popover(
        panel=DOPESHEET_PT_anim_offset_mode.bl_idname,
        text="",
    )
    sub.active = context.scene.use_anim_offset_mode

def register():
    bpy.types.Scene.use_keyframe_insert_auto_old = BoolProperty () # internal

    bpy.types.Scene.use_anim_offset_mode = BoolProperty (
            name="Offset editing",
            description="Apply property value changes as an offset to all keyframe points\n"
                        "NOTE: This option only takes effect when 'Auto keying' is disabled",
            default=False,
            update=_on_anim_offset_mode_change
        )
    bpy.types.Scene.anim_offset_target = EnumProperty (
            name="Affect keyframes",
            description="Which keyframe points are affected when Offset editing is enabled",
            items=[
                (TARGET_ALL, 'All', 'Offset all keyframe points'),
                (TARGET_SELECTED, 'Selected', 'Offset only the selected keyframe points'),
                (TARGET_CURRENT, 'Current Frame', 'Offset only the keyframe on the current frame (use the easing window to blend it into the neighbouring keyframes)'),
            ],
            default=TARGET_ALL,
            update=_on_anim_offset_setting_change,
        )
    bpy.types.Scene.anim_offset_side = EnumProperty (
            name="Affected side",
            description="Which side of the current frame is affected. The keyframe on the current frame itself is always included",
            items=[
                (SIDE_BOTH, 'Both', 'Keyframes before and after the current frame'),
                (SIDE_AFTER, 'Forward', 'Only keyframes on and after the current frame'),
                (SIDE_BEFORE, 'Backward', 'Only keyframes on and before the current frame'),
            ],
            default=SIDE_BOTH,
            update=_on_anim_offset_setting_change,
        )
    bpy.types.Scene.anim_offset_proportional = BoolProperty (
            name="Proportional falloff (time)",
            description="Like proportional editing, but in time: the offset also reaches the keyframes around the affected ones "
                        "and fades out over the radius, following the chosen falloff.\n"
                        "NOTE: Has no effect when all keyframes are affected",
            default=False,
            update=_on_anim_offset_setting_change,
        )
    bpy.types.Scene.anim_offset_insert_missing = BoolProperty (
            name="Insert key if missing",
            description="When there is no keyframe on the current frame, insert one (on every animated channel of the edited property) "
                        "so that it receives the change, just like Auto Keying would.\n"
                        "NOTE: Cancelling a transform removes the inserted keyframe again",
            default=True,
            update=_on_anim_offset_setting_change,
        )
    bpy.types.Scene.anim_offset_mode_only_selected_channels = BoolProperty (
            name="Affect only selected channels",
            description="Affect only F-Curves (channels) that are selected in the Graph Editor or Dope Sheet.\n"
                        "NOTE: Locked and muted channels are never affected",
            default=False,
            update=_on_anim_offset_setting_change,
        )
    bpy.types.Scene.anim_offset_rotation_space = EnumProperty (
            name="Rotation offset space",
            description="How a change of a rotation is applied to the existing rotation keyframes",
            items=[
                ('PARENT', 'Parent', 'Rotate every keyframe by the same rotation in the space of the parent'),
                ('LOCAL', 'Local', 'Rotate every keyframe by the same rotation in its own local space'),
            ],
            default='PARENT',
            update=_on_anim_offset_setting_change,
        )
    bpy.types.Scene.anim_offset_show_indicator = BoolProperty (
            name="Show indicator",
            description="Show a border and label in the 3D Viewport while Offset editing is enabled, "
                        "and highlight the affected keyframe range in the Graph Editor and Dope Sheet",
            default=True,
            update=lambda scene, context: _redraw_editors(context),
        )
    bpy.types.Scene.anim_offset_easing_window = IntProperty (
            name="Easing window",
            description="Size of the window before and after the affected keyframes in which the applied offset is blended out\n"
                        "NOTE: This option only takes effect when the selected keyframes (for all F-Curves) are in consecutive order",
            default=0,
            min=0,
            max=9999,
            update=_on_anim_offset_setting_change,
        )
    bpy.types.Scene.anim_offset_easing_unit = EnumProperty (
            name="Easing window unit",
            description="What the size of the easing window is measured in",
            items=[
                (UNIT_KEYS, 'Keys', 'Number of keyframes on each side'),
                (UNIT_FRAMES, 'Frames', 'Number of frames on each side, regardless of how many keyframes are in there'),
            ],
            default=UNIT_KEYS,
            update=_on_anim_offset_setting_change,
        )
    bpy.types.Scene.anim_offset_easing_function = EnumProperty (
            name="Easing function",
            description="Curve shape for the easing blend",
            items=[
                ('smooth', 'Smooth', 'Smooth falloff (like proportional editing "Smooth")'),
                ('root', 'Root', 'Square root falloff (like proportional editing "Root")'),
                ('constant', 'Constant', 'Full offset everywhere inside of the radius'),
                ('linear', 'Linear', 'Linear blend'),
                ('sine', 'Sine', 'Sine curve'),
                ('quadratic', 'Quadratic', 'Quadratic curve'),
                ('cubic', 'Cubic', 'Cubic curve'),
                ('quartic', 'Quartic', 'Quartic curve'),
                ('quintic', 'Quintic', 'Quintic curve'),
                ('exponential', 'Exponential', 'Exponential curve'),
                ('circular', 'Circular', 'Circular curve'),
            ],
            default='linear',
            update=_on_anim_offset_setting_change,
        )
    bpy.types.Scene.anim_offset_easing_mode = EnumProperty (
            name="Easing mode",
            description="Easing direction applied to the blend curve",
            items=[
                ('in', 'In', 'Ease in (slow start, fast end)'),
                ('out', 'Out', 'Ease out (fast start, slow end)'),
                ('in_out', 'In & Out', 'Ease in and out (S-curve)'),
            ],
            default='in_out',
            update=_on_anim_offset_setting_change,
        )

    bpy.utils.register_class(AnimAutoOffsetPreferences)
    bpy.utils.register_class(DOPESHEET_PT_anim_offset_mode)
    bpy.utils.register_class(ANIM_OT_adjust_easing_window)
    _register_message_bus()
    _migrate_legacy_settings()
    _register_draw_handlers()
    bpy.types.GRAPH_HT_header.append(draw_header)
    bpy.types.DOPESHEET_HT_header.append(draw_header)
    bpy.app.handlers.depsgraph_update_post.append(post_depsgraph_update)
    bpy.app.handlers.depsgraph_update_pre.append(pre_depsgraph_update)
    bpy.app.handlers.frame_change_pre.append(on_frame_change)
    bpy.app.handlers.frame_change_post.append(post_frame_change)
    bpy.app.handlers.undo_post.append(post_redo_undo)
    bpy.app.handlers.redo_post.append(post_redo_undo)
    bpy.app.handlers.undo_pre.append(pre_redo_undo)
    bpy.app.handlers.redo_pre.append(pre_redo_undo)
    bpy.app.handlers.load_post.append(post_load)
    bpy.app.timers.register(_custom_poll_timer, first_interval=CUSTOM_POLL_INTERVAL, persistent=True)

def unregister():
    if bpy.app.timers.is_registered(_custom_poll_timer):
        bpy.app.timers.unregister(_custom_poll_timer)
    if bpy.app.timers.is_registered(_session_idle_timer):
        bpy.app.timers.unregister(_session_idle_timer)
    reset_runtime_state()
    bpy.app.handlers.load_post.remove(post_load)
    bpy.app.handlers.redo_pre.remove(pre_redo_undo)
    bpy.app.handlers.undo_pre.remove(pre_redo_undo)
    bpy.app.handlers.redo_post.remove(post_redo_undo)
    bpy.app.handlers.undo_post.remove(post_redo_undo)
    bpy.app.handlers.frame_change_pre.remove(on_frame_change)
    bpy.app.handlers.frame_change_post.remove(post_frame_change)
    bpy.app.handlers.depsgraph_update_pre.remove(pre_depsgraph_update)
    bpy.app.handlers.depsgraph_update_post.remove(post_depsgraph_update)
    bpy.types.DOPESHEET_HT_header.remove(draw_header)
    bpy.types.GRAPH_HT_header.remove(draw_header)
    _unregister_draw_handlers()
    _unregister_message_bus()
    bpy.utils.unregister_class(ANIM_OT_adjust_easing_window)
    bpy.utils.unregister_class(DOPESHEET_PT_anim_offset_mode)
    bpy.utils.unregister_class(AnimAutoOffsetPreferences)

    del bpy.types.Scene.anim_offset_easing_mode
    del bpy.types.Scene.anim_offset_easing_function
    del bpy.types.Scene.anim_offset_easing_unit
    del bpy.types.Scene.anim_offset_easing_window
    del bpy.types.Scene.anim_offset_show_indicator
    del bpy.types.Scene.anim_offset_rotation_space
    del bpy.types.Scene.anim_offset_mode_only_selected_channels
    del bpy.types.Scene.anim_offset_insert_missing
    del bpy.types.Scene.anim_offset_proportional
    del bpy.types.Scene.anim_offset_side
    del bpy.types.Scene.anim_offset_target
    del bpy.types.Scene.use_anim_offset_mode
    del bpy.types.Scene.use_keyframe_insert_auto_old
