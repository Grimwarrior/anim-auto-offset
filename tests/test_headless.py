"""Headless tests, run with:

    blender -b --factory-startup --python tests/test_headless.py

Loads the add-on straight from the repository folder, exits with code 1 if any check fails.
"""
import sys, os, importlib.util, math
import bpy
import numpy as np
from mathutils import Vector, Quaternion, Euler

baseline = False
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location(
    'anim_auto_offset', os.path.join(_root, '__init__.py'),
    submodule_search_locations=[_root])
addon = importlib.util.module_from_spec(_spec)
sys.modules['anim_auto_offset'] = addon
_spec.loader.exec_module(addon)
addon.register()

results = []
def check(name, cond, extra=None):
    results.append((name, bool(cond)))
    print(('PASS ' if cond else 'FAIL ') + name + (('  ' + str(extra)) if (not cond and extra is not None) else ''))

def approx(a, b, tol=1e-4):
    return np.allclose(np.array(a, dtype=float), np.array(b, dtype=float), atol=tol)

def reset():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    # factory reset drops non persistent handlers, addon handlers are @persistent but re-register for safety
    s = bpy.context.scene
    s.tool_settings.use_keyframe_insert_auto = False
    s.use_anim_offset_mode = True
    s.anim_offset_proportional = True
    return s

def new_cube(name='Cube'):
    me = bpy.data.meshes.new(name)
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    bpy.context.view_layer.objects.active = ob
    return ob

def fcurves(id_data):
    return addon._get_obj_fcurves(id_data)

def keys(id_data, path, idx):
    fc = fcurves(id_data).find(path, index=idx)
    return [kp.co[1] for kp in fc.keyframe_points]

def update():
    bpy.context.view_layer.update()

# ---------------------------------------------------------------- scrubbing
s = reset()
ob = new_cube()
for f, x in ((1, 0.0), (10, 5.0), (20, 10.0)):
    s.frame_set(f); ob.location.x = x; ob.keyframe_insert('location', index=0, frame=f)
update()
before = keys(ob, 'location', 0)
for f in (1, 5, 10, 15, 20, 3, 12):
    s.frame_set(f)
    update()
check('scrubbing does not change keys', approx(keys(ob, 'location', 0), before), keys(ob, 'location', 0))

# ---------------------------------------------------------------- plain offset
s.frame_set(10)
ob.location.x = 7.0
update()
check('offset all keys', approx(keys(ob, 'location', 0), [2, 7, 12]), keys(ob, 'location', 0))

if baseline:
    print('baseline run, remaining tests are for the new features'); sys.exit(0)

# ---------------------------------------------------------------- easing weights + off-by-one
s = reset()
ob = new_cube()
for i in range(6):
    ob.location.x = 0.0
    ob.keyframe_insert('location', index=0, frame=10 + i * 10)
fc = fcurves(ob).find('location', index=0)
for i, kp in enumerate(fc.keyframe_points):
    sel = i in (2, 3)
    kp.select_control_point = kp.select_left_handle = kp.select_right_handle = sel
s.anim_offset_target = 'SELECTED'
s.anim_offset_easing_window = 2
s.anim_offset_easing_function = 'linear'
s.anim_offset_easing_mode = 'in_out'
s.frame_set(30)
ob.location.x = 3.0
update()
check('easing window uses the full window (no zero-weight neighbour)',
      approx(keys(ob, 'location', 0), [1, 2, 3, 3, 2, 1]), keys(ob, 'location', 0))

w = addon._get_easing_weights(fc, np.array([False, False, True, True, False, False]), 1, 'linear', 'in_out')
check('window of 1 moves 1 neighbour each side', (w[1] > 0) and (w[4] > 0) and w[0] == 0 and w[5] == 0, w)

E = addon.EASINGS
check('sine in is ease-in (slow start)', E['sine']['in'](0.5) < 0.5 < E['sine']['out'](0.5))
check('sine in/out values', approx([E['sine']['in'](0.5), E['sine']['out'](0.5)], [1 - math.cos(math.pi / 4), math.sin(math.pi / 4)]))

# handles follow when auto handles are recalculated (no stale handles)
fc.update()
ys = [kp.co[1] for kp in fc.keyframe_points]
check('handles recalculated after offset', all(np.isfinite([kp.handle_left[1] for kp in fc.keyframe_points])))

# ---------------------------------------------------------------- rotations
def key_q(ob, kind, order='XYZ'):
    path = 'rotation_euler' if kind == 'EULER' else 'rotation_quaternion'
    n = 3 if kind == 'EULER' else 4
    ks = [keys(ob, path, i) for i in range(n)]
    out = []
    for k in range(len(ks[0])):
        v = [ks[i][k] for i in range(n)]
        out.append(Euler(v, order).to_quaternion() if kind == 'EULER' else Quaternion(v).normalized())
    return out

def same_rot(a, b, tol=1e-4):
    return abs(abs(a.dot(b)) - 1.0) < tol

for space in ('PARENT', 'LOCAL'):
    for kind in ('EULER', 'QUATERNION'):
        s = reset()
        s.anim_offset_rotation_space = space
        ob = new_cube()
        ob.rotation_mode = 'XYZ' if kind == 'EULER' else 'QUATERNION'
        path = 'rotation_euler' if kind == 'EULER' else 'rotation_quaternion'
        orig = [Euler((0.2, 0.5, -0.3)), Euler((0.9, -0.4, 0.7)), Euler((-0.5, 0.3, 1.1))]
        for f, e in zip((1, 10, 20), orig):
            if kind == 'EULER':
                ob.rotation_euler = e
            else:
                ob.rotation_quaternion = e.to_quaternion()
            ob.keyframe_insert(path, frame=f)
        s.frame_set(10)
        cur = ob.rotation_euler.to_quaternion() if kind == 'EULER' else Quaternion(ob.rotation_quaternion)
        twist = Quaternion((0.3, 0.8, -0.2), math.radians(40))
        new = twist @ cur if space == 'PARENT' else cur @ twist
        if kind == 'EULER':
            ob.rotation_euler = new.to_euler('XYZ', ob.rotation_euler)
        else:
            ob.rotation_quaternion = new
        update()
        kq = key_q(ob, kind)
        oq = [e.to_quaternion() for e in orig]
        expected = [(twist @ q) if space == 'PARENT' else (q @ twist) for q in oq]
        ok = all(same_rot(a, b) for a, b in zip(kq, expected))
        check(f'{kind} rotation offset ({space})', ok, [(a, b) for a, b in zip(kq, expected)])
        check(f'{kind} current key equals the edited pose ({space})', same_rot(kq[1], new))

# euler big angle crossing 180 degrees must not flip keys wildly
s = reset()
ob = new_cube()
ob.rotation_euler = (math.radians(170), 0, 0)
ob.keyframe_insert('rotation_euler', frame=1)
ob.rotation_euler = (math.radians(175), 0, 0)
ob.keyframe_insert('rotation_euler', frame=10)
s.frame_set(10)
ob.rotation_euler = (math.radians(185), 0, 0)
update()
xs = keys(ob, 'rotation_euler', 0)
check('euler keys keep continuity across 180', all(abs(x) < math.radians(400) and abs(xs[1] - xs[0]) < math.radians(20) for x in xs), [math.degrees(x) for x in xs])

# ---------------------------------------------------------------- discrete / locked / selected channels
s = reset()
ob = new_cube()
ob.hide_viewport = False
ob.keyframe_insert('hide_viewport', frame=1)
ob.hide_viewport = True
ob.keyframe_insert('hide_viewport', frame=10)
check('bool is not a continuous property', addon.read_numeric_property(ob, 'hide_viewport') is None)
check('float is', addon.read_numeric_property(ob, 'location') is not None)
before = keys(ob, 'hide_viewport', 0)
s.frame_set(10)
ob.hide_viewport = False
update()
check('bool keys untouched', approx(keys(ob, 'hide_viewport', 0), before))

s = reset()
ob = new_cube()
for f in (1, 10):
    ob.location = (0, 0, 0)
    ob.keyframe_insert('location', frame=f)
fcurves(ob).find('location', index=1).lock = True
fcurves(ob).find('location', index=2).mute = True
s.frame_set(10)
ob.location = (1, 1, 1)
update()
check('unlocked channel offset', approx(keys(ob, 'location', 0), [1, 1]))
check('locked channel untouched', approx(keys(ob, 'location', 1), [0, 0]))
check('muted channel untouched', approx(keys(ob, 'location', 2), [0, 0]))

s = reset()
ob = new_cube()
for f in (1, 10):
    ob.location = (0, 0, 0)
    ob.keyframe_insert('location', frame=f)
for i in range(3):
    fcurves(ob).find('location', index=i).select = (i == 0)
s.anim_offset_mode_only_selected_channels = True
s.frame_set(10)
ob.location = (1, 1, 1)
update()
check('only selected channels', approx(keys(ob, 'location', 0), [1, 1]) and approx(keys(ob, 'location', 1), [0, 0]))

# ---------------------------------------------------------------- non object IDs
s = reset()
cam_d = bpy.data.cameras.new('Cam')
cam = bpy.data.objects.new('Cam', cam_d)
s.collection.objects.link(cam)
for f, v in ((1, 50.0), (10, 60.0)):
    cam_d.lens = v
    cam_d.keyframe_insert('lens', frame=f)
s.frame_set(10)
cam_d.lens = 70.0
update()
check('camera data (lens) offset', approx(keys(cam_d, 'lens', 0), [60, 70]), keys(cam_d, 'lens', 0))

s = reset()
ob = new_cube()
ob.shape_key_add(name='Basis')
sk = ob.shape_key_add(name='Smile')
key = ob.data.shape_keys
for f, v in ((1, 0.0), (10, 0.5)):
    sk.value = v
    sk.keyframe_insert('value', frame=f)
s.frame_set(10)
sk.value = 0.8
update()
check('shape key value offset', approx(keys(key, 'key_blocks["Smile"].value', 0), [0.3, 0.8]), keys(key, 'key_blocks["Smile"].value', 0))

# ---------------------------------------------------------------- pose bone, ZYX order
s = reset()
arm_d = bpy.data.armatures.new('Arm')
arm = bpy.data.objects.new('Arm', arm_d)
s.collection.objects.link(arm)
bpy.context.view_layer.objects.active = arm
bpy.ops.object.mode_set(mode='EDIT')
eb = arm_d.edit_bones.new('B.001'); eb.head = (0, 0, 0); eb.tail = (0, 0, 1)
bpy.ops.object.mode_set(mode='POSE')
pb = arm.pose.bones['B.001']
pb.rotation_mode = 'ZYX'
orig = [Euler((0.2, 0.5, -0.3), 'ZYX'), Euler((0.9, -0.4, 0.7), 'ZYX'), Euler((-0.5, 0.3, 1.1), 'ZYX')]
for f, e in zip((1, 10, 20), orig):
    pb.rotation_euler = e
    pb.keyframe_insert('rotation_euler', frame=f)
s.frame_set(10)
twist = Quaternion((1, 0, 0), math.radians(25))
new = twist @ pb.rotation_euler.to_quaternion()
pb.rotation_euler = new.to_euler('ZYX', pb.rotation_euler)
update()
path = 'pose.bones["B.001"].rotation_euler'
ks = [keys(arm, path, i) for i in range(3)]
got = [Euler([ks[i][k] for i in range(3)], 'ZYX').to_quaternion() for k in range(3)]
exp = [twist @ e.to_quaternion() for e in orig]
check('pose bone ZYX euler offset', all(same_rot(a, b) for a, b in zip(got, exp)), (got, exp))

# ---------------------------------------------------------------- misaligned rotation keys -> per channel fallback
s = reset()
ob = new_cube()
for f, v in ((1, 0.0), (10, 0.5)):
    ob.rotation_euler = (v, 0.0, 0.0); ob.keyframe_insert('rotation_euler', index=0, frame=f)
for f, v in ((1, 0.0), (7, 0.2), (10, 0.5)):
    ob.rotation_euler = (0.0, v, 0.0); ob.keyframe_insert('rotation_euler', index=1, frame=f)
ob.rotation_euler = (0.0, 0.0, 0.0); ob.keyframe_insert('rotation_euler', index=2, frame=1)
s.frame_set(10)
ob.rotation_euler = (0.6, 0.5, 0.0)
update()
check('misaligned rotation keys fall back to additive', approx(keys(ob, 'rotation_euler', 0), [0.1, 0.6]) , keys(ob, 'rotation_euler', 0))

# ---------------------------------------------------------------- current frame mode
def loc_scene(frames_values, **kw):
    s = reset()
    s.anim_offset_target = 'CURRENT'
    for k, v in kw.items():
        setattr(s, k, v)
    ob = new_cube()
    for f, x in frames_values:
        ob.location = (x, 0, 0)
        ob.keyframe_insert('location', index=0, frame=f)
    return s, ob

def frames_of(ob, path='location', idx=0):
    return [kp.co[0] for kp in fcurves(ob).find(path, index=idx).keyframe_points]

# key exists on the current frame: only that key moves
s, ob = loc_scene([(1, 0.0), (10, 5.0), (20, 10.0)])
s.frame_set(10)
ob.location.x = 7.0
update()
check('current frame: existing key offset, others untouched', approx(keys(ob, 'location', 0), [0, 7, 10]), keys(ob, 'location', 0))

# no key on the current frame: key is inserted and holds the new value
s, ob = loc_scene([(1, 0.0), (10, 9.0), (20, 18.0)])
s.frame_set(5)
fc = fcurves(ob).find('location', index=0)
before = [kp.co[1] for kp in fc.keyframe_points]
ob.location.x = 4.0
update()
fc = fcurves(ob).find('location', index=0)
check('current frame: key inserted', frames_of(ob) == [1, 5, 10, 20], frames_of(ob))
check('current frame: inserted key holds the new value', approx(fc.evaluate(5), 4.0) and approx(keys(ob, 'location', 0)[1], 4.0), keys(ob, 'location', 0))
check('current frame: other keys untouched', approx([keys(ob, 'location', 0)[i] for i in (0, 2, 3)], before))

# dragging: several consecutive updates must produce a single key
s, ob = loc_scene([(1, 0.0), (10, 9.0), (20, 18.0)])
s.frame_set(5)
for x in (1.0, 2.0, 3.0, 4.0):
    ob.location.x = x
    update()
check('current frame: drag inserts one key with the final value',
      frames_of(ob) == [1, 5, 10, 20] and approx(keys(ob, 'location', 0)[1], 4.0), (frames_of(ob), keys(ob, 'location', 0)))

# insert_missing disabled: nothing happens without a key
s, ob = loc_scene([(1, 0.0), (10, 9.0), (20, 18.0)], anim_offset_insert_missing=False)
s.frame_set(5)
ob.location.x = 4.0
update()
check('current frame: no key inserted when disabled', frames_of(ob) == [1, 10, 20] and approx(keys(ob, 'location', 0), [0, 9, 18]))

# easing window blends the neighbours, with and without an existing key
s, ob = loc_scene([(10, 0.0), (20, 0.0), (30, 0.0), (40, 0.0), (50, 0.0)], anim_offset_easing_window=2, anim_offset_easing_function='linear')
s.frame_set(30)
ob.location.x = 3.0
update()
check('current frame + easing window (existing key)', approx(keys(ob, 'location', 0), [1, 2, 3, 2, 1]), keys(ob, 'location', 0))

s, ob = loc_scene([(10, 0.0), (20, 0.0), (30, 0.0), (40, 0.0)], anim_offset_easing_window=1, anim_offset_easing_function='linear')
s.frame_set(25)
ob.location.x = 3.0
update()
k = keys(ob, 'location', 0)
check('current frame + easing window (inserted key)', frames_of(ob) == [10, 20, 25, 30, 40] and approx(k[2], 3.0) and 0 < k[1] < 3 and 0 < k[3] < 3 and k[0] == 0 and k[4] == 0, k)

# locked channel does not get a key
s, ob = loc_scene([(1, 0.0), (10, 0.0)])
for f in (1, 10):
    ob.location = (0, 0, 0)
    ob.keyframe_insert('location', index=1, frame=f)
fcurves(ob).find('location', index=1).lock = True
s.frame_set(5)
ob.location = (2, 2, 0)
update()
check('current frame: locked channel gets no key', frames_of(ob, idx=0) == [1, 5, 10] and frames_of(ob, idx=1) == [1, 10])

# rotation: key inserted on all channels and holds the edited pose
s = reset()
s.anim_offset_target = 'CURRENT'
ob = new_cube()
a, b = Euler((0.2, 0.5, -0.3)), Euler((0.9, -0.4, 0.7))
for f, e in ((1, a), (20, b)):
    ob.rotation_euler = e
    ob.keyframe_insert('rotation_euler', frame=f)
s.frame_set(10)
cur = ob.rotation_euler.to_quaternion()
new = Quaternion((0.3, 0.8, -0.2), math.radians(30)) @ cur
ob.rotation_euler = new.to_euler('XYZ', ob.rotation_euler)
update()
ok_frames = all(frames_of(ob, 'rotation_euler', i) == [1, 10, 20] for i in range(3))
kq = key_q(ob, 'EULER')
check('current frame rotation: key on all channels', ok_frames, [frames_of(ob, 'rotation_euler', i) for i in range(3)])
check('current frame rotation: key holds the edited pose', same_rot(kq[1], new))
check('current frame rotation: other keys untouched', same_rot(kq[0], a.to_quaternion()) and same_rot(kq[2], b.to_quaternion()))

# migration of the old checkbox
s = reset()
s['anim_offset_mode_only_selected'] = True
addon._migrate_legacy_settings()
check('legacy "only selected" checkbox migrates to the enum', s.anim_offset_target == 'SELECTED' and 'anim_offset_mode_only_selected' not in s.keys())

# ================================================================ sessions (snapshot based transform)
def key_arrays(ob, path='location', idx=0):
    fc = fcurves(ob).find(path, index=idx)
    return np.array([[kp.co[0], kp.co[1]] for kp in fc.keyframe_points])

# no drift after a long drag there and back
s, ob = loc_scene([(1, 0.0), (10, 5.0), (20, 10.0)], anim_offset_target='ALL')
s.frame_set(10)
start = key_arrays(ob).copy()
for _ in range(60):
    ob.location.x += 0.013; update()
for _ in range(60):
    ob.location.x -= 0.013; update()
check('session: no drift after a long drag there and back', np.allclose(key_arrays(ob), start, atol=1e-5), key_arrays(ob)[:, 1] - start[:, 1])

# cancelling (value goes back to the start) restores the keys exactly, also with easing
s, ob = loc_scene([(10, 0.0), (20, 0.0), (30, 0.0), (40, 0.0), (50, 0.0)], anim_offset_easing_window=2)
s.frame_set(30)
start_value = ob.location.x
start = key_arrays(ob).copy()
for x in (0.5, 1.7, 3.0, 2.2):
    ob.location.x = x; update()
check('session: drag changed the keys', not np.allclose(key_arrays(ob), start))
ob.location.x = start_value; update()
check('session: cancel restores keys exactly', np.array_equal(key_arrays(ob), start), key_arrays(ob)[:, 1])

# cancel removes the key that was inserted by the session, and a new drag re-inserts it
s, ob = loc_scene([(1, 0.0), (10, 9.0), (20, 18.0)])
s.frame_set(5)
start_value = ob.location.x
start = key_arrays(ob).copy()
ob.location.x = start_value + 1.0; update()
check('session: key inserted during drag', frames_of(ob) == [1, 5, 10, 20])
ob.location.x = start_value; update()
check('session: cancel removes the inserted key', frames_of(ob) == [1, 10, 20] and np.allclose(key_arrays(ob), start, atol=1e-6), frames_of(ob))
ob.location.x = start_value + 2.0; update()
check('session: dragging again re-inserts it', frames_of(ob) == [1, 5, 10, 20] and approx(keys(ob, 'location', 0)[1], start_value + 2.0), (frames_of(ob), keys(ob, 'location', 0)))

# a session is dropped on frame change and when settings change
s, ob = loc_scene([(1, 0.0), (10, 5.0), (20, 10.0)], anim_offset_target='ALL')
s.frame_set(10)
ob.location.x += 1.0; update()
check('session exists after an edit', len(addon._sessions) == 1)
s.frame_set(12)
check('session dropped on frame change', len(addon._sessions) == 0)
s.frame_set(10)
ob.location.x += 1.0; update()
s.anim_offset_easing_window = 3
check('session dropped when settings change', len(addon._sessions) == 0)
ob.location.x += 1.0; update()
addon._last_session_activity -= 10
check('idle sessions are dropped', addon._session_idle_timer() is None and len(addon._sessions) == 0)

# keyframes edited behind the session's back must not break anything
s, ob = loc_scene([(1, 0.0), (10, 5.0), (20, 10.0)], anim_offset_target='ALL')
s.frame_set(10)
ob.location.x += 1.0; update()
fcurves(ob).find('location', index=0).keyframe_points.insert(15, 1.0)
try:
    ob.location.x += 1.0; update()
    ok = True
except Exception as e:
    ok = False
check('session survives manual keyframe edits', ok and len(addon._sessions) == 1)

# rotations: cancel is exact too
for kind in ('EULER', 'QUATERNION'):
    s = reset()
    s.anim_offset_target = 'ALL'
    ob = new_cube()
    ob.rotation_mode = 'XYZ' if kind == 'EULER' else 'QUATERNION'
    path = 'rotation_euler' if kind == 'EULER' else 'rotation_quaternion'
    for f, e in zip((1, 10, 20), (Euler((0.2, 0.5, -0.3)), Euler((0.9, -0.4, 0.7)), Euler((-0.5, 0.3, 1.1)))):
        if kind == 'EULER':
            ob.rotation_euler = e
        else:
            ob.rotation_quaternion = e.to_quaternion()
        ob.keyframe_insert(path, frame=f)
    s.frame_set(10)
    n = 3 if kind == 'EULER' else 4
    start = [keys(ob, path, i) for i in range(n)]
    start_val = tuple(ob.rotation_euler) if kind == 'EULER' else tuple(ob.rotation_quaternion)
    axis = Vector((0.3, 1.0, 0.2)).normalized()
    for i in range(15):
        step = Quaternion(axis, math.radians(4))
        if kind == 'EULER':
            ob.rotation_euler = (step @ ob.rotation_euler.to_quaternion()).to_euler('XYZ', ob.rotation_euler)
        else:
            ob.rotation_quaternion = step @ Quaternion(ob.rotation_quaternion)
        update()
    moved = [keys(ob, path, i) for i in range(n)]
    if kind == 'EULER':
        ob.rotation_euler = start_val
    else:
        ob.rotation_quaternion = start_val
    update()
    back = [keys(ob, path, i) for i in range(n)]
    check(f'{kind} session: drag moved the keys', not approx(moved, start))
    check(f'{kind} session: cancel restores keys exactly', all(np.array_equal(a, b) for a, b in zip(back, start)), (back, start))

# ================================================================ frame based falloff
frames = [(10, 0.0), (20, 0.0), (22, 0.0), (25, 0.0), (30, 0.0), (40, 0.0), (50, 0.0)]
s, ob = loc_scene(frames, anim_offset_easing_window=10, anim_offset_easing_unit='FRAMES', anim_offset_easing_function='linear')
s.frame_set(30)
ob.location.x = 10.0; update()
check('frame falloff (linear, 10 frames)', approx(keys(ob, 'location', 0), [0, 0, 2, 5, 10, 0, 0]), keys(ob, 'location', 0))

s, ob = loc_scene(frames, anim_offset_easing_window=10, anim_offset_easing_unit='KEYS', anim_offset_easing_function='linear')
s.frame_set(30)
ob.location.x = 10.0; update()
check('key falloff differs from frame falloff', not approx(keys(ob, 'location', 0), [0, 0, 2, 5, 10, 0, 0]))

# the window in frames does not depend on how many keys are in there, also for a selected range
s, ob = loc_scene(frames, anim_offset_target='SELECTED', anim_offset_easing_window=6, anim_offset_easing_unit='FRAMES', anim_offset_easing_function='linear')
fc_test = fcurves(ob).find('location', index=0)
w = addon._get_easing_weights(fc_test, np.array([False, False, False, False, True, False, False]), 6, 'linear', 'in_out', 'FRAMES')
check('frame falloff weights', approx(w, [0, 0, 0, 1 - 5 / 6, 1, 0, 0]), w)

# ================================================================ custom properties
check('custom path detection', all([
    addon.is_custom_property_path('["foo"]'),
    addon.is_custom_property_path('pose.bones["a"]["foo bar"]'),
    addon.is_custom_property_path("modifiers['m']['x']"),
    not addon.is_custom_property_path('location'),
    not addon.is_custom_property_path('pose.bones["a"].location'),
    not addon.is_custom_property_path('key_blocks["Smile"].value'),
]))

def custom_scene(**kw):
    s = reset()
    for k, v in kw.items():
        setattr(s, k, v)
    ob = new_cube()
    s.use_anim_offset_mode = False
    ob['foo'] = 0.0
    for f, v in ((1, 0.0), (10, 5.0), (20, 10.0)):
        ob['foo'] = v
        ob.keyframe_insert('["foo"]', frame=f)
    s.use_anim_offset_mode = True
    return s, ob

s, ob = custom_scene(anim_offset_target='ALL')
s.frame_set(10)
addon.poll_custom_properties(s)                 # seeds the last seen value
check('custom property: nothing changes without an edit', not addon.poll_custom_properties(s) and approx(keys(ob, '["foo"]', 0), [0, 5, 10]))
ob['foo'] = 7.0                                 # no depsgraph update for this, only polling can see it
changed = addon.poll_custom_properties(s)
check('custom property: edit offsets all keys', changed and approx(keys(ob, '["foo"]', 0), [2, 7, 12]), keys(ob, '["foo"]', 0))
check('custom property: not applied twice', not addon.poll_custom_properties(s) and approx(keys(ob, '["foo"]', 0), [2, 7, 12]))
update()
check('custom property: no feedback after a depsgraph update', not addon.poll_custom_properties(s) and approx(keys(ob, '["foo"]', 0), [2, 7, 12]))

s.frame_set(20)                                  # animation overwrites the value, that is not an edit
addon.poll_custom_properties(s)
check('custom property: frame change is not an edit', not addon.poll_custom_properties(s) and approx(keys(ob, '["foo"]', 0), [2, 7, 12]))

# current frame mode with insertion, and cancel
s, ob = custom_scene(anim_offset_target='CURRENT')
s.frame_set(5)
addon.poll_custom_properties(s)
start_value = ob['foo']
ob['foo'] = start_value + 1.0
addon.poll_custom_properties(s)
check('custom property: key inserted on the current frame', frames_of(ob, '["foo"]') == [1, 5, 10, 20] and approx(keys(ob, '["foo"]', 0)[1], start_value + 1.0), (frames_of(ob, '["foo"]'), keys(ob, '["foo"]', 0)))
ob['foo'] = start_value
addon.poll_custom_properties(s)
check('custom property: cancel removes the inserted key', frames_of(ob, '["foo"]') == [1, 10, 20])

# polling a slider drag in big steps still gives the exact result
s, ob = custom_scene(anim_offset_target='ALL')
s.frame_set(10)
addon.poll_custom_properties(s)
for v in (5.4, 6.9, 5.2, 8.0):
    ob['foo'] = v
    addon.poll_custom_properties(s)
check('custom property: result only depends on the final value', approx(keys(ob, '["foo"]', 0), [3, 8, 13]), keys(ob, '["foo"]', 0))

# locked custom properties, mode off
s, ob = custom_scene(anim_offset_target='ALL')
s.frame_set(10)
addon.poll_custom_properties(s)
fcurves(ob).find('["foo"]', index=0).lock = True
ob['foo'] = 9.0
addon.poll_custom_properties(s)
check('custom property: locked channel untouched', approx(keys(ob, '["foo"]', 0), [0, 5, 10]))
fcurves(ob).find('["foo"]', index=0).lock = False
s.use_anim_offset_mode = False
ob['foo'] = 3.0
check('custom property: nothing when the mode is off', not addon.poll_custom_properties(s))

# pose bone custom property
s = reset()
arm_d = bpy.data.armatures.new('Arm2')
arm = bpy.data.objects.new('Arm2', arm_d)
s.collection.objects.link(arm)
bpy.context.view_layer.objects.active = arm
bpy.ops.object.mode_set(mode='EDIT')
eb = arm_d.edit_bones.new('B'); eb.head = (0, 0, 0); eb.tail = (0, 0, 1)
bpy.ops.object.mode_set(mode='POSE')
pb = arm.pose.bones['B']
s.use_anim_offset_mode = False
for f, v in ((1, 0.0), (10, 1.0)):
    pb['ik_fk'] = v
    pb.keyframe_insert('["ik_fk"]', frame=f)
s.use_anim_offset_mode = True
s.anim_offset_target = 'ALL'
s.frame_set(10)
addon.poll_custom_properties(s)
pb['ik_fk'] = 0.75
addon.poll_custom_properties(s)
check('pose bone custom property offset', approx(keys(arm, 'pose.bones["B"]["ik_fk"]', 0), [-0.25, 0.75]), keys(arm, 'pose.bones["B"]["ik_fk"]', 0))
bpy.ops.object.mode_set(mode='OBJECT')

# ================================================================ indicator / overlays
check('draw handlers registered', len(addon._draw_handlers) == 3, addon._draw_handlers)

class _Ctx:  # only what _get_overlay_ranges reads
    def __init__(self, ob): self.active_object = ob

s, ob = loc_scene([(1, 0.0), (6, 0.0), (8, 0.0), (11, 0.0), (15, 0.0), (20, 0.0)], anim_offset_easing_window=2)
s.frame_set(8)
r = addon._get_overlay_ranges(_Ctx(ob), addon._get_settings(s))
check('overlay range (current key, window in keys)', r == (8, 8, 1, 15), r)
s.anim_offset_easing_unit = 'FRAMES'
r = addon._get_overlay_ranges(_Ctx(ob), addon._get_settings(s))
check('overlay range (window in frames)', r == (8, 8, 6, 10), r)
s.frame_set(9)
r = addon._get_overlay_ranges(_Ctx(ob), addon._get_settings(s))
check('overlay range (no key on the frame, it will be inserted)', r == (9, 9, 7, 11), r)
s.anim_offset_insert_missing = False
check('overlay: nothing when there is nothing to offset', addon._get_overlay_ranges(_Ctx(ob), addon._get_settings(s)) is None)
s.anim_offset_insert_missing = True
s.anim_offset_easing_window = 0
r = addon._get_overlay_ranges(_Ctx(ob), addon._get_settings(s))
check('overlay range (no window)', r == (9, 9, 9, 9), r)
c1 = addon._get_overlay_ranges_cached(_Ctx(ob), addon._get_settings(s))
c2 = addon._get_overlay_ranges_cached(_Ctx(ob), addon._get_settings(s))
check('overlay cache returns the same range', c1 == c2 == (9, 9, 9, 9))

# ================================================================ side of the current frame
s, ob = loc_scene([(1, 0.0), (10, 0.0), (20, 0.0)], anim_offset_target='ALL', anim_offset_side='AFTER')
s.frame_set(10)
ob.location.x = 3.0; update()
check('side forward (all keys): only keys from the current frame on', approx(keys(ob, 'location', 0), [0, 3, 3]), keys(ob, 'location', 0))

s, ob = loc_scene([(1, 0.0), (10, 0.0), (20, 0.0)], anim_offset_target='ALL', anim_offset_side='BEFORE')
s.frame_set(10)
ob.location.x = 3.0; update()
check('side backward (all keys): only keys up to the current frame', approx(keys(ob, 'location', 0), [3, 3, 0]), keys(ob, 'location', 0))

s, ob = loc_scene([(1, 0.0), (10, 0.0), (20, 0.0)], anim_offset_target='ALL', anim_offset_side='BOTH')
s.frame_set(10)
ob.location.x = 3.0; update()
check('side both (all keys): everything', approx(keys(ob, 'location', 0), [3, 3, 3]))

prop_frames = [(10, 0.0), (20, 0.0), (22, 0.0), (25, 0.0), (30, 0.0), (40, 0.0), (50, 0.0)]
opts = dict(anim_offset_easing_window=25, anim_offset_easing_unit='FRAMES', anim_offset_easing_function='linear')
s, ob = loc_scene(prop_frames, anim_offset_side='AFTER', **opts)
s.frame_set(30)
ob.location.x = 10.0; update()
check('current frame + falloff, forward only', approx(keys(ob, 'location', 0), [0, 0, 0, 0, 10, 6, 2]), keys(ob, 'location', 0))

s, ob = loc_scene(prop_frames, anim_offset_side='BEFORE', **opts)
s.frame_set(30)
ob.location.x = 10.0; update()
check('current frame + falloff, backward only', approx(keys(ob, 'location', 0), [2, 6, 6.8, 8, 10, 0, 0]), keys(ob, 'location', 0))

s, ob = loc_scene(prop_frames, anim_offset_side='BOTH', **opts)
s.frame_set(30)
ob.location.x = 10.0; update()
check('current frame + falloff, both sides', approx(keys(ob, 'location', 0), [2, 6, 6.8, 8, 10, 6, 2]), keys(ob, 'location', 0))

# selected keys are cut at the current frame
s, ob = loc_scene(prop_frames, anim_offset_target='SELECTED', anim_offset_side='AFTER', anim_offset_easing_window=0)
fc_sel = fcurves(ob).find('location', index=0)
for i, kp in enumerate(fc_sel.keyframe_points):
    kp.select_control_point = kp.select_left_handle = kp.select_right_handle = i in (1, 2, 3)
s.frame_set(22)
ob.location.x = 4.0; update()
check('selected keys + forward: keys before the current frame are left alone', approx(keys(ob, 'location', 0), [0, 0, 4, 4, 0, 0, 0]), keys(ob, 'location', 0))

# an inserted key sits on the current frame, so it is always affected
s, ob = loc_scene([(1, 0.0), (10, 9.0), (20, 18.0)], anim_offset_side='AFTER', **dict(anim_offset_easing_window=10, anim_offset_easing_unit='FRAMES', anim_offset_easing_function='linear'))
s.frame_set(5)
start_value = ob.location.x
ob.location.x = start_value + 1.0; update()
k = keys(ob, 'location', 0)
check('forward: inserted key gets the offset, the key before is untouched', frames_of(ob) == [1, 5, 10, 20] and approx(k[0], 0.0) and approx(k[1], start_value + 1.0) and approx(k[2], 9.5) and approx(k[3], 18.0), k)

# rotations
s = reset()
s.anim_offset_target = 'ALL'
s.anim_offset_side = 'AFTER'
ob = new_cube()
orig = [Euler((0.2, 0.5, -0.3)), Euler((0.9, -0.4, 0.7)), Euler((-0.5, 0.3, 1.1))]
for f, e in zip((1, 10, 20), orig):
    ob.rotation_euler = e
    ob.keyframe_insert('rotation_euler', frame=f)
s.frame_set(10)
twist = Quaternion((0.3, 0.8, -0.2), math.radians(35))
new = twist @ ob.rotation_euler.to_quaternion()
ob.rotation_euler = new.to_euler('XYZ', ob.rotation_euler)
update()
kq = key_q(ob, 'EULER')
oq = [e.to_quaternion() for e in orig]
check('rotation: forward only', same_rot(kq[0], oq[0]) and same_rot(kq[1], twist @ oq[1]) and same_rot(kq[2], twist @ oq[2]))

# ================================================================ proportional falloff (time)
s, ob = loc_scene(prop_frames, anim_offset_easing_window=25, anim_offset_easing_unit='FRAMES', anim_offset_easing_function='linear', anim_offset_proportional=False)
s.frame_set(30)
ob.location.x = 10.0; update()
check('proportional off: radius is ignored', approx(keys(ob, 'location', 0), [0, 0, 0, 0, 10, 0, 0]), keys(ob, 'location', 0))
s.anim_offset_proportional = True
ob.location.x = 14.0; update()
check('proportional on again: falloff applies', approx(keys(ob, 'location', 0), [0.8, 2.4, 2.72, 3.2, 14, 2.4, 0.8]), keys(ob, 'location', 0))

# settings written by older versions (a window, no proportional switch) keep their falloff
s = reset()
s.property_unset('anim_offset_proportional')
s.anim_offset_easing_window = 4
addon._migrate_legacy_settings()
check('legacy easing window turns proportional falloff on', s.anim_offset_proportional is True)
s = reset()
s.property_unset('anim_offset_proportional')
s.anim_offset_easing_window = 0
addon._migrate_legacy_settings()
check('no legacy easing window leaves it off', s.anim_offset_proportional is False)

# falloff shapes (same as the ones of proportional editing)
s, ob = loc_scene([(0, 0.0), (15, 0.0), (20, 0.0), (25, 0.0), (40, 0.0)])
fc_shape = fcurves(ob).find('location', index=0)
selected = np.array([False, False, True, False, False])
def shape(name):
    return addon._get_easing_weights(fc_shape, selected, 20, name, 'in_out', 'FRAMES')
t = 0.75 # 5 frames away from the radius of 20
check('falloff smooth', approx(shape('smooth')[1], 3 * t * t - 2 * t ** 3), shape('smooth'))
check('falloff root', approx(shape('root')[1], math.sqrt(t)), shape('root'))
check('falloff constant', approx(shape('constant'), [0, 1, 1, 1, 0]), shape('constant'))
check('falloff sharp (quadratic in)', approx(addon._get_easing_weights(fc_shape, selected, 20, 'quadratic', 'in', 'FRAMES')[1], t * t))
check('falloff sphere (circular out)', approx(addon._get_easing_weights(fc_shape, selected, 20, 'circular', 'out', 'FRAMES')[1], math.sqrt(2 * t - t * t)))

# overlay ranges know about the side
s, ob = loc_scene([(1, 0.0), (6, 0.0), (8, 0.0), (11, 0.0), (15, 0.0), (20, 0.0)], anim_offset_easing_window=3, anim_offset_easing_unit='FRAMES', anim_offset_side='AFTER')
s.frame_set(8)
r = addon._get_overlay_ranges(_Ctx(ob), addon._get_settings(s))
check('overlay range forward', r == (8, 8, 8, 11), r)
s.anim_offset_side = 'BEFORE'
r = addon._get_overlay_ranges(_Ctx(ob), addon._get_settings(s))
check('overlay range backward', r == (8, 8, 5, 8), r)

# ---------------------------------------------------------------- robustness
s = reset()
ob = new_cube()
ob.animation_data_create()
act = bpy.data.actions.new('Empty')
slot = act.slots.new('OBJECT', 'X')
ob.animation_data.action = act
ob.animation_data.action_slot = slot
try:
    r = addon._get_obj_fcurves(ob)
    check('action without layers does not crash', r is None)
except Exception as e:
    check('action without layers does not crash', False, repr(e))

# ---------------------------------------------------------------- editing keys in the graph editor
# Moving keyframes in the Graph Editor changes the animated property too. That is the animation
# driving the value, not a user edit, so the add-on must not offset (or re-sort/insert) keys then.
s = reset()
ob = new_cube()
for f, x in ((1, 0.0), (10, 5.0), (20, 10.0)):
    s.frame_set(f); ob.location.x = x; ob.keyframe_insert('location', index=0, frame=f)
s.anim_offset_target = 'CURRENT'
s.anim_offset_insert_missing = True
s.frame_set(10)
update()
fc = fcurves(ob).find('location', index=0)
fc.keyframe_points[1].co[1] = 8.0 # what dragging a key does
fc.keyframe_points[1].handle_left[1] += 3.0
fc.keyframe_points[1].handle_right[1] += 3.0
fc.update()
ob.location.x = 8.0 # headless Blender does not flush the animated value back, the real one does
update()
update()
check('moving a key in the graph editor is not offset again', approx(keys(ob, 'location', 0), [0, 8, 10]), keys(ob, 'location', 0))
check('moving a key in the graph editor inserts no keys', len(fc.keyframe_points) == 3, len(fc.keyframe_points))

# ... while a real property edit still offsets
ob.location.x = 9.0
update()
check('property edit still offsets after a graph editor edit', approx(keys(ob, 'location', 0), [0, 9, 10]), keys(ob, 'location', 0))

# ---------------------------------------------------------------- deleting animated objects
# The depsgraph_update_pre handler used to read depsgraph.updates, which before an update still
# describes the previous evaluation: after deleting just-edited objects (Outliner delete) it read
# freed IDs and crashed Blender. Deleting must be safe, and the next edit must still be offset.
s = reset()
victims = []
for i in range(60):
    o = new_cube(f'Victim{i}')
    for f, x in ((1, 0.0), (10, float(i))):
        s.frame_set(f); o.location.x = x; o.keyframe_insert('location', index=0, frame=f)
    if victims:
        c = o.constraints.new('DAMPED_TRACK'); c.target = victims[-1]
    victims.append(o)
keeper = new_cube('Keeper')
for f, x in ((1, 0.0), (10, 5.0), (20, 10.0)):
    s.frame_set(f); keeper.location.x = x; keeper.keyframe_insert('location', index=0, frame=f)
s.frame_set(10)
update()
for o in victims:
    o.location.x += 1.0     # edited: now in the last depsgraph update
update()
for o in victims:
    bpy.data.objects.remove(o, do_unlink=True)
bpy.data.orphans_purge(do_recursive=True)
for n in range(3):
    for k in range(200):
        bpy.data.meshes.new(f'junk{n}_{k}')    # reuse the freed memory
    update()
check('deleting just-edited animated objects does not crash', True)
keeper.location.x = 6.0
update()
check('editing after deleting objects still offsets', approx(keys(keeper, 'location', 0), [1, 6, 11]), keys(keeper, 'location', 0))

# Same, with a second view layer whose depsgraph is evaluated first: the handlers must read the
# depsgraph that was just evaluated, not the context's, which still lists the deleted objects.
s = reset()
second = s.view_layers.new('Second')
victims = []
for i in range(60):
    o = new_cube(f'Victim2_{i}')
    for f, x in ((1, 0.0), (10, float(i))):
        s.frame_set(f); o.location.x = x; o.keyframe_insert('location', index=0, frame=f)
    victims.append(o)
keeper = new_cube('Keeper2')
for f, x in ((1, 0.0), (10, 5.0), (20, 10.0)):
    s.frame_set(f); keeper.location.x = x; keeper.keyframe_insert('location', index=0, frame=f)
s.frame_set(10)
update(); second.depsgraph.update()
for o in victims:
    o.location.x += 1.0
update(); second.depsgraph.update()
for o in victims:
    bpy.data.objects.remove(o, do_unlink=True)
bpy.data.orphans_purge(do_recursive=True)
for k in range(300):
    bpy.data.meshes.new(f'junk_vl_{k}')
second.depsgraph.update()
update()
check('deleting objects with several view layers does not crash', True)
bpy.context.view_layer.objects.active = keeper
keeper.location.x = 6.0
update()
check('editing after deleting with several view layers still offsets', approx(keys(keeper, 'location', 0), [1, 6, 11]), keys(keeper, 'location', 0))

bad = sum(1 for _, ok in results if not ok)
print(f'\n{len(results) - bad}/{len(results)} passed')
sys.exit(1 if bad else 0)
