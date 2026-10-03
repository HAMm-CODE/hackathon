"""
Render the reconstructed racket swing in Blender from out/keyframes.json.

Tested with Blender 5.0.1. Run from the repo root:
  blender -b -P blender/render_racket.py                      # full mp4, front video angle
  blender -b -P blender/render_racket.py -- --preview         # 4 still frames to check framing (fast)
  blender -b -P blender/render_racket.py -- --view rear --engine CYCLES --samples 32
  blender -P blender/render_racket.py -- --no-render          # open the scene in the Blender UI instead

Options after "--":
  --keyframes PATH   default out/keyframes.json
  --out PATH         default out/racket_render.mp4
  --view front|rear  camera matching that video angle (default front)
  --engine EEVEE|CYCLES|WORKBENCH   default EEVEE (falls back automatically if unavailable)
  --samples N        render samples (default 16)
  --slowmo X         playback slow-motion factor (default 4)
  --preview          render frames at start, mid swing, contact, end as PNGs only
  --no-render        build the scene and stop (use without -b to inspect it)
"""
import argparse, json, math, os, sys
import bpy
from mathutils import Matrix, Quaternion, Vector

# ---------------------------------------------------------------- arguments and paths
argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--keyframes", default=None); ap.add_argument("--out", default=None)
ap.add_argument("--view", default="front", choices=["front", "rear"])
ap.add_argument("--engine", default="EEVEE", choices=["EEVEE", "CYCLES", "WORKBENCH"])
ap.add_argument("--samples", type=int, default=16); ap.add_argument("--slowmo", type=float, default=4.0)
ap.add_argument("--preview", action="store_true"); ap.add_argument("--no-render", action="store_true")
a = ap.parse_args(argv)
try:
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
except NameError:
    ROOT = os.getcwd()
if not os.path.exists(os.path.join(ROOT, "out", "keyframes.json")):
    ROOT = os.getcwd()
KF = a.keyframes or os.path.join(ROOT, "out", "keyframes.json")
OUT = a.out or os.path.join(ROOT, "out", "racket_render.mp4")
K = json.load(open(KF))
FR = K["frames"]; N = len(FR); FS = K["sample_rate_hz"]; IMP = K["impact_sample"]; B1 = K["shock_bridge"][1]
P = Vector(K["pivot_to_sensor_m"])

# keyframes use a y-up display frame (x right, y up, camera on +z). Blender is z-up:
# (x, y, z)_display -> (x, -z, y)_blender
T = Matrix(((1, 0, 0), (0, 0, -1), (0, 1, 0)))
def to_b(v): return T @ Vector(v)
def rot_b(q_xyzw):
    x, y, z, w = q_xyzw
    return (T @ Quaternion((w, x, y, z)).to_matrix() @ T.transposed()).to_quaternion()

# ---------------------------------------------------------------- scene
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.context.preferences.edit.keyframe_new_interpolation_type = "LINEAR"   # no easing between samples
sc = bpy.context.scene
fps = 30
frames_out = max(2, round(N / FS * a.slowmo * fps))        # video frames for the whole window
sc.render.fps = fps; sc.frame_start = 1; sc.frame_end = frames_out
sc.render.resolution_x, sc.render.resolution_y = 1280, 720
sample_to_frame = lambda i: 1 + i / (N - 1) * (frames_out - 1)

def set_engine(name):
    options = {"EEVEE": ["BLENDER_EEVEE", "BLENDER_EEVEE_NEXT"], "CYCLES": ["CYCLES"], "WORKBENCH": ["BLENDER_WORKBENCH"]}[name]
    for e in options:
        try:
            sc.render.engine = e; return e
        except TypeError:
            continue
    return sc.render.engine
eng = set_engine(a.engine)
if eng == "CYCLES":
    sc.cycles.samples = a.samples; sc.cycles.device = "CPU"
elif "EEVEE" in eng:
    try: sc.eevee.taa_render_samples = a.samples
    except AttributeError: pass

world = bpy.data.worlds.new("court"); sc.world = world; world.use_nodes = True
world.node_tree.nodes["Background"].inputs[0].default_value = (0.012, 0.075, 0.27, 1)   # court blue (linear)
world.node_tree.nodes["Background"].inputs[1].default_value = 1.0

def material(name, rgb, alpha=1.0, emission=0.0):
    m = bpy.data.materials.new(name); m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*rgb, 1); b.inputs["Roughness"].default_value = 0.5
    if alpha < 1:
        b.inputs["Alpha"].default_value = alpha
        for attr, val in (("blend_method", "BLEND"), ("surface_render_method", "BLENDED")):
            try: setattr(m, attr, val)
            except (AttributeError, TypeError): pass
    if emission:
        for key in ("Emission Color", "Emission"):
            if key in b.inputs: b.inputs[key].default_value = (*rgb, 1); break
        b.inputs["Emission Strength"].default_value = emission
    return m
M_FRAME = material("frame", (0.02, 0.025, 0.03))
M_GRIP = material("grip", (0.85, 0.87, 0.85))
M_STR = material("strings", (0.62, 0.75, 0.10), alpha=0.55)
M_DAMP = material("dampener", (0.85, 0.25, 0.04), emission=0.5)
M_TRAIL = material("trail", (0.65, 0.78, 0.12), emission=2.0)
M_GHOST = material("ghost", (0.9, 0.92, 0.9), alpha=0.18)

def racket(name, mat_frame, mat_grip, strings=True):
    """Racket built in the sensor frame (X toward butt, Y across face, Z out of face, metres)
    under an empty at the hand pivot; the body is offset by p so the pivot is the rotation centre."""
    pivot = bpy.data.objects.new(name, None); sc.collection.objects.link(pivot); pivot.rotation_mode = "QUATERNION"
    body = bpy.data.objects.new(name + "_body", None); sc.collection.objects.link(body)
    body.parent = pivot; body.location = P
    def own(o, m):
        o.data.materials.append(m); o.parent = body; return o
    bpy.ops.mesh.primitive_cylinder_add(radius=0.016, depth=0.20, vertices=24, location=(0.29, 0, 0), rotation=(0, math.pi / 2, 0))
    own(bpy.context.object, mat_grip)
    bpy.ops.mesh.primitive_cylinder_add(radius=0.012, depth=0.06, vertices=16, location=(0.16, 0, 0), rotation=(0, math.pi / 2, 0))
    own(bpy.context.object, mat_frame)
    for y in (0.07, -0.07):
        s, e = Vector((0.13, 0, 0)), Vector((0.002, y, 0)); d = e - s
        bpy.ops.mesh.primitive_cylinder_add(radius=0.009, depth=d.length, vertices=12, location=(s + e) / 2)
        o = bpy.context.object; o.rotation_mode = "QUATERNION"; o.rotation_quaternion = Vector((0, 0, 1)).rotation_difference(d)
        own(o, mat_frame)
    # head: an elliptical tube, built as a curve so the tube keeps a constant thickness
    cu = bpy.data.curves.new(name + "_head", "CURVE"); cu.dimensions = "3D"; cu.bevel_depth = 0.009; cu.bevel_resolution = 3
    sp = cu.splines.new("POLY"); n = 72; sp.points.add(n - 1); sp.use_cyclic_u = True
    for k in range(n):
        th = 2 * math.pi * k / n
        sp.points[k].co = (-0.133 + 0.163 * math.cos(th), 0.125 * math.sin(th), 0, 1)
    head = bpy.data.objects.new(name + "_head", cu); sc.collection.objects.link(head); own(head, mat_frame)
    if strings:
        bpy.ops.mesh.primitive_circle_add(vertices=64, radius=1.0, fill_type="NGON", location=(-0.133, 0, 0))
        bed = bpy.context.object; bed.scale = (0.157, 0.119, 1); own(bed, M_STR)
        bpy.ops.mesh.primitive_uv_sphere_add(radius=0.012, location=(0, 0, 0.006)); own(bpy.context.object, M_DAMP)
    return pivot

rk = racket("racket", M_FRAME, M_GRIP)
for i, f in enumerate(FR):
    rk.rotation_quaternion = rot_b(f["q_xyzw"]); rk.keyframe_insert("rotation_quaternion", frame=sample_to_frame(i))
ghost = racket("contact_pose", M_GHOST, M_GHOST, strings=False)
ghost.rotation_quaternion = rot_b(FR[IMP - 1]["q_xyzw"])

# frame turns orange after contact (lower confidence)
col = M_FRAME.node_tree.nodes["Principled BSDF"].inputs["Base Color"]
col.default_value = (0.02, 0.025, 0.03, 1); col.keyframe_insert("default_value", frame=sample_to_frame(IMP))
col.default_value = (0.80, 0.24, 0.04, 1); col.keyframe_insert("default_value", frame=sample_to_frame(IMP + 2))

# tip path that grows with the swing
cu = bpy.data.curves.new("tip_path", "CURVE"); cu.dimensions = "3D"; cu.bevel_depth = 0.0035; cu.use_fill_caps = True
sp = cu.splines.new("POLY"); sp.points.add(N - 1)
for i, f in enumerate(FR): sp.points[i].co = (*to_b(f["tip"]), 1)
trail = bpy.data.objects.new("tip_path", cu); sc.collection.objects.link(trail); trail.data.materials.append(M_TRAIL)
try: cu.bevel_factor_mapping_end = "RESOLUTION"  # grow by point index, i.e. by time
except (AttributeError, TypeError): pass
cu.bevel_factor_end = 0.0; cu.keyframe_insert("bevel_factor_end", frame=1)
cu.bevel_factor_end = 1.0; cu.keyframe_insert("bevel_factor_end", frame=frames_out)

# hand marker (rotation centre)
bpy.ops.mesh.primitive_uv_sphere_add(radius=0.03, location=(0, 0, 0)); bpy.context.object.data.materials.append(M_GRIP)

# light and camera at the chosen video angle
bpy.ops.object.light_add(type="SUN", rotation=(math.radians(40), 0, math.radians(30))); bpy.context.object.data.energy = 3.5
bpy.ops.object.light_add(type="AREA", location=(0, -1.5, 1.5)); bpy.context.object.data.energy = 60; bpy.context.object.data.size = 2
pts = [to_b(f["tip"]) for f in FR] + [Vector((0, 0, 0))]
lo = Vector([min(p[k] for p in pts) for k in range(3)]); hi = Vector([max(p[k] for p in pts) for k in range(3)])
center = (lo + hi) / 2; radius = max((p - center).length for p in pts) + 0.12
fwd = to_b(K["cameras_display"][a.view]["forward"]).normalized()
up = to_b(K["cameras_display"][a.view]["up"]).normalized()
cam_data = bpy.data.cameras.new("cam"); cam_data.lens = 50
cam = bpy.data.objects.new("camera", cam_data); sc.collection.objects.link(cam); sc.camera = cam
half_fov = cam_data.angle_y / 2 if hasattr(cam_data, "angle_y") else math.radians(14)
dist = 0.9 * radius / math.tan(half_fov)
cam.location = center - fwd * dist
right = fwd.cross(up).normalized(); up2 = right.cross(fwd).normalized()
cam.matrix_world = Matrix.Translation(cam.location) @ Matrix((right, up2, -fwd)).transposed().to_4x4()

# ---------------------------------------------------------------- render
sc.render.film_transparent = False
if a.no_render:
    print("scene built; frames 1 to", frames_out)
elif a.preview:
    sc.render.image_settings.file_format = "PNG"
    base = os.path.splitext(OUT)[0]
    for label, i in (("start", 0), ("mid", IMP // 2), ("contact", IMP - 1), ("end", N - 1)):
        sc.frame_set(round(sample_to_frame(i)))
        sc.render.filepath = f"{base}_preview_{label}.png"
        bpy.ops.render.render(write_still=True)
        print("wrote", sc.render.filepath)
else:
    try:
        sc.render.image_settings.media_type = "VIDEO"        # Blender 5.0+
    except (AttributeError, TypeError):
        pass
    sc.render.image_settings.file_format = "FFMPEG"
    sc.render.ffmpeg.format = "MPEG4"; sc.render.ffmpeg.codec = "H264"; sc.render.ffmpeg.constant_rate_factor = "HIGH"
    sc.render.filepath = OUT
    bpy.ops.render.render(animation=True)
    print("wrote", OUT, f"({frames_out} frames, {a.slowmo}x slow motion)")
