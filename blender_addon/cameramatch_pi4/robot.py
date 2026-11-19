"""Procedural hard-surface humanoid robot + planted-foot walk cycle (rigid parts, keyframed per frame).

Local conventions: robot faces +Y, X is its right, Z up. Limb objects have their origin at the proximal
joint and extend along local -Z.
make_materials(floor_z0) -> material dict; build_robot(coll, mats) -> dict of part objects
animate_walk(parts, path, floor_z, frames, fps, ...) keyframes every part's world transform.
"""
import math

import bmesh
import bpy
from mathutils import Matrix, Quaternion, Vector

# ----------------------------------------------------------------------------- dimensions (m)
THIGH, SHIN = 0.44, 0.44
HIP_X, HIP_Z = 0.105, -0.07          # hip joints relative to pelvis centre
ANKLE_H = 0.085                       # ankle above sole bottom
HEEL_Y, TOE_Y = -0.075, 0.185         # foot extents relative to ankle
SHOULDER = Vector((0.235, 0.0, 0.44))  # relative to torso origin (pelvis centre)
UPPER_ARM, FOREARM = 0.29, 0.26
NECK_TOP = 0.60


# ----------------------------------------------------------------------------- mesh helpers
class Builder:
    def __init__(self):
        self.bm = bmesh.new()

    def _tag(self, verts, mi):
        for f in {f for v in verts for f in v.link_faces}:
            f.material_index = mi

    def box(self, size, center, mi, top=(1.0, 1.0), bottom=(1.0, 1.0), rot=None):
        r = bmesh.ops.create_cube(self.bm, size=1.0)
        sx, sy, sz = size
        for v in r["verts"]:
            fx, fy = top if v.co.z > 0 else bottom
            v.co = Vector((v.co.x * sx * fx, v.co.y * sy * fy, v.co.z * sz))
            if rot is not None:
                v.co = rot @ v.co
            v.co += Vector(center)
        self._tag(r["verts"], mi)

    def cyl(self, r1, depth, center, mi, axis="Z", r2=None, seg=32, rot=None):
        r = bmesh.ops.create_cone(self.bm, cap_ends=True, cap_tris=False, segments=seg,
                                  radius1=r1, radius2=r1 if r2 is None else r2, depth=depth)
        R = {"Z": Matrix.Identity(3), "X": Matrix.Rotation(math.pi / 2, 3, "Y"),
             "Y": Matrix.Rotation(math.pi / 2, 3, "X")}[axis]
        if rot is not None:
            R = rot @ R
        for v in r["verts"]:
            v.co = R @ v.co + Vector(center)
        self._tag(r["verts"], mi)

    def sphere(self, rad, center, mi, seg=32):
        r = bmesh.ops.create_uvsphere(self.bm, u_segments=seg, v_segments=seg // 2, radius=rad)
        for v in r["verts"]:
            v.co += Vector(center)
        self._tag(r["verts"], mi)

    def to_object(self, name, mats, coll, bevel=0.008, segs=3):
        me = bpy.data.meshes.new(name)
        self.bm.to_mesh(me)
        self.bm.free()
        me.polygons.foreach_set("use_smooth", [True] * len(me.polygons))
        for m in mats:
            me.materials.append(m)
        ob = bpy.data.objects.new(name, me)
        coll.objects.link(ob)
        if bevel:
            b = ob.modifiers.new("Bevel", "BEVEL")
            b.width = bevel
            b.segments = segs
            b.limit_method = "ANGLE"
            b.angle_limit = math.radians(35)
            b.harden_normals = True
            b.use_clamp_overlap = True
        ob.rotation_mode = "QUATERNION"
        return ob


# material slots: 0 paint, 1 graphite, 2 rubber, 3 chrome, 4 visor, 5 led, 6 decal
def build_robot(coll, mats, side_sign_names=("L", "R")):
    M = [mats[k] for k in ("paint", "graphite", "rubber", "chrome", "visor", "led", "decal")]
    P, G, RB, CH, VI, LED, DC = range(7)
    parts = {}

    b = Builder()                                                    # pelvis
    b.box((0.30, 0.19, 0.12), (0, 0, -0.015), P, top=(1, 1), bottom=(0.82, 0.85))
    b.cyl(0.042, 0.30, (0, 0, HIP_Z), G, axis="X")
    for s in (-1, 1):
        b.sphere(0.056, (s * HIP_X, 0, HIP_Z), G)
    b.box((0.20, 0.012, 0.035), (0, 0.098, -0.01), G)
    parts["pelvis"] = b.to_object("Robot_pelvis", M, coll)

    b = Builder()                                                    # torso (pivot at pelvis centre)
    b.cyl(0.10, 0.17, (0, 0, 0.125), G, r2=0.11)
    for z in (0.085, 0.125, 0.165):
        b.cyl(0.118, 0.014, (0, 0, z), G)
    b.box((0.27, 0.18, 0.30), (0, 0, 0.34), G)                     # inner frame (shows in the panel seams)
    b.box((0.30, 0.205, 0.11), (0, 0, 0.245), P, top=(1.06, 1.0))  # lower chest panel
    b.box((0.32, 0.215, 0.18), (0, 0, 0.40), P, top=(1.25, 1.05))  # upper chest panel
    b.box((0.13, 0.01, 0.06), (0, 0.105, 0.245), G)
    b.cyl(0.008, 0.006, (0.085, 0.113, 0.43), LED, axis="Y", seg=24)
    b.box((0.28, 0.11, 0.25), (0, -0.155, 0.33), P)
    for z in (0.27, 0.30, 0.33, 0.36):
        b.box((0.19, 0.012, 0.012), (0, -0.212, z), G)
    b.cyl(0.072, 0.04, (0, 0, 0.505), G)
    b.cyl(0.045, 0.09, (0, 0, 0.555), G)
    for s in (-1, 1):
        b.sphere(0.06, (s * SHOULDER.x, SHOULDER.y, SHOULDER.z), G)
    parts["torso"] = b.to_object("Robot_torso", M, coll, bevel=0.01)

    b = Builder()                                                    # head shell (very rounded)
    b.box((0.20, 0.235, 0.215), (0, -0.005, 0.12), P, top=(0.92, 0.9))
    parts["head"] = b.to_object("Robot_head", M, coll, bevel=0.05, segs=6)
    b = Builder()                                                    # head details
    b.box((0.172, 0.03, 0.075), (0, 0.108, 0.13), VI)
    for s_ in (-1, 1):                                               # two small sensor lenses behind the visor
        b.cyl(0.011, 0.004, (s_ * 0.035, 0.1235, 0.135), LED, axis="Y", seg=20)
    for s in (-1, 1):
        b.cyl(0.045, 0.03, (s * 0.1, -0.01, 0.115), G, axis="X")
    b.box((0.11, 0.07, 0.035), (0, 0.02, 0.005), G)
    parts["head_det"] = b.to_object("Robot_head_details", M, coll, bevel=0.01, segs=4)

    for side, s in zip(side_sign_names, (-1, 1)):                    # s = -1 left (-X), +1 right
        b = Builder()                                                # thigh
        b.cyl(0.045, THIGH, (0, 0, -THIGH / 2), G)
        b.cyl(0.062, 0.10, (0, 0, 0), G, axis="X")
        b.box((0.15, 0.165, 0.17), (0, 0.012, -0.12), P, bottom=(0.92, 0.9))
        b.box((0.135, 0.15, 0.12), (0, 0.012, -0.27), P, bottom=(0.86, 0.84))
        b.cyl(0.064, 0.13, (0, 0, -THIGH), G, axis="X")
        b.box((0.03, 0.04, 0.04), (0, -0.075, -0.22), G)
        parts[f"thigh_{side}"] = b.to_object(f"Robot_thigh_{side}", M, coll, bevel=0.01)

        b = Builder()                                                # shin
        b.box((0.10, 0.05, 0.095), (0, 0.066, -0.025), P)
        b.cyl(0.048, SHIN, (0, 0, -SHIN / 2), G)
        b.box((0.125, 0.135, 0.13), (0, 0.016, -0.135), P, bottom=(0.94, 0.92))
        b.box((0.11, 0.122, 0.13), (0, 0.016, -0.275), P, bottom=(0.84, 0.8))
        b.sphere(0.046, (0, 0, -SHIN), G)
        b.box((0.03, 0.04, 0.04), (0, -0.06, -0.30), G)
        parts[f"shin_{side}"] = b.to_object(f"Robot_shin_{side}", M, coll, bevel=0.01)

        b = Builder()                                                # foot
        L = TOE_Y - HEEL_Y
        cy = (TOE_Y + HEEL_Y) / 2
        b.box((0.108, L - 0.01, 0.05), (0, cy, -0.047), P, top=(0.88, 0.8))
        b.box((0.116, L, 0.022), (0, cy, -ANKLE_H + 0.011), RB)
        b.box((0.08, 0.08, 0.05), (0, 0, -0.01), G)
        parts[f"foot_{side}"] = b.to_object(f"Robot_foot_{side}", M, coll, bevel=0.008)

        b = Builder()                                                # upper arm
        b.box((0.135, 0.155, 0.10), (s * 0.02, 0, 0.015), P, top=(0.85, 0.85))
        b.cyl(0.035, UPPER_ARM, (0, 0, -UPPER_ARM / 2), G)
        b.box((0.10, 0.11, 0.17), (0, 0, -0.155), P, bottom=(0.85, 0.85))
        b.cyl(0.052, 0.10, (0, 0, -UPPER_ARM), G, axis="X")
        parts[f"uarm_{side}"] = b.to_object(f"Robot_upperarm_{side}", M, coll, bevel=0.01)

        b = Builder()                                                # forearm
        b.cyl(0.03, FOREARM, (0, 0, -FOREARM / 2), G)
        b.box((0.096, 0.106, 0.18), (0, 0.004, -0.12), P, bottom=(0.8, 0.82))
        b.cyl(0.036, 0.03, (0, 0, -FOREARM + 0.005), G)
        parts[f"farm_{side}"] = b.to_object(f"Robot_forearm_{side}", M, coll, bevel=0.008)

        b = Builder()                                                # hand: palm faces the body (-s X)
        b.box((0.032, 0.078, 0.085), (0, 0, -0.055), G)
        b.box((0.012, 0.07, 0.06), (s * 0.019, 0, -0.05), P)
        for i, fy in enumerate((-0.027, -0.009, 0.009, 0.027)):
            z0 = -0.098
            r1 = Matrix.Rotation(-s * math.radians(25), 3, "Y")
            b.box((0.02, 0.016, 0.042), (-s * 0.006, fy, z0 - 0.018), G, rot=r1)
            r2 = Matrix.Rotation(-s * math.radians(70), 3, "Y")
            b.box((0.018, 0.015, 0.034), (-s * 0.028, fy, z0 - 0.045), G, rot=r2)
        rt = Matrix.Rotation(math.radians(-40), 3, "X")
        b.box((0.02, 0.018, 0.05), (-s * 0.012, 0.045, -0.04), G, rot=rt)
        parts[f"hand_{side}"] = b.to_object(f"Robot_hand_{side}", M, coll, bevel=0.005)

        b = Builder()                                                # calf piston: housing + rod
        b.cyl(0.017, 0.17, (0, 0, -0.085), G)
        parts[f"pist_h_{side}"] = b.to_object(f"Robot_piston_housing_{side}", M, coll, bevel=0.003)
        b = Builder()
        b.cyl(0.0095, 0.17, (0, 0, -0.085), CH)
        parts[f"pist_r_{side}"] = b.to_object(f"Robot_piston_rod_{side}", M, coll, bevel=0.002)

    # chest decal (painted text)
    cu = bpy.data.curves.new("Robot_decal", "FONT")
    cu.body = "RX-07"
    cu.size = 0.032
    cu.extrude = 0.0006
    cu.align_x = "CENTER"
    dec = bpy.data.objects.new("Robot_decal", cu)
    coll.objects.link(dec)
    cu.materials.append(mats["decal"])
    dec.rotation_mode = "QUATERNION"
    parts["decal"] = dec
    return parts


# ----------------------------------------------------------------------------- kinematics
def _frame(fwd, up=Vector((0, 0, 1))):
    """Rotation with +Y = fwd (projected horizontal), +Z = up."""
    y = (fwd - up * fwd.dot(up)).normalized()
    x = y.cross(up).normalized()
    z = x.cross(y)
    return Matrix((x, y, z)).transposed()


def _limb_rot(direction, pole):
    """Rotation whose local -Z points along `direction` and +Y towards `pole`."""
    z = -direction.normalized()
    y = (pole - z * pole.dot(z)).normalized()
    x = y.cross(z)
    return Matrix((x, y, z)).transposed()


def _smooth(u):
    u = min(max(u, 0.0), 1.0)
    return u * u * (3 - 2 * u)


def _M(R, t):
    m = R.to_4x4()
    m.translation = t
    return m


def animate_walk(parts, path, floor_z, f0, f1, fps, v=0.85, step=0.55, stance=0.56, pelvis_h=0.94,
                 phase0=0.1, look_target=None, look_from=None, look_ramp=40, foot_dx=0.10):
    """path(s) -> (Vector xy point, Vector xy unit tangent); floor_z(x, y) -> z. Walk along +s."""
    Ts = step / v
    cyc = 2 * Ts
    th_hs, th_to = math.radians(14), math.radians(-28)
    stats = {"clamped": 0, "max_ratio": 0.0}
    prevq = {}

    def plant(s, side_sign):
        p, t = path(s)
        n = Vector((t.y, -t.x, 0))                                  # right-hand normal of the walk direction
        q = Vector((p.x, p.y, 0)) + n * (side_sign * foot_dx)
        q.z = floor_z(q.x, q.y) + ANKLE_H
        return q, _frame(Vector((t.x, t.y, 0)))

    def foot_pose(anchor, R, pitch, pivot_local):
        """Foot flat at anchor (ankle point) rotated by pitch about a sole pivot (heel or toe)."""
        piv = anchor + R @ pivot_local
        Rp = R @ Matrix.Rotation(pitch, 3, "X")
        ank = piv + Rp @ (-pivot_local)
        return ank, Rp

    heel = Vector((0, HEEL_Y, -ANKLE_H))
    toe = Vector((0, TOE_Y, -ANKLE_H))

    def foot_at(t, o, side_sign):
        psi = t / cyc + phase0 - o
        k = math.floor(psi)
        u = psi - k
        s_mid = lambda kk: v * ((kk + stance / 2 + o - phase0) * cyc)   # noqa: E731
        if u < stance:
            A, R = plant(s_mid(k), side_sign)
            us = u / stance
            if us < 0.18:
                return foot_pose(A, R, th_hs * (1 - _smooth(us / 0.18)), heel), 0.0
            if us > 0.62:
                return foot_pose(A, R, th_to * _smooth((us - 0.62) / 0.38), toe), 0.0
            return (A, R), 0.0
        us = (u - stance) / (1 - stance)
        A0, R0 = plant(s_mid(k), side_sign)
        A1, R1 = plant(s_mid(k + 1), side_sign)
        a_start, r_start = foot_pose(A0, R0, th_to, toe)
        a_end, r_end = foot_pose(A1, R1, th_hs, heel)
        w = _smooth(us)
        a = a_start.lerp(a_end, w)
        a.z += 0.075 * math.sin(math.pi * min(us * 1.15, 1.0))
        pitch = th_to + (th_hs - th_to) * _smooth(us * 1.2)
        Rb = r_start.to_quaternion().slerp(r_end.to_quaternion(), w).to_matrix()
        # rebuild with interpolated pitch around the ankle (heading from slerp, pitch explicit)
        head = Rb @ Vector((0, 1, 0))
        Rf = _frame(Vector((head.x, head.y, 0))) @ Matrix.Rotation(pitch, 3, "X")
        return (a, Rf), us

    def key(name, M, fr):
        ob = parts[name]
        q = M.to_quaternion()
        if name in prevq and prevq[name].dot(q) < 0:
            q.negate()
        prevq[name] = q
        ob.location = M.translation
        ob.rotation_quaternion = q
        ob.keyframe_insert("location", frame=fr)
        ob.keyframe_insert("rotation_quaternion", frame=fr)

    def ik(H, A, l1, l2, pole):
        d_vec = A - H
        d = d_vec.length
        ratio = d / (l1 + l2)
        stats["max_ratio"] = max(stats["max_ratio"], ratio)
        if d > (l1 + l2) * 0.999:
            stats["clamped"] += 1
            d = (l1 + l2) * 0.999
        u = d_vec.normalized()
        w = (pole - u * pole.dot(u)).normalized()
        a = math.acos(max(-1, min(1, (l1 * l1 + d * d - l2 * l2) / (2 * l1 * d))))
        K = H + l1 * (math.cos(a) * u + math.sin(a) * w)
        Aeff = H + u * d
        return K, Aeff

    look_w = 0.0
    for fr in range(f0, f1 + 1):
        t = (fr - f0) / fps
        phi = t / cyc + phase0
        s = v * t
        p, tan = path(s)
        tan3 = Vector((tan.x, tan.y, 0)).normalized()
        nrm = Vector((tan3.y, -tan3.x, 0))
        bob = 0.012 * math.cos(4 * math.pi * (phi - stance / 2))
        sway = 0.022 * math.cos(2 * math.pi * (phi - stance / 2))  # +: toward left foot (-X)
        base = Vector((p.x, p.y, 0)) - nrm * sway
        base.z = floor_z(p.x, p.y) + pelvis_h + bob
        yaw = -math.radians(5) * math.cos(2 * math.pi * phi)
        Rh = _frame(tan3)
        Rp = Rh @ Matrix.Rotation(yaw, 3, "Z") @ Matrix.Rotation(math.radians(-3), 3, "X")
        Mp = _M(Rp, base)
        key("pelvis", Mp, fr)
        Rt = Rh @ Matrix.Rotation(-0.8 * yaw, 3, "Z") @ Matrix.Rotation(math.radians(-5), 3, "X") @ \
            Matrix.Rotation(math.radians(1.5) * math.cos(2 * math.pi * (phi - stance / 2)), 3, "Y")
        Mt = _M(Rt, base)
        key("torso", Mt, fr)
        # head: stabilised on the walk heading, then turns to the camera
        Rhead = Rh @ Matrix.Rotation(math.radians(-2), 3, "X")
        if look_target is not None and look_from is not None and fr >= look_from:
            look_w = _smooth((fr - look_from) / look_ramp)
        neck = Mt @ Vector((0, 0, NECK_TOP))
        if look_w > 0:
            tgt = look_target(fr) - neck
            Rl = _frame(Vector((tgt.x, tgt.y, 0)))
            pitch = math.atan2(tgt.z, Vector((tgt.x, tgt.y)).length)
            Rl = Rl @ Matrix.Rotation(pitch * 0.7, 3, "X")
            Rhead = Rhead.to_quaternion().slerp(Rl.to_quaternion(), 0.85 * look_w).to_matrix()
        Mh = _M(Rhead, neck)
        key("head", Mh, fr)
        key("head_det", Mh, fr)
        dec = Mt @ Vector((-0.07, 0.1104, 0.385))
        key("decal", _M(Rt @ Matrix(((-1, 0, 0), (0, 0, 1), (0, 1, 0))), dec), fr)

        for side, o, sg in (("L", 0.0, -1), ("R", 0.5, 1)):
            (A, Rf), us = foot_at(t, o, sg)
            H = Mp @ Vector((sg * HIP_X, 0, HIP_Z))
            pole = Rp @ Vector((0, 1, 0))
            K, Aeff = ik(H, A, THIGH, SHIN, pole)
            key(f"thigh_{side}", _M(_limb_rot(K - H, pole), H), fr)
            Rs = _limb_rot(Aeff - K, pole)
            key(f"shin_{side}", _M(Rs, K), fr)
            key(f"foot_{side}", _M(Rf, Aeff), fr)
            Rth = _limb_rot(K - H, pole)
            p1 = H + Rth @ Vector((0, -0.075, -0.22))
            p2 = K + Rs @ Vector((0, -0.06, -0.30))
            dvec = p2 - p1
            key(f"pist_h_{side}", _M(_limb_rot(dvec, pole), p1), fr)
            key(f"pist_r_{side}", _M(_limb_rot(-dvec, pole), p2), fr)
            # arms: swing opposite to the same-side leg
            sw = math.radians(17) * math.cos(2 * math.pi * (phi - o)) * (-1)
            S = Mt @ Vector((sg * SHOULDER.x, SHOULDER.y, SHOULDER.z))
            Ru = Rt @ Matrix.Rotation(sw, 3, "X") @ Matrix.Rotation(-sg * math.radians(7), 3, "Y")
            key(f"uarm_{side}", _M(Ru, S), fr)
            E = S + Ru @ Vector((0, 0, -UPPER_ARM))
            flex = math.radians(24) + max(0.0, sw) * 0.9
            Rf_ = Ru @ Matrix.Rotation(flex, 3, "X")
            key(f"farm_{side}", _M(Rf_, E), fr)
            Wr = E + Rf_ @ Vector((0, 0, -FOREARM))
            key(f"hand_{side}", _M(Rf_ @ Matrix.Rotation(math.radians(8), 3, "X"), Wr), fr)
    return stats


# ----------------------------------------------------------------------------- materials
FLOOR0 = 0.0   # floor height (world z) for the low-dirt gradient, set by make_materials


def principled(name):
    m = bpy.data.materials.new(name)
    if m.node_tree is None:
        m.use_nodes = True
    return m, m.node_tree, m.node_tree.nodes.get("Principled BSDF")




def paint_material():
    m, nt, b = principled("Robot_paint")
    N, L = nt.nodes, nt.links

    def node(t, **kw):
        n = N.new(t)
        for k, v in kw.items():
            setattr(n, k, v)
        return n

    def math_(op, a, bb=None, clamp=False):
        n = node("ShaderNodeMath", operation=op, use_clamp=clamp)
        for i, v in enumerate((a, bb)):
            if v is None:
                continue
            if isinstance(v, (int, float)):
                n.inputs[i].default_value = v
            else:
                L.new(v, n.inputs[i])
        return n.outputs[0]

    def mrange(v, a, bb, c=0.0, d=1.0):
        n = node("ShaderNodeMapRange")
        L.new(v, n.inputs["Value"])
        n.inputs["From Min"].default_value, n.inputs["From Max"].default_value = a, bb
        n.inputs["To Min"].default_value, n.inputs["To Max"].default_value = c, d
        return n.outputs["Result"]

    tc = node("ShaderNodeTexCoord")
    geo = node("ShaderNodeNewGeometry")
    ao = node("ShaderNodeAmbientOcclusion", samples=8, only_local=True)
    ao.inputs["Distance"].default_value = 0.05
    crev = math_("SUBTRACT", 1.0, ao.outputs["AO"])
    n1 = node("ShaderNodeTexNoise")
    n1.inputs["Scale"].default_value = 7.0
    n1.inputs["Detail"].default_value = 8.0
    L.new(tc.outputs["Object"], n1.inputs["Vector"])
    n2 = node("ShaderNodeTexNoise")
    n2.inputs["Scale"].default_value = 38.0
    n2.inputs["Detail"].default_value = 6.0
    L.new(tc.outputs["Object"], n2.inputs["Vector"])
    smudge = mrange(n1.outputs["Fac"], 0.5, 0.78)
    sepw = node("ShaderNodeSeparateXYZ")
    L.new(geo.outputs["Position"], sepw.inputs[0])
    low = mrange(sepw.outputs["Z"], FLOOR0 + 0.05, FLOOR0 + 0.5, 0.75, 0.0)
    grime = math_("MINIMUM", math_("ADD", math_("ADD", math_("MULTIPLY", crev, 1.3), math_("MULTIPLY", smudge, 0.3)),
                                   math_("MULTIPLY", low, mrange(n2.outputs["Fac"], 0.35, 0.65))), 1.0)
    # edge wear: bevel-normal vs true normal
    bev = node("ShaderNodeBevel", samples=8)
    bev.inputs["Radius"].default_value = 0.006
    dot = node("ShaderNodeVectorMath", operation="DOT_PRODUCT")
    L.new(bev.outputs["Normal"], dot.inputs[0])
    L.new(geo.outputs["True Normal"], dot.inputs[1])
    edge = mrange(dot.outputs["Value"], 0.995, 0.93)
    wear = math_("MULTIPLY", edge, mrange(n2.outputs["Fac"], 0.5, 0.62), clamp=True)
    base = node("ShaderNodeMix", data_type="RGBA")
    L.new(math_("MULTIPLY", grime, 0.85), base.inputs["Factor"])
    base.inputs["A"].default_value = (0.50, 0.49, 0.46, 1)
    base.inputs["B"].default_value = (0.075, 0.062, 0.05, 1)
    base2 = node("ShaderNodeMix", data_type="RGBA")
    L.new(wear, base2.inputs["Factor"])
    L.new(base.outputs["Result"], base2.inputs["A"])
    base2.inputs["B"].default_value = (0.58, 0.58, 0.6, 1)
    L.new(base2.outputs["Result"], b.inputs["Base Color"])
    L.new(wear, b.inputs["Metallic"])
    rough = math_("ADD", math_("ADD", 0.3, math_("MULTIPLY", n1.outputs["Fac"], 0.14)), math_("MULTIPLY", grime, 0.3))
    L.new(math_("MINIMUM", rough, 0.85), b.inputs["Roughness"])
    L.new(math_("MULTIPLY", math_("SUBTRACT", 1.0, grime, clamp=True), 0.35), b.inputs["Coat Weight"])
    b.inputs["Coat Roughness"].default_value = 0.18
    bump = node("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.025
    n3 = node("ShaderNodeTexNoise")
    n3.inputs["Scale"].default_value = 260.0
    L.new(tc.outputs["Object"], n3.inputs["Vector"])
    L.new(n3.outputs["Fac"], bump.inputs["Height"])
    L.new(bump.outputs["Normal"], b.inputs["Normal"])
    return m


def simple(name, base, metal, rough, coat=0.0, emit=None, rough_noise=0.0):
    m, nt, b = principled(name)
    b.inputs["Base Color"].default_value = (*base, 1)
    b.inputs["Metallic"].default_value = metal
    b.inputs["Roughness"].default_value = rough
    b.inputs["Coat Weight"].default_value = coat
    b.inputs["Coat Roughness"].default_value = 0.03
    if emit is not None:
        b.inputs["Emission Color"].default_value = (*emit[0], 1)
        b.inputs["Emission Strength"].default_value = emit[1]
    if rough_noise:
        n = nt.nodes.new("ShaderNodeTexNoise")
        n.inputs["Scale"].default_value = 25.0
        mr = nt.nodes.new("ShaderNodeMapRange")
        mr.inputs["To Min"].default_value = rough - rough_noise
        mr.inputs["To Max"].default_value = rough + rough_noise
        nt.links.new(n.outputs["Fac"], mr.inputs["Value"])
        nt.links.new(mr.outputs["Result"], b.inputs["Roughness"])
    return m


def make_materials(floor_z0):
    global FLOOR0
    FLOOR0 = floor_z0
    return {"paint": paint_material(),
            "graphite": simple("Robot_graphite", (0.028, 0.029, 0.033), 0.85, 0.42, rough_noise=0.08),
            "rubber": simple("Robot_rubber", (0.02, 0.02, 0.02), 0.0, 0.85),
            "chrome": simple("Robot_chrome", (0.85, 0.85, 0.86), 1.0, 0.08),
            "visor": simple("Robot_visor", (0.004, 0.004, 0.005), 0.0, 0.04, coat=1.0),
            "led": simple("Robot_led", (0.0, 0.0, 0.0), 0.0, 0.3, emit=((0.55, 0.85, 1.0), 0.12)),
            "decal": simple("Robot_decal", (0.035, 0.035, 0.038), 0.0, 0.5)}


# ----------------------------------------------------------------------------- one-call entry (add-on button)
def add_walking_robot(scene, start, floor_fn, log, speed=0.85, look_at_camera=True, min_dist=3.0, path_xy=None):
    """Robot walking from `start` toward where the camera ends (straight line), or along `path_xy` (N, 2 ground
    points, e.g. from a drawn curve), feet on floor_fn(x, y).

    Straight line: speed is capped so it stops `min_dist` m short of the camera's final position.
    Path: it walks at `speed` and stops at the end of the path if it gets there first.
    """
    cam = scene.camera
    f0, f1 = scene.frame_start, scene.frame_end
    fps = scene.render.fps / scene.render.fps_base
    track = {}
    for fr in range(f0, f1 + 1):
        scene.frame_set(fr)
        track[fr] = cam.matrix_world.translation.copy()
    scene.frame_set(f0)
    dur = max((f1 - f0) / fps, 1e-3)
    if path_xy is not None:
        from .paths import ground_path
        path, length = ground_path(path_xy)
        walk = min(speed * dur, length)
        start = Vector((path_xy[0][0], path_xy[0][1], 0.0))
        d = Vector((track[f1].x - start.x, track[f1].y - start.y))
    else:
        d = Vector((track[f1].x - start.x, track[f1].y - start.y))
        if d.length < min_dist + 0.5:
            raise RuntimeError("3D cursor is too close to the camera: place it further down the shot")
        dirv = d.normalized()
        walk = min(speed * dur, d.length - min_dist)
        p0 = Vector((start.x, start.y))

        def path(s):
            return p0 + dirv * s, dirv.copy()
    v = walk / dur
    if v < 0.35:
        log(f"warning: only {walk:.1f} m to walk in {dur:.1f} s -> slow gait ({v:.2f} m/s)")

    coll = bpy.data.collections.new("CM_Robot")
    scene.collection.children.link(coll)
    mats = make_materials(floor_fn(start.x, start.y))
    parts = build_robot(coll, mats)
    ed = bpy.context.preferences.edit
    old = ed.keyframe_new_interpolation_type
    ed.keyframe_new_interpolation_type = "LINEAR"
    try:
        stats = animate_walk(parts, path, floor_fn, f0, f1, fps, v=max(v, 0.2),
                             look_target=(lambda fr: track[fr]) if look_at_camera else None,
                             look_from=int(f0 + 0.62 * (f1 - f0)) if look_at_camera else None)
    finally:
        ed.keyframe_new_interpolation_type = old
    scene.frame_set(f0)
    end_p, _ = path(walk)
    end_d = (Vector((track[f1].x, track[f1].y)) - end_p).length
    log(f"robot '{coll.name}': walks {walk:.1f} m at {v:.2f} m/s{' along the curve' if path_xy is not None else ''}, "
        f"ends {end_d:.1f} m from the camera "
        f"(leg reach max {stats['max_ratio'] * 100:.0f}%, {stats['clamped']} clamped)")
    return coll, stats
