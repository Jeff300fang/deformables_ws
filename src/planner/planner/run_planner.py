#!/usr/bin/env python3

import sys
import time
from pathlib import Path

import numpy as np
import jax

jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import rclpy
import viser

from rclpy.node import Node
from geometry_msgs.msg import PoseArray, PoseStamped
from std_msgs.msg import Bool

JAX_DEFORMABLES_PATH = Path(
    "/home/jeff/trustworthroboticsgroup/CoRL2026/deformables_ws/src/jax-deformable"
)
sys.path.insert(0, str(JAX_DEFORMABLES_PATH))

from environments import RopeEnv
from planners import gpu_sls

def teleport_rope_closest_points_rigid(sampled_points, grip_positions):
    sampled_points = np.asarray(sampled_points, dtype=np.float32)
    grip_positions = np.asarray(grip_positions, dtype=np.float32)

    if sampled_points is None or grip_positions is None:
        return sampled_points

    # closest rope node to each gripper
    closest_idxs = []
    closest_points = []

    for grip_pos in grip_positions:
        dists = np.linalg.norm(sampled_points - grip_pos[None, :], axis=1)
        idx = int(np.argmin(dists))
        closest_idxs.append(idx)
        closest_points.append(sampled_points[idx])

    src = np.asarray(closest_points, dtype=np.float32)
    dst = grip_positions.astype(np.float32)

    src_center = np.mean(src, axis=0)
    dst_center = np.mean(dst, axis=0)

    src_vec = src[1] - src[0]
    dst_vec = dst[1] - dst[0]

    src_norm = np.linalg.norm(src_vec)
    dst_norm = np.linalg.norm(dst_vec)

    if src_norm < 1e-8 or dst_norm < 1e-8:
        # fallback: translation only
        offset = dst_center - src_center
        return (sampled_points + offset).astype(np.float32)

    a = src_vec / src_norm
    b = dst_vec / dst_norm

    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)

    if s < 1e-8:
        if c > 0:
            R = np.eye(3, dtype=np.float32)
        else:
            # 180 degree rotation around any axis perpendicular to a
            tmp = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            if abs(np.dot(tmp, a)) > 0.9:
                tmp = np.array([0.0, 1.0, 0.0], dtype=np.float32)

            axis = np.cross(a, tmp)
            axis = axis / np.linalg.norm(axis)

            K = np.array(
                [
                    [0.0, -axis[2], axis[1]],
                    [axis[2], 0.0, -axis[0]],
                    [-axis[1], axis[0], 0.0],
                ],
                dtype=np.float32,
            )

            R = np.eye(3, dtype=np.float32) + 2.0 * (K @ K)
    else:
        K = np.array(
            [
                [0.0, -v[2], v[1]],
                [v[2], 0.0, -v[0]],
                [-v[1], v[0], 0.0],
            ],
            dtype=np.float32,
        )

        R = (
            np.eye(3, dtype=np.float32)
            + K
            + K @ K * ((1.0 - c) / (s**2))
        )

    teleported = (sampled_points - src_center) @ R.T + dst_center
    return teleported.astype(np.float32)

def teleport_rope_closest_points_shear(sampled_points, grip_positions):
    pts = np.asarray(sampled_points, dtype=np.float32).copy()
    grips = np.asarray(grip_positions, dtype=np.float32)

    # arc-length parameter along rope
    seg = pts[1:] - pts[:-1]
    seg_len = np.linalg.norm(seg, axis=1)
    s_nodes = np.concatenate([[0.0], np.cumsum(seg_len)])

    anchors_s = []
    anchors_delta = []

    for g in grips:
        # closest point on rope segments
        a = pts[:-1]
        b = pts[1:]
        ab = b - a
        ab_len2 = np.sum(ab * ab, axis=1)

        t = np.sum((g[None, :] - a) * ab, axis=1) / np.maximum(ab_len2, 1e-12)
        t = np.clip(t, 0.0, 1.0)

        closest = a + t[:, None] * ab
        d = np.linalg.norm(closest - g[None, :], axis=1)

        k = int(np.argmin(d))
        s_anchor = s_nodes[k] + t[k] * seg_len[k]
        delta = g - closest[k]

        anchors_s.append(s_anchor)
        anchors_delta.append(delta)

    s0, s1 = anchors_s
    d0, d1 = anchors_delta

    if abs(s1 - s0) < 1e-8:
        return pts

    # Linear shear field along rope arc length.
    # This guarantees:
    #   rope(s0) -> left gripper
    #   rope(s1) -> right gripper
    w1 = (s_nodes - s0) / (s1 - s0)
    w0 = 1.0 - w1

    displacement = w0[:, None] * d0[None, :] + w1[:, None] * d1[None, :]
    return (pts + displacement).astype(np.float32)

def resample_fixed_spacing(points, num_points, spacing=0.1):
    pts = np.asarray(points, dtype=np.float32)

    seg = pts[1:] - pts[:-1]
    seg_len = np.linalg.norm(seg, axis=1)

    s = np.concatenate([[0.0], np.cumsum(seg_len)])
    total_len = s[-1]

    # exactly num_points samples
    new_s = np.arange(num_points, dtype=np.float32) * spacing

    new_pts = []
    j = 0

    for target in new_s:
        if target <= total_len:
            while j < len(s) - 2 and s[j + 1] < target:
                j += 1

            denom = max(s[j + 1] - s[j], 1e-8)
            t = (target - s[j]) / denom
            p = (1 - t) * pts[j] + t * pts[j + 1]
        else:
            # extend past observed curve if needed
            direction = pts[-1] - pts[-2]
            direction = direction / max(np.linalg.norm(direction), 1e-8)
            p = pts[-1] + (target - total_len) * direction

        new_pts.append(p.astype(np.float32))

    return np.asarray(new_pts, dtype=np.float32)

def resample_fixed_link_length_extend(points, num_points, link_length=0.1):
    points = np.asarray(points, dtype=np.float32)

    if points.shape[0] < 2:
        return None

    sampled = [points[0].copy()]
    last = points[0].astype(np.float32)

    i = 1
    last_dir = None

    while i < len(points) and len(sampled) < num_points:
        p = points[i]
        v = p - last
        dist = np.linalg.norm(v)

        if dist < 1e-8:
            i += 1
            continue

        direction = v / dist
        last_dir = direction

        if dist >= link_length:
            new_p = last + link_length * direction
            sampled.append(new_p.astype(np.float32))
            last = new_p.astype(np.float32)
            # do not increment i
        else:
            i += 1

    # If observed rope ended early, extend straight using last direction
    if len(sampled) < num_points:
        if last_dir is None:
            v = points[-1] - points[-2]
            norm = np.linalg.norm(v)
            if norm < 1e-8:
                return None
            last_dir = v / norm

        while len(sampled) < num_points:
            new_p = sampled[-1] + link_length * last_dir
            sampled.append(new_p.astype(np.float32))

    return np.asarray(sampled, dtype=np.float32)

def parabola_obstacle_z(x, x_center, z_base, height, half_width):
    u = (x - x_center) / max(half_width, 1e-8)
    return z_base + height * (1.0 - u * u)

def resample_fixed_link_length_extend_lr(
    points,
    num_points,
    link_length=0.1,
    grip_positions=None,
):
    points = np.asarray(points, dtype=np.float32)

    if points.shape[0] < 2:
        return None

    # grip_positions is expected as [left_grip, right_grip]
    if grip_positions is not None:
        grip_positions = np.asarray(grip_positions, dtype=np.float32)
        left_grip = grip_positions[0]
        right_grip = grip_positions[1]

        # Make point order deterministic:
        # points[0] should be the rope end closer to the left gripper.
        d_start_left = np.linalg.norm(points[0] - left_grip)
        d_end_left = np.linalg.norm(points[-1] - left_grip)

        if d_end_left < d_start_left:
            points = points[::-1].copy()

    # First do normal fixed-link resampling from left to right.
    sampled = [points[0].copy()]
    last = points[0].astype(np.float32)

    i = 1
    while i < len(points) and len(sampled) < num_points:
        p = points[i]
        v = p - last
        dist = np.linalg.norm(v)

        if dist < 1e-8:
            i += 1
            continue

        direction = v / dist

        if dist >= link_length:
            new_p = last + link_length * direction
            sampled.append(new_p.astype(np.float32))
            last = new_p.astype(np.float32)
        else:
            i += 1

    if len(sampled) < 2:
        return None

    # Extend deterministically:
    # left, right, left, right, ...
    extend_left_next = True

    while len(sampled) < num_points:
        if extend_left_next:
            # Outward direction from second node toward first node.
            v = sampled[0] - sampled[1]
            norm = np.linalg.norm(v)

            if norm < 1e-8:
                return None

            direction = v / norm
            new_p = sampled[0] + link_length * direction
            sampled.insert(0, new_p.astype(np.float32))

        else:
            # Outward direction from second-last node toward last node.
            v = sampled[-1] - sampled[-2]
            norm = np.linalg.norm(v)

            if norm < 1e-8:
                return None

            direction = v / norm
            new_p = sampled[-1] + link_length * direction
            sampled.append(new_p.astype(np.float32))

        extend_left_next = not extend_left_next

    return np.asarray(sampled, dtype=np.float32)

def state_from_rope_points(
    env,
    state,
    sampled_points,
    grip_positions=None,
):
    sampled_points = jnp.asarray(sampled_points)

    x_node, x_weld, x_grip = env.unpack_state(state)

    if sampled_points.shape != x_node.shape:
        raise ValueError(
            f"sampled_points shape {sampled_points.shape} does not match "
            f"x_node shape {x_node.shape}"
        )

    x_node = sampled_points

    if grip_positions is not None:
        grip_positions = jnp.asarray(grip_positions)

        if grip_positions.shape != x_grip.shape:
            raise ValueError(
                f"grip_positions shape {grip_positions.shape} does not match "
                f"x_grip shape {x_grip.shape}"
            )

        x_grip = grip_positions

    new_state = jnp.concatenate(
        [
            x_node.reshape(-1),
            x_weld.reshape(-1),
            x_grip.reshape(-1),
        ]
    )

    return new_state

def project_grippers_to_nearest_rope_points(sampled_points, grip_positions):
    pts = np.asarray(sampled_points, dtype=np.float32)
    grips = np.asarray(grip_positions, dtype=np.float32).copy()

    new_grips = []

    for grip in grips:
        dists = np.linalg.norm(pts - grip[None, :], axis=1)
        idx = int(np.argmin(dists))
        new_grips.append(pts[idx])

    return np.asarray(new_grips, dtype=np.float32)

def project_rope_to_gripper_vertical_plane(sampled_points, grip_positions):
    pts = np.asarray(sampled_points, dtype=np.float32).copy()
    grips = np.asarray(grip_positions, dtype=np.float32)

    g0 = grips[0]
    g1 = grips[1]

    # horizontal direction between grippers
    d = g1 - g0
    d[2] = 0.0

    norm = np.linalg.norm(d)
    if norm < 1e-8:
        return pts

    e = d / norm

    # project each point onto line through g0 in xy, keep original z
    rel = pts - g0[None, :]
    along = rel[:, 0] * e[0] + rel[:, 1] * e[1]

    pts[:, 0] = g0[0] + along * e[0]
    pts[:, 1] = g0[1] + along * e[1]

    return pts

def add_half_ellipsoid_obstacle(
    server,
    x_center=0.0,
    y_center=0.0,
    z_base=0.0,
    radius_x=0.1,
    radius_y=0.1,
    height_z=0.3,
    color=(255, 100, 100),
    name="/half_ellipsoid_obstacle",
):
    us = np.linspace(0.0, np.pi / 2.0, 80)      # upper half only
    vs = np.linspace(0.0, 2.0 * np.pi, 160)

    pts = []

    for u in us:
        for v in vs:
            x = x_center + radius_x * np.sin(u) * np.cos(v)
            y = y_center + radius_y * np.sin(u) * np.sin(v)
            z = z_base + height_z * np.cos(u)
            pts.append([x, y, z])

    pts = np.asarray(pts, dtype=np.float32)

    server.scene.add_point_cloud(
        name=name,
        points=pts,
        colors=np.tile(
            np.array(color, dtype=np.uint8)[None, :],
            (pts.shape[0], 1),
        ),
        point_size=0.004,
    )

def make_control_constraints(u_min, u_max):
    def constraints(x, u, t):
        return jnp.concatenate((u - u_max, u_min - u))

    return constraints


def make_constant_disturbance(alpha):
    def disturbance(X):
        N, nx = X.shape
        E0 = alpha * jnp.eye(nx, dtype=X.dtype)
        return jnp.broadcast_to(E0, (N, nx, nx))

    return disturbance

# def make_control_and_obstacle_constraints(
#     env,
#     u_min: jnp.ndarray,
#     u_max: jnp.ndarray,
#     cone_centers_xy: jnp.ndarray,
#     cone_radius: float,
#     cone_z_top: float,
#     num_edge_samples: int = 5,
# ):
#     def constraints(x, u, t):
#         control_constraints = jnp.concatenate((u - u_max, u_min - u))

#         rope_nodes, _, gripper_pos = env.unpack_state(x)

#         left_pos, right_pos = gripper_pos[0], gripper_pos[1]
#         effector_constraints = jnp.array([-left_pos[1], right_pos[1]])

#         # --------------------------------------------------
#         # Sample actual rope nodes + points between nodes
#         # --------------------------------------------------
#         node_pts = rope_nodes

#         a = rope_nodes[:-1]
#         b = rope_nodes[1:]

#         alphas = jnp.linspace(
#             0.0,
#             1.0,
#             num_edge_samples + 2,
#         )[1:-1]

#         edge_pts = (
#             a[:, None, :] * (1.0 - alphas[None, :, None])
#             + b[:, None, :] * alphas[None, :, None]
#         ).reshape(-1, 3)

#         pts = jnp.concatenate([node_pts, edge_pts], axis=0)

#         pt_xy = pts[:, 0:2]
#         pt_z = pts[:, 2]

#         # --------------------------------------------------
#         # Cone obstacle constraints on all sampled rope points
#         # constraint <= 0 is feasible
#         # --------------------------------------------------
#         obstacle_constraints = []

#         for cone_center_xy in cone_centers_xy:
#             radial_dist = jnp.linalg.norm(
#                 pt_xy - cone_center_xy[None, :],
#                 axis=1,
#             )

#             z_required = cone_z_top * jnp.maximum(
#                 1.0 - radial_dist / cone_radius,
#                 0.0,
#             )

#             cone_constraints = z_required - pt_z

#             cone_constraints = jnp.where(
#                 radial_dist <= cone_radius,
#                 cone_constraints,
#                 -1.0,
#             )

#             obstacle_constraints.append(cone_constraints)

#         obstacle_constraints = jnp.concatenate(obstacle_constraints)

#         return jnp.concatenate(
#             (
#                 control_constraints,
#                 effector_constraints,
#                 obstacle_constraints,
#             )
#         )

#     return constraints

# def make_control_and_obstacle_constraints(
#     env,
#     u_min: jnp.ndarray,
#     u_max: jnp.ndarray,
#     cone_centers_xy: jnp.ndarray,
#     cone_radius: float,
#     cone_z_top: float,
# ):
#     def constraints(x, u, t):
#         control_constraints = jnp.concatenate((u - u_max, u_min - u))

#         rope_nodes, _, gripper_pos = env.unpack_state(x)
#         left_pos, right_pos = gripper_pos[0], gripper_pos[1]
#         effector_constraints = jnp.array([-left_pos[1], right_pos[1]])

#         node_xy = rope_nodes[:, 0:2]
#         node_z = rope_nodes[:, 2]

#         obstacle_constraints = []
#         for cone_center_xy in cone_centers_xy:
#             radial_dist = jnp.linalg.norm(node_xy - cone_center_xy[None, :], axis=1)
#             z_required = cone_z_top * jnp.maximum(1.0 - (radial_dist / cone_radius), 0.0)

#             cone_constraints = z_required - node_z
#             cone_constraints = jnp.where(
#                 radial_dist <= cone_radius,
#                 cone_constraints,
#                 -1.0,
#             )
#             obstacle_constraints.append(cone_constraints)

#         obstacle_constraints = jnp.concatenate(obstacle_constraints)

#         return jnp.concatenate((control_constraints, effector_constraints, obstacle_constraints))

#     return constraints

# def make_control_and_obstacle_constraints(
#     env,
#     u_min,
#     u_max,
#     obstacle_centers,
#     parabola_radius_x,
#     parabola_width_y,
#     parabola_height,
# ):
#     def constraints(x, u, t):
#         control_constraints = jnp.concatenate((u - u_max, u_min - u))

#         rope_nodes, _, gripper_pos = env.unpack_state(x)

#         left_pos, right_pos = gripper_pos[0], gripper_pos[1]
#         effector_constraints = jnp.array([
#             -left_pos[1],
#             right_pos[1],
#         ])

#         obstacle_constraints = []

#         for center in obstacle_centers:
#             x_c, y_c = center

#             dx = rope_nodes[:, 0] - x_c
#             dy = rope_nodes[:, 1] - y_c
#             z = rope_nodes[:, 2]

#             inside_x = jnp.abs(dx) <= parabola_radius_x
#             inside_y = jnp.abs(dy) <= (parabola_width_y / 2.0)

#             z_required = parabola_height * (
#                 1.0 - (dx / parabola_radius_x) ** 2
#             )

#             constraint = z_required - z

#             active = inside_x & inside_y

#             constraint = jnp.where(active, constraint, -1.0)

#             obstacle_constraints.append(constraint)

#         obstacle_constraints = jnp.concatenate(obstacle_constraints)

#         return jnp.concatenate((
#             control_constraints,
#             effector_constraints,
#             obstacle_constraints,
#         ))

#     return constraints

# def make_control_and_obstacle_constraints(
#     env,
#     u_min: jnp.ndarray,
#     u_max: jnp.ndarray,
#     obstacle_centers_xy: jnp.ndarray,
#     parabola_radius_x: float,
#     parabola_width_y: float,
#     parabola_height: float,
#     z_base: float = 0.0,
#     tau: float = 1e-3,
#     eps: float = 1e-6,
# ):
#     """
#     Constraint convention:
#         constraint <= 0 is feasible.

#     Forbidden parabolic slab:
#         |x - x_c| <= parabola_radius_x
#         |y - y_c| <= parabola_width_y / 2
#         z <= z_base + parabola_height * (1 - ((x - x_c) / parabola_radius_x)^2)

#     The obstacle constraint is positive only inside this forbidden volume.
#     """

#     def smooth_abs(a):
#         return jnp.sqrt(a * a + eps * eps)

#     def smooth_min(a, b):
#         # smooth approximation of min(a, b)
#         return -tau * jax.nn.logsumexp(
#             jnp.stack((-a / tau, -b / tau), axis=0),
#             axis=0,
#         )

#     def smooth_min3(a, b, c):
#         return smooth_min(smooth_min(a, b), c)

#     def constraints(x, u, t):
#         control_constraints = jnp.concatenate((u - u_max, u_min - u))

#         rope_nodes, _, gripper_pos = env.unpack_state(x)

#         left_pos, right_pos = gripper_pos[0], gripper_pos[1]

#         effector_constraints = jnp.array([
#             -left_pos[1],
#             right_pos[1],
#         ])

#         node_x = rope_nodes[:, 0]
#         node_y = rope_nodes[:, 1]
#         node_z = rope_nodes[:, 2]

#         obstacle_constraints = []

#         for center_xy in obstacle_centers_xy:
#             x_c = center_xy[0]
#             y_c = center_xy[1]

#             dx = node_x - x_c
#             dy = node_y - y_c

#             z_surface = z_base + parabola_height * (
#                 1.0 - (dx / parabola_radius_x) ** 2
#             )

#             # Positive means inside each corresponding part of the forbidden slab.
#             phi_x = parabola_radius_x - smooth_abs(dx)
#             phi_y = 0.5 * parabola_width_y - smooth_abs(dy)
#             phi_z = z_surface - node_z

#             # Positive only when inside x slab, inside y slab, and below surface.
#             # Therefore feasible/safe is constraint <= 0.
#             slab_violation = smooth_min3(phi_x, phi_y, phi_z)

#             obstacle_constraints.append(slab_violation)

#         obstacle_constraints = jnp.concatenate(obstacle_constraints)

#         return jnp.concatenate(
#             (
#                 control_constraints,
#                 effector_constraints,
#                 obstacle_constraints,
#             )
#         )

#     return constraints

def make_control_and_obstacle_constraints(
    env,
    u_min: jnp.ndarray,
    u_max: jnp.ndarray,
    obstacle_centers_xy: jnp.ndarray,
    ellipsoid_radius_x: float,
    ellipsoid_radius_y: float,
    ellipsoid_height_z: float,
    z_base: float = 0.0,
):
    """
    Constraint convention:
        constraint <= 0 is feasible.

    Forbidden obstacle:
        upper half ellipsoid centered at (x_c, y_c, z_base)

        ((x-x_c)/rx)^2 + ((y-y_c)/ry)^2 + ((z-z_base)/rz)^2 <= 1
        and z >= z_base

    The constraint is positive inside the upper half ellipsoid.
    """

    def constraints(x, u, t):
        control_constraints = jnp.concatenate((u - u_max, u_min - u))

        rope_nodes, _, gripper_pos = env.unpack_state(x)

        left_pos, right_pos = gripper_pos[0], gripper_pos[1]
        effector_constraints = jnp.array([
            -left_pos[1],
            right_pos[1],
        ])

        node_x = rope_nodes[:, 0]
        node_y = rope_nodes[:, 1]
        node_z = rope_nodes[:, 2]

        obstacle_constraints = []

        for center_xy in obstacle_centers_xy:
            x_c = center_xy[0]
            y_c = center_xy[1]

            dx = (node_x - x_c) / ellipsoid_radius_x
            dy = (node_y - y_c) / ellipsoid_radius_y
            dz = (node_z - z_base) / ellipsoid_height_z

            ellipsoid_value = dx**2 + dy**2 + dz**2

            # positive inside ellipsoid
            violation = 1.0 - ellipsoid_value

            # only keep upper half; below z_base is safe
            violation = jnp.where(node_z >= z_base, violation, -1.0)

            obstacle_constraints.append(violation)

        obstacle_constraints = jnp.concatenate(obstacle_constraints)

        return jnp.concatenate(
            (
                control_constraints,
                effector_constraints,
                obstacle_constraints,
            )
        )

    return constraints

def add_parabolic_wall(
    server,
    x_center=0.0,
    y_center=0.0,
    z_base=0.0,
    radius_x=0.1,
    width_y=0.3,
    height=0.15,
    color=(255, 0, 0),
    name="/parabolic_wall",
):
    """
    Smooth parabolic wall extruded along y.

    Surface:
        z = z_base + height * (1 - (x/radius_x)^2)

    for |x| <= radius_x
    and |y| <= width_y / 2
    """

    xs = np.linspace(-radius_x, radius_x, 160)
    ys = np.linspace(-width_y / 2.0, width_y / 2.0, 80)

    pts = []

    for x in xs:
        z = z_base + height * (
            1.0 - (x / radius_x) ** 2
        )

        for y in ys:
            pts.append([
                x_center + x,
                y_center + y,
                z,
            ])

    pts = np.asarray(pts, dtype=np.float32)

    colors = np.tile(
        np.array(color, dtype=np.uint8)[None, :],
        (pts.shape[0], 1),
    )

    server.scene.add_point_cloud(
        name=name,
        points=pts,
        colors=colors,
        point_size=0.003,
    )

class RopeStateSolverNode(Node):
    def __init__(self):
        super().__init__("rope_state_solver_node")

        self.declare_parameter("rope_state_topic", "/rope_poses")
        self.rope_state_topic = str(self.get_parameter("rope_state_topic").value)

        self.latest_ee_pos_left = None
        self.latest_ee_pos_right = None

        self.latest_ee_pos_seq_left = 0
        self.latest_ee_pos_seq_right = 0

        self.ee_base_frame_seq_left = 0
        self.ee_base_frame_seq_right = 0

        self.last_solve_latest_ee_pos_seq_left = -1
        self.last_solve_latest_ee_pos_seq_right = -1

        self.last_solve_ee_base_frame_seq_left = -1
        self.last_solve_ee_base_frame_seq_right = -1

        self.cone_obstacle_x_center = 0.05
        self.cone_obstacle_y_center = 0.0
        self.cone_obstacle_z_base = 0.0
        self.cone_obstacle_height = 0.3
        self.cone_obstacle_radius = 0.1
        self.cone_obstacle_clearance = 0.03

        self.server = viser.ViserServer()
        _ = self.server.scene.add_grid(name="ground")

        rs = np.linspace(0.0, self.cone_obstacle_radius, 30)
        thetas = np.linspace(0.0, 2.0 * np.pi, 80)

        obstacle_pts = []

        for r in rs:
            z = self.cone_obstacle_z_base + self.cone_obstacle_height * (
                1.0 - r / max(self.cone_obstacle_radius, 1e-8)
            )

            for theta in thetas:
                x = self.cone_obstacle_x_center + r * np.cos(theta)
                y = self.cone_obstacle_y_center + r * np.sin(theta)
                obstacle_pts.append([x, y, z])

        obstacle_pts = np.asarray(obstacle_pts, dtype=np.float32)

        self.server.scene.add_point_cloud(
            name="/cone_obstacle",
            points=obstacle_pts,
            colors=np.tile(
                np.array([[255, 80, 80]], dtype=np.uint8),
                (obstacle_pts.shape[0], 1),
            ),
            point_size=0.01,
        )

        self.env = RopeEnv(
            time_step=0.02,
            num_segments=10,
            rope_length=1.0,
            rope_diameter=0.01,
            youngs_modulus=1e5,
            mass_density=300,
            num_floating_grippers=2,
            grip_stiffness=2000,
            gripper_radius=0.02,
            contact_smoothing=3e-3,
        )
        self.prev_U_solved = None
        self.prev_X_solved = None

        self.N = 50
        self.dt = self.env.params.dt

        self.x_grip = None
        self.state = None
        self.first_solve = True
        self.controller = None

        self.control0 = self.env.control(
            c_grip=jnp.array([1.0, 1.0]),
        )

        self.state_goals = []

        # Subgoal #1, go down and pick up rope
        # y_coords = (
        #     jnp.arange(self.env.params.num_nodes) * self.env.params.segment_length
        #     - 0.5
        # )

        # z_coords = jnp.ones(self.env.params.num_nodes) * 0.05

        # nodes = jnp.stack(
        #     (
        #         jnp.zeros_like(y_coords),  # x
        #         y_coords,                  # y
        #         z_coords,                  # z
        #     ),
        #     axis=1,
        # )

        # self.state_goals.append(self.env.state(x_node=nodes))
        self.state_goals.append(None)

        # Subgoal #2, raise rope to 0.15
        y_coords = (
            jnp.arange(self.env.params.num_nodes) * self.env.params.segment_length
            - 0.5
        )
        # Starting Left EE: x=0.25, y=0.35, z=0.08
        # Starting Right EE: x=0.25, y=-0.38, z=0.12
        # Goal Left EE: x=-0.13, y=0.35, z=0.08
        # Goal Right EE: x=-0.13, y=-0.38, z=0.12
        # right ee end: -0.15, -0.35, z=0.18
        x_coords = jnp.ones(self.env.params.num_nodes) * -0.13
        z_coords = jnp.ones(self.env.params.num_nodes) * 0.1

        nodes = jnp.stack(
            (
                x_coords,                  # x
                y_coords[::-1],            # y
                z_coords,                  # z
            ),
            axis=1,
        )
        self.state_goals.append(self.env.state(x_node=nodes))

        print(self.state_goals[1])

        vmax = 0.2
        u_max = jnp.array([vmax, vmax, vmax, 10.0])
        u_min = jnp.array([-vmax, -vmax, -vmax, -10.0])
        self.u_max = jnp.repeat(u_max, 2)
        self.u_min = jnp.repeat(u_min, 2)
        # self.constraints = make_control_and_obstacle_constraints(
        #     env=self.env,
        #     u_min=self.u_min,
        #     u_max=self.u_max,
        #     cone_centers_xy=jnp.array([
        #         [self.cone_obstacle_x_center,
        #         self.cone_obstacle_y_center]
        #     ]),
        #     cone_radius=self.cone_obstacle_radius,
        #     cone_z_top=self.cone_obstacle_height,
        # )
        self.constraints = make_control_and_obstacle_constraints(
            env=self.env,
            u_min=self.u_min,
            u_max=self.u_max,
            obstacle_centers_xy=jnp.array([
                [
                    self.cone_obstacle_x_center,
                    self.cone_obstacle_y_center,
                ]
            ]),
            ellipsoid_radius_x=0.10,
            ellipsoid_radius_y=0.10,
            ellipsoid_height_z=self.cone_obstacle_height,
            z_base=0.0,
        )

        self.sub = self.create_subscription(
            PoseArray,
            self.rope_state_topic,
            self.rope_state_callback,
            1,
        )

        self.goal_pose_pub_left = self.create_publisher(
            PoseStamped,
            "/left/iiwa/goal_pose",
            1,
        )

        self.goal_pose_pub_right = self.create_publisher(
            PoseStamped,
            "/right/iiwa/goal_pose",
            1,
        )

        self.ee_sub_left = self.create_subscription(
            PoseStamped,
            "/left/workstation/end_effector_pose",
            self.end_effector_left_pose_callback,
            1,
        )

        self.ee_sub_right = self.create_subscription(
            PoseStamped,
            "/right/workstation/end_effector_pose",
            self.end_effector_right_pose_callback,
            1,
        )

        self.ee_base_frame_sub_left = self.create_subscription(
            PoseStamped,
            "/left/end_effector_pose",
            self.ee_base_frame_left_callback,
            1,
        )

        self.ee_base_frame_sub_right = self.create_subscription(
            PoseStamped,
            "/right/end_effector_pose",
            self.ee_base_frame_right_callback,
            1,
        )

        self.grip_pub_left = self.create_publisher(
            Bool,
            "/left_grip",
            1,
        )

        self.grip_pub_right = self.create_publisher(
            Bool,
            "/right_grip",
            1,
        )

        self.get_logger().info(f"Subscribed to {self.rope_state_topic}")
        self.get_logger().info("Waiting for first real gripper pose before initializing state.")

        self.num_iterations = 0
        self.solve_save_dir = Path.cwd() / "rope_mpc_solves"
        self.solve_save_dir.mkdir(parents=True, exist_ok=True)
        self.ee_base_frame_left = None
        self.ee_base_frame_right = None
        self.left_grasping_procedure = False
        self.right_grasping_procedure = False

        self.left_gripper_closed = False
        self.right_gripper_closed = False
        self.grasping_starting_position_left = None
        self.grasping_starting_position_right = None

        self.grip_activation_dist = 0.035
        self.grip_ground_z = 0.005
        self.grip_down_speed = 0.15

        self.in_contact = False
        self.prev_sampled = None
        self.reference = None
        self.grasp_reference_sampled = None
        self.visualize_rollouts_enabled = True

        self.post_grasp_lift_active = False
        self.post_grasp_lift_height = 0.05  # 5 cm
        self.post_grasp_lift_speed = 0.15

        self.left_lift_start_z = None
        self.right_lift_start_z = None


    def ee_base_frame_left_callback(self, msg):
        self.ee_base_frame_left = msg
        self.ee_base_frame_seq_left += 1

    def ee_base_frame_right_callback(self, msg):
        self.ee_base_frame_right = msg
        self.ee_base_frame_seq_right += 1

    def end_effector_left_pose_callback(self, msg):
        p = np.array(
            [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z],
            dtype=np.float32,
        )

        self.latest_ee_pos_left = p
        self.latest_ee_pos_seq_left += 1

    def end_effector_right_pose_callback(self, msg):
        p = np.array(
            [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z],
            dtype=np.float32,
        )

        self.latest_ee_pos_right = p
        self.latest_ee_pos_seq_right += 1

    def get_latest_grip_positions(self):
        if self.latest_ee_pos_left is None or self.latest_ee_pos_right is None:
            return None

        return np.stack(
            [
                self.latest_ee_pos_left,
                self.latest_ee_pos_right,
            ],
            axis=0,
        )

    def initialize_controller(self):
        def cost(W, reference, x, u, t):
            state_err = x[:-6] - reference[:x.shape[-1] - 6]
            control_err = u[:-2] - reference[x.shape[-1]:-2]

            # state weighting
            q = jnp.ones_like(state_err[:-6])

            # rope node coordinates are interleaved [x, y, z]
            q = q.at[0::3].set(.0)   # x weight
            q = q.at[1::3].set(1.0)    # y weight
            q = q.at[2::3].set(1.0)    # z weight

            state_cost = jnp.sum(q * (state_err[:-6] ** 2))

            control_cost = 0.1 * jnp.sum(control_err[:-2] ** 2)

            return state_cost + control_cost

        def dynamics(x, u, t, parameter):
            return self.env.step(x, u)

        admm_cfg = gpu_sls.ADMMConfig(
            eps_abs=5e-2,
            eps_rel=1e-2,
            rho_max=1e3,
            max_iterations=400,
            rho_update_frequency=25,
            initial_rho=10.0,
        )

        sls_cfg = gpu_sls.SLSConfig(
            max_sls_iterations=1,
            sls_primal_tol=1e-2,
            enable_fastsls=False,
            initialize_nominal=True,
            max_initial_sqp_iterations=0,
            warm_start=False,
            rti=False,
        )

        sqp_cfg = gpu_sls.SQPConfig(
            max_sqp_iterations=1,
            warm_start=False,
            feas_tol=1e-2,
            step_tol=1e-4,
            line_search=True,
        )

        cfg = gpu_sls.MPCConfig(
            n=self.state.size,
            nu=self.control0.size,
            N=self.N,
            dt=self.dt,
            W=None,
            u_ref=self.control0,
        )

        disturbance = make_constant_disturbance(
            alpha=0.03 * self.dt,
        )

        nc = self.constraints(self.state, self.control0, 0.0).size

        self.controller = gpu_sls.GenericMPC(
            sls_cfg,
            sqp_cfg,
            admm_cfg,
            config=cfg,
            dynamics=dynamics,
            cost=cost,
            constraints=self.constraints,
            obstacles=jnp.zeros((0, 3)),
            disturbance=disturbance,
            num_constraints=nc,
            shift=1,
            X_in=None,
            U_in=None,
        )

        self.get_logger().info("Initialized MPC controller.")

    # def closest_point_on_rope_segments(self, points, p):
    #     points = np.asarray(points, dtype=np.float32)
    #     p = np.asarray(p, dtype=np.float32)

    #     a = points[:-1]
    #     b = points[1:]
    #     ab = b - a

    #     ab_len2 = np.sum(ab * ab, axis=1)
    #     ap = p[None, :] - a

    #     t = np.sum(ap * ab, axis=1) / np.maximum(ab_len2, 1e-12)
    #     t = np.clip(t, 0.0, 1.0)

    #     closest = a + t[:, None] * ab
    #     dists = np.linalg.norm(closest - p[None, :], axis=1)

    #     seg_idx = int(np.argmin(dists))
    #     return float(dists[seg_idx]), seg_idx, float(t[seg_idx]), closest[seg_idx]

    def closest_point_on_rope_segments(self, points, p):
        points = np.asarray(points, dtype=np.float32)
        p = np.asarray(p, dtype=np.float32)

        a = points[:-1]
        b = points[1:]
        ab = b - a

        ab_len2 = np.sum(ab * ab, axis=1)
        ap = p[None, :] - a

        t = np.sum(ap * ab, axis=1) / np.maximum(ab_len2, 1e-12)
        t = np.clip(t, 0.0, 1.0)

        closest = a + t[:, None] * ab

        # Only compare z-distance
        dists = np.abs(closest[:, 2] - p[2])

        seg_idx = int(np.argmin(dists))

        return (
            float(dists[seg_idx]),
            seg_idx,
            float(t[seg_idx]),
            closest[seg_idx],
        )

    def visualize_rollouts(
        self,
        X_rollout,
        raw_points=None,
        stride=5,
    ):
        """
        Debug visualization of predicted MPC rollout states.
        """

        # clear previous debug rollout objects
        self.server.scene.reset()

        # rs = np.linspace(0.0, self.cone_obstacle_radius, 30)
        # thetas = np.linspace(0.0, 2.0 * np.pi, 80)

        # obstacle_pts = []

        # for r in rs:
        #     z = self.cone_obstacle_z_base + self.cone_obstacle_height * (
        #         1.0 - r / max(self.cone_obstacle_radius, 1e-8)
        #     )

        #     for theta in thetas:
        #         x = self.cone_obstacle_x_center + r * np.cos(theta)
        #         y = self.cone_obstacle_y_center + r * np.sin(theta)
        #         obstacle_pts.append([x, y, z])

        # obstacle_pts = np.asarray(obstacle_pts, dtype=np.float32)

        # self.server.scene.add_point_cloud(
        #     name="/cone_obstacle",
        #     points=obstacle_pts,
        #     colors=np.tile(
        #         np.array([[255, 80, 80]], dtype=np.uint8),
        #         (obstacle_pts.shape[0], 1),
        #     ),
        #     point_size=0.01,
        # )

        add_half_ellipsoid_obstacle(
            self.server,
            x_center=self.cone_obstacle_x_center,
            y_center=self.cone_obstacle_y_center,
            z_base=0.0,
            radius_x=0.10,
            radius_y=0.10,
            height_z=self.cone_obstacle_height,
            color=(255, 100, 100),
            name="/obstacle",
        )
        # keep ground
        _ = self.server.scene.add_grid(name="ground")

        # render current state normally
        self.env.visualize(self.server, self.state)

        # raw observed rope
        if raw_points is not None:
            raw_points_np = np.asarray(raw_points, dtype=np.float32)
            self.server.scene.add_point_cloud(
                name="/debug/raw_rope_points",
                points=raw_points_np,
                colors=np.tile(
                    np.array([[255, 255, 255]], dtype=np.uint8),
                    (raw_points_np.shape[0], 1),
                ),
                point_size=0.008,
            )

        X_rollout_np = np.asarray(X_rollout)

        for k in range(0, X_rollout_np.shape[0], stride):
            xk = jnp.asarray(X_rollout_np[k])

            x_node, x_weld, x_grip = self.env.unpack_state(xk)

            rope_pts = np.asarray(x_node)

            # rollout rope polyline
            self.server.scene.add_line_segments(
                name=f"/debug/rollout_rope_{k}",
                points=np.stack(
                    [rope_pts[:-1], rope_pts[1:]],
                    axis=1,
                ),
                colors=np.tile(
                    np.array([[0, 255, 0]], dtype=np.uint8),
                    (rope_pts.shape[0] - 1, 2, 1),
                ),
                line_width=2.0,
            )

            # rollout rope nodes
            self.server.scene.add_point_cloud(
                name=f"/debug/rollout_nodes_{k}",
                points=rope_pts,
                colors=np.tile(
                    np.array([[0, 255, 0]], dtype=np.uint8),
                    (rope_pts.shape[0], 1),
                ),
                point_size=0.01,
            )

            # rollout grippers
            x_grip_np = np.asarray(x_grip)

            self.server.scene.add_point_cloud(
                name=f"/debug/rollout_grips_{k}",
                points=x_grip_np,
                colors=np.tile(
                    np.array([[255, 0, 0]], dtype=np.uint8),
                    (x_grip_np.shape[0], 1),
                ),
                point_size=0.03,
            )

    def grasp_one_side(self, side):
        if side == "left":
            ee_base_frame = self.ee_base_frame_left
            latest_ee_pos = self.latest_ee_pos_left
            goal_pub = self.goal_pose_pub_left
            grip_pub = self.grip_pub_left
            starting_attr = "grasping_starting_position_left"
            closed_attr = "left_gripper_closed"
        else:
            ee_base_frame = self.ee_base_frame_right
            latest_ee_pos = self.latest_ee_pos_right
            goal_pub = self.goal_pose_pub_right
            grip_pub = self.grip_pub_right
            starting_attr = "grasping_starting_position_right"
            closed_attr = "right_gripper_closed"

        if latest_ee_pos is None or ee_base_frame is None:
            self.get_logger().warn(f"Cannot grasp {side}: missing EE pose.")
            return

        if getattr(self, closed_attr):
            return

        if getattr(self, starting_attr) is None:
            setattr(self, starting_attr, ee_base_frame)

        if latest_ee_pos[2] <= self.grip_ground_z:
            grip_pub.publish(Bool(data=True))
            setattr(self, closed_attr, True)
            self.get_logger().info(f"Closing {side} gripper")
            return

        start_pose = getattr(self, starting_attr)

        goal_pose = PoseStamped()
        goal_pose.header.stamp = self.get_clock().now().to_msg()
        goal_pose.header.frame_id = "world"

        goal_pose.pose.position.x = float(start_pose.pose.position.x)
        goal_pose.pose.position.y = float(start_pose.pose.position.y)
        goal_pose.pose.position.z = float(
            ee_base_frame.pose.position.z - self.grip_down_speed * self.dt
        )
        goal_pose.pose.orientation.w = 1.0

        goal_pub.publish(goal_pose)
            
    def rope_state_callback(self, msg):
        # ==========================================================
        # After both grippers close, lift upward by 5 cm
        # ==========================================================
        if (
            self.left_gripper_closed
            and self.right_gripper_closed
            and not self.in_contact
        ):
            # initialize lift phase
            if not self.post_grasp_lift_active:
                time.sleep(3)
                self.post_grasp_lift_active = True

                self.left_lift_start_z = (
                    self.ee_base_frame_left.pose.position.z
                )

                self.right_lift_start_z = (
                    self.ee_base_frame_right.pose.position.z
                )

                self.get_logger().info(
                    "Both grippers closed. Starting post-grasp lift."
                )

            left_current_z = self.ee_base_frame_left.pose.position.z
            right_current_z = self.ee_base_frame_right.pose.position.z

            left_done = (
                left_current_z
                >= self.left_lift_start_z + self.post_grasp_lift_height
            )

            right_done = (
                right_current_z
                >= self.right_lift_start_z + self.post_grasp_lift_height
            )

            # publish left upward motion
            if not left_done:
                left_goal_pose = PoseStamped()
                left_goal_pose.header.stamp = self.get_clock().now().to_msg()
                left_goal_pose.header.frame_id = "world"

                left_goal_pose.pose.position.x = float(
                    self.ee_base_frame_left.pose.position.x
                )

                left_goal_pose.pose.position.y = float(
                    self.ee_base_frame_left.pose.position.y
                )

                left_goal_pose.pose.position.z = float(
                    self.ee_base_frame_left.pose.position.z
                    + self.post_grasp_lift_speed * self.dt
                )
                self.get_logger().info("Publishing up left")
                left_goal_pose.pose.orientation.w = 1.0

                self.goal_pose_pub_left.publish(left_goal_pose)

            # publish right upward motion
            if not right_done:
                right_goal_pose = PoseStamped()
                right_goal_pose.header.stamp = self.get_clock().now().to_msg()
                right_goal_pose.header.frame_id = "world"

                right_goal_pose.pose.position.x = float(
                    self.ee_base_frame_right.pose.position.x
                )

                right_goal_pose.pose.position.y = float(
                    self.ee_base_frame_right.pose.position.y
                )

                right_goal_pose.pose.position.z = float(
                    self.ee_base_frame_right.pose.position.z
                    + self.post_grasp_lift_speed * self.dt
                )

                right_goal_pose.pose.orientation.w = 1.0
                self.get_logger().info("Publishing up right")
                self.goal_pose_pub_right.publish(right_goal_pose)

            # once both lifted enough, transition into contact mode
            if left_done and right_done:
                self.get_logger().info(
                    "Post-grasp lift complete. Enabling contact mode."
                )

                self.in_contact = True
                self.first_solve = True
                self.post_grasp_lift_active = False
            time.sleep(0.01)
            return

        if self.num_iterations >= 1000:
            self.get_logger().info("Done")
            return

        if self.ee_base_frame_right is None or self.ee_base_frame_left is None:
            self.get_logger().info("No ee base frame pose")
            return

        if len(msg.poses) == 0:
            self.get_logger().info("Skipping, no poses")
            return

        grip_positions = self.get_latest_grip_positions()

        if grip_positions is None:
            self.get_logger().warn("No left/right end effector poses yet. Skipping solve.")
            return

        callback_start = time.time()

        raw_points = np.asarray(
            [
                [
                    pose.position.x,
                    pose.position.y,
                    pose.position.z,
                ]
                for pose in msg.poses
            ],
            dtype=np.float32,
        )
        if raw_points.shape[0] == 0:
            self.get_logger().warn("Missing points")
            return


        if not self.in_contact:
            sampled = resample_fixed_link_length_extend(
                raw_points,
                self.env.params.num_nodes,
                link_length=self.env.params.segment_length,
            )

        if self.in_contact:
            sampled = resample_fixed_link_length_extend_lr(
                raw_points,
                self.env.params.num_nodes,
                link_length=0.1,
                grip_positions=grip_positions,
            )

        if self.grasp_reference_sampled is None:
            self.grasp_reference_sampled = sampled

        if self.grasp_reference_sampled is None:
            self.get_logger().warn("grasp referenced sampled is None")
            return

        closest_dist_left, closest_seg_idx, closest_t, closest_point = (
            self.closest_point_on_rope_segments(self.grasp_reference_sampled, grip_positions[0])
        )

        closest_dist_right, closest_seg_idx, closest_t, closest_point = (
            self.closest_point_on_rope_segments(self.grasp_reference_sampled, grip_positions[1])
        )
        if not self.in_contact:
            self.get_logger().info(
                f"Closest LEFT rope point: dist={closest_dist_left:.4f} m, grip={grip_positions[0]}"
            )
            self.get_logger().info(
                f"Closest RIGHT rope point: dist={closest_dist_right:.4f} m, grip={grip_positions[1]}"
            )

        if (
            closest_dist_left <= self.grip_activation_dist
            and not self.left_grasping_procedure
            and not self.left_gripper_closed
        ):
            self.left_grasping_procedure = True
            self.get_logger().info("LEFT entered gripping range; switching LEFT to grasp subroutine.")

        if (
            closest_dist_right <= self.grip_activation_dist
            and not self.right_grasping_procedure
            and not self.right_gripper_closed
        ):
            self.right_grasping_procedure = True
            self.get_logger().info("RIGHT entered gripping range; switching RIGHT to grasp subroutine.")

        if sampled is None:
            self.get_logger().warn("Could not resample rope_state.")
            return

        if self.state is None:
            self.x_grip = jnp.asarray(grip_positions)
            self.state = self.env.state(x_grip=self.x_grip)

            self.state = state_from_rope_points(
                self.env,
                self.state,
                sampled,
                grip_positions=grip_positions,
            )

            self.get_logger().info(
                "Initialized state from first rope state and real gripper pose."
            )
        self.get_logger().info(f"CURRENT STATE {self.state}")
        if self.state_goals[0] is None:
            nodes_goal = jnp.asarray(sampled)
            nodes_goal = nodes_goal.at[:, 2].set(0.05)

            self.state_goals[0] = self.env.state(x_node=nodes_goal)
            self.get_logger().info("Initialized subgoal #1 from current rope state.")

        if not self.in_contact:
            x_node, x_weld, x_grip = self.env.unpack_state(self.state)

            grip_positions_jnp = jnp.asarray(grip_positions)

            self.state = jnp.concatenate(
                [
                    x_node.reshape(-1),
                    x_weld.reshape(-1),
                    grip_positions_jnp.reshape(-1),
                ]
            )

        if self.in_contact:
            # sampled = teleport_rope_closest_points_rigid(
            #     sampled_points=sampled,
            #     grip_positions=grip_positions,
            # )
            sampled = teleport_rope_closest_points_shear(
                            sampled_points=sampled,
                            grip_positions=grip_positions,
                        )

            sampled = project_rope_to_gripper_vertical_plane(
                sampled_points=sampled,
                grip_positions=grip_positions,
            )

            sampled = resample_fixed_spacing(
                sampled,
                num_points=self.env.params.num_nodes,
                spacing=self.env.params.segment_length,
            )

            grip_positions = project_grippers_to_nearest_rope_points(
                sampled_points=sampled,
                grip_positions=grip_positions,
            )
            
            # alpha = 0.7
            # if self.prev_sampled is not None:
            #     sampled = alpha * self.prev_sampled + (1.0 - alpha) * sampled
            # self.prev_sampled = sampled.copy()

            sampled_jnp = jnp.asarray(sampled)
            
            x_node, x_weld, x_grip = self.env.unpack_state(self.state)

            grip_positions_jnp = jnp.asarray(grip_positions)

            self.state = state_from_rope_points(
                self.env,
                self.state,
                sampled_jnp,
                grip_positions=grip_positions,
            )

        if self.controller is None:
            self.initialize_controller()
        if not self.visualize_rollouts_enabled:
            self.env.visualize(self.server, self.state)

        if self.first_solve and not self.in_contact:
            X_in = jnp.tile(self.state[None, :], (self.N + 1, 1))
            U_in = jnp.tile(self.control0[None, :], (self.N, 1))
            U_in = U_in.at[:, 2].set(-0.2)
            U_in = U_in.at[:, 5].set(-0.2)

            self.controller.X0 = X_in
            self.controller.U0 = U_in
            self.reference = jnp.concatenate([self.state_goals[0], self.control0], axis=0)
            self.first_solve = False
        
        if self.first_solve and self.in_contact:
            self.controller.X0 = jnp.tile(self.state[None, :], (self.N + 1, 1))
            self.controller.U0 = jnp.tile(self.control0[None, :], (self.N, 1))
            self.reference = jnp.concatenate([self.state_goals[1], self.control0], axis=0)
            self.first_solve = False

        if (
            self.latest_ee_pos_seq_left
            <= self.last_solve_latest_ee_pos_seq_left
            or self.latest_ee_pos_seq_right
            <= self.last_solve_latest_ee_pos_seq_right
            or self.ee_base_frame_seq_left
            <= self.last_solve_ee_base_frame_seq_left
            or self.ee_base_frame_seq_right
            <= self.last_solve_ee_base_frame_seq_right
        ):
            self.get_logger().info(
                "Skipping solve because one or more left/right EE poses "
                "or base-frame poses have not updated since last solve."
            )
            return

        self.last_solve_latest_ee_pos_seq_left = (
            self.latest_ee_pos_seq_left
        )

        self.last_solve_latest_ee_pos_seq_right = (
            self.latest_ee_pos_seq_right
        )

        self.last_solve_ee_base_frame_seq_left = (
            self.ee_base_frame_seq_left
        )

        self.last_solve_ee_base_frame_seq_right = (
            self.ee_base_frame_seq_right
        )
        self.controller.U0 = self.controller.U0.at[:, -2:].set(1.0)
        solve_start = time.time()
        out = self.controller.run(
            x0=self.state,
            reference=self.reference,
            parameter=None,
            Xi=jnp.zeros((self.state.shape[0], self.state.shape[0])),
        )

        solve_time = time.time() - solve_start

        X_sol = out[1]
        U_sol = out[2]

        save_path = self.solve_save_dir / f"solve_{self.num_iterations:05d}.npz"
        self.num_iterations += 1
        np.savez_compressed(
            save_path,
            iteration=np.array(self.num_iterations),
            state=np.asarray(self.state),
            X=np.asarray(X_sol),
            U=np.asarray(U_sol),
            solve_time=np.array(solve_time),
            in_contact=np.array(self.in_contact),
            left_ee_pos=np.asarray(self.latest_ee_pos_left),
            right_ee_pos=np.asarray(self.latest_ee_pos_right),
        )

        self.get_logger().info(f"Saved solve debug data to {save_path}")

        U_sol.block_until_ready()

        solve_converged = out[-1]
        self.get_logger().info(f"{solve_converged}")

        constraint_vals = self.constraints(
            self.state,
            self.control0,
            0.0,
        )

        constraint_vals_np = np.asarray(constraint_vals)

        self.get_logger().info("\n================ CONSTRAINT VALUES ================")
        self.get_logger().info(f"{constraint_vals_np}")

        print("===================================================\n")

        U = U_sol
        X = X_sol
        if solve_converged:
            pass
            # U = U_sol
            # X = X_sol
            # U_shift = jnp.concatenate(
            #     [
            #         U_sol[1:],
            #         U_sol[-1:],
            #     ],
            #     axis=0,
            # )

            # # Shift states forward.
            # X_shift = jnp.concatenate(
            #     [
            #         X_sol[1:],
            #         X_sol[-1:],
            #     ],
            #     axis=0,
            # )
            # self.prev_U_solved = U_sol
            # self.prev_X_solved = X_sol
        elif self.in_contact and not solve_converged:
            # if self.prev_U_solved is None:
            #     self.get_logger().warn("No previous solved trajectory.")
            #     return

            # # Use stored fallback.
            # U = self.prev_U_solved
            # X = self.prev_X_solved

            # # Shift forward for next iteration.
            # U_shift = jnp.concatenate(
            #     [
            #         U[1:],
            #         U[-1:],
            #     ],
            #     axis=0,
            # )

            # X_shift = jnp.concatenate(
            #     [
            #         X[1:],
            #         X[-1:],
            #     ],
            #     axis=0,
            # )

            # # Re-anchor initial state.
            # X_shift = X_shift.at[0].set(self.state)

            # self.prev_U_solved = U_shift
            # self.prev_X_solved = X_shift

            # Also use shifted version as warm start.
            self.controller.U0 = jnp.tile(self.control0[None, :], (self.N, 1))
            self.controller.X0 = jnp.tile(self.state[None, :], (self.N + 1, 1))
            # self.get_logger().info("Failed solve, retrying")
            # return

        # Force first state to match current measured state.
        # X_shift = X_shift.at[0].set(self.state)

        # self.prev_U_solved = U_shift
        # self.prev_X_solved = X_shift
        # ==========================================================
        # Debug rollout visualization
        # ==========================================================
        if self.visualize_rollouts_enabled:
            X_rollout = jnp.stack(X)

            self.visualize_rollouts(
                X_rollout,
                raw_points=raw_points,
                stride=5,
            )

        u0_np = np.asarray(U[0])
        u1_np = np.asarray(U[1])

        if not np.isfinite(u0_np).all():
            self.get_logger().warn("u0 has NaN/Inf. Skipping.")
            return

        u0_np = np.clip(u0_np, -np.asarray(self.u_max), np.asarray(self.u_max))
        u1_np = np.clip(u1_np, -np.asarray(self.u_max), np.asarray(self.u_max))
        # control = u0_np * 1.0
        # control = u0_np + u1_np
        control = u0_np
        if not self.in_contact:
            control = u0_np * 2.0

        left_control = control[0:3]
        right_control = control[3:6]
        if self.left_grasping_procedure and not self.in_contact:
            self.grasp_one_side("left")
        else:
            left_goal_pose = PoseStamped()
            left_goal_pose.header.stamp = self.get_clock().now().to_msg()
            left_goal_pose.header.frame_id = "world"
            left_goal_pose.pose.position.x = float(
                self.ee_base_frame_left.pose.position.x + left_control[0] * self.dt
            )
            left_goal_pose.pose.position.y = float(
                self.ee_base_frame_left.pose.position.y + left_control[1] * self.dt
            )
            left_goal_pose.pose.position.z = float(
                self.ee_base_frame_left.pose.position.z + left_control[2] * self.dt
            )
            left_goal_pose.pose.orientation.w = 1.0
            self.goal_pose_pub_left.publish(left_goal_pose)

        if self.right_grasping_procedure and not self.in_contact:
            self.grasp_one_side("right")
        else:
            right_goal_pose = PoseStamped()
            right_goal_pose.header.stamp = self.get_clock().now().to_msg()
            right_goal_pose.header.frame_id = "world"
            right_goal_pose.pose.position.x = float(
                self.ee_base_frame_right.pose.position.x + right_control[0] * self.dt
            )
            right_goal_pose.pose.position.y = float(
                self.ee_base_frame_right.pose.position.y + right_control[1] * self.dt
            )
            right_goal_pose.pose.position.z = float(
                self.ee_base_frame_right.pose.position.z + right_control[2] * self.dt
            )
            right_goal_pose.pose.orientation.w = 1.0
            self.goal_pose_pub_right.publish(right_goal_pose)

        callback_time = time.time() - callback_start

        self.get_logger().info("=" * 80)
        self.get_logger().info(f"ITERATION NUMBER: {self.num_iterations}")
        self.get_logger().info(
            f"LEFT goal pose: "
            f"x={self.ee_base_frame_left.pose.position.x + left_control[0] * self.dt:.4f} "
            f"y={self.ee_base_frame_left.pose.position.y + left_control[1] * self.dt:.4f} "
            f"z={self.ee_base_frame_left.pose.position.z + left_control[2] * self.dt:.4f}"
        )

        self.get_logger().info(
            f"RIGHT goal pose: "
            f"x={self.ee_base_frame_right.pose.position.x + right_control[0] * self.dt:.4f} "
            f"y={self.ee_base_frame_right.pose.position.y + right_control[1] * self.dt:.4f} "
            f"z={self.ee_base_frame_right.pose.position.z + right_control[2] * self.dt:.4f}"
        )

        self.get_logger().info(
            f"Received rope_state with {raw_points.shape[0]} raw points"
        )

        self.get_logger().info(
            f"Current LEFT EE pos: {grip_positions[0]}"
        )

        self.get_logger().info(
            f"Current RIGHT EE pos: {grip_positions[1]}"
        )

        self.get_logger().info(
            f"MPC solve time: {solve_time * 1000:.2f} ms"
        )

        self.get_logger().info(
            f"Total callback time: {callback_time * 1000:.2f} ms"
        )

        self.get_logger().info(f"u0_left: {u0_np[0:3]}")
        self.get_logger().info(f"u0_right:  {u0_np[3:6]}")

        self.get_logger().info(f"control_left:  {left_control}")
        self.get_logger().info(f"control_right: {right_control}")

        self.get_logger().info("=" * 80)
        time.sleep(0.5)


def main(args=None):
    rclpy.init(args=args)

    node = RopeStateSolverNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()