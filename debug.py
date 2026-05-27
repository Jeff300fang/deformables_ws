#!/usr/bin/env python3

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import viser

JAX_DEFORMABLES_PATH = Path(
    "/home/jeff/trustworthroboticsgroup/CoRL2026/deformables_ws/src/jax-deformable"
)
sys.path.insert(0, str(JAX_DEFORMABLES_PATH))

from environments import RopeEnv


def add_cone_obstacle(
    server,
    x_center=0.05,
    y_center=0.0,
    z_base=0.0,
    height=0.15,
    radius=0.1,
):
    rs = np.linspace(0.0, radius, 30)
    thetas = np.linspace(0.0, 2.0 * np.pi, 80)

    pts = []
    for r in rs:
        z = z_base + height * (1.0 - r / max(radius, 1e-8))
        for theta in thetas:
            x = x_center + r * np.cos(theta)
            y = y_center + r * np.sin(theta)
            pts.append([x, y, z])

    pts = np.asarray(pts, dtype=np.float32)

    server.scene.add_point_cloud(
        name="/cone_obstacle",
        points=pts,
        colors=np.tile(
            np.array([[255, 80, 80]], dtype=np.uint8),
            (pts.shape[0], 1),
        ),
        point_size=0.01,
    )


def draw_rollout(env, server, X, stride=5):
    for k in range(0, X.shape[0], stride):
        xk = jnp.asarray(X[k])
        x_node, _, x_grip = env.unpack_state(xk)

        rope_pts = np.asarray(x_node)
        grip_pts = np.asarray(x_grip)

        server.scene.add_line_segments(
            name=f"/rollout/rope_{k}",
            points=np.stack([rope_pts[:-1], rope_pts[1:]], axis=1),
            colors=np.tile(
                np.array([[0, 255, 0]], dtype=np.uint8),
                (rope_pts.shape[0] - 1, 2, 1),
            ),
            line_width=2.0,
        )

        server.scene.add_point_cloud(
            name=f"/rollout/nodes_{k}",
            points=rope_pts,
            colors=np.tile(
                np.array([[0, 255, 0]], dtype=np.uint8),
                (rope_pts.shape[0], 1),
            ),
            point_size=0.01,
        )

        server.scene.add_point_cloud(
            name=f"/rollout/grips_{k}",
            points=grip_pts,
            colors=np.tile(
                np.array([[255, 0, 0]], dtype=np.uint8),
                (grip_pts.shape[0], 1),
            ),
            point_size=0.03,
        )


def print_adjacent_node_distances(name, rope_pts):
    print(f"\n{name} adjacent node distances:")

    total_length = 0.0

    for i in range(rope_pts.shape[0] - 1):
        p0 = rope_pts[i]
        p1 = rope_pts[i + 1]

        dist = np.linalg.norm(p1 - p0)
        total_length += dist

        dx = p1[0] - p0[0]
        dy = p1[1] - p0[1]
        dz = p1[2] - p0[2]

        print(
            f"node[{i}] <-> node[{i+1}] | "
            f"dist={dist:.6f} m | "
            f"dx={dx:.6f}, dy={dy:.6f}, dz={dz:.6f}"
        )

    print(f"total rope length ({name}): {total_length:.6f} m")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("npz_path", type=str)
    parser.add_argument("--dt", type=float, default=0.2)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--loop", action="store_true")
    args = parser.parse_args()

    npz_path = Path(args.npz_path)
    data = np.load(npz_path)

    print("Loaded:", npz_path)
    print("Keys:", list(data.keys()))

    state = data["state"]
    X = data["X"]
    U = data["U"]

    print("state shape:", state.shape)
    print("X shape:", X.shape)
    print("U shape:", U.shape)

    env = RopeEnv(
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

    server = viser.ViserServer()

    server.scene.add_grid(name="ground")
    add_cone_obstacle(server)

    print("Open viser in browser, then replay starts.")
    time.sleep(1.0)

    print("Open viser in browser.")
    print("Controls:")
    print("  Enter / n : next frame")
    print("  b         : previous frame")
    print("  q         : quit")

    k = 0

    while True:
        server.scene.reset()

        server.scene.add_grid(name="ground")
        add_cone_obstacle(server)

        xk = jnp.asarray(X[k])

        env.visualize(server, xk)

        draw_rollout(env, server, X, stride=args.stride)

        # ------------------------------------------------------------------
        # Distances from rollout state X[k]
        # ------------------------------------------------------------------
        x_node_rollout, _, _ = env.unpack_state(xk)
        rope_pts_rollout = np.asarray(x_node_rollout)

        # ------------------------------------------------------------------
        # Distances from saved "state"
        # ------------------------------------------------------------------
        state_jax = jnp.asarray(state)
        x_node_state, _, _ = env.unpack_state(state_jax)
        rope_pts_state = np.asarray(x_node_state)

        print("\n" + "=" * 80)
        print(f"Frame {k}/{X.shape[0] - 1}")

        print_adjacent_node_distances(
            "ROLLOUT X[k]",
            rope_pts_rollout,
        )

        print_adjacent_node_distances(
            "SAVED state",
            rope_pts_state,
        )

        print("\nControl:")
        print(U[k])

        cmd = input("\n[Enter/n=next, b=back, q=quit] ").strip().lower()

        if cmd == "q":
            break
        elif cmd == "b":
            k = max(0, k - 1)
        else:
            k = min(X.shape[0] - 1, k + 1)


if __name__ == "__main__":
    main()