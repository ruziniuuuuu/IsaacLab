# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for OpenUSD worker initialization in the simulation launcher."""

import os
import subprocess
import sys
import textwrap


def test_launcher_serializes_collider_dense_openusd_physics_parsing():
    """Repeated native parsing stays stable after importing the launcher."""
    code = textwrap.dedent(
        """
        import os

        from isaaclab.app import launch_simulation  # noqa: F401
        from pxr import Gf, Usd, UsdGeom, UsdPhysics

        assert os.environ["PXR_WORK_THREAD_LIMIT"] == "1"

        def make_stage():
            stage = Usd.Stage.CreateInMemory()
            root = UsdGeom.Xform.Define(stage, "/object").GetPrim()
            stage.SetDefaultPrim(root)
            UsdPhysics.RigidBodyAPI.Apply(root)
            for index in range(32):
                mesh = UsdGeom.Mesh.Define(stage, f"/object/collider_{index}")
                mesh.CreatePointsAttr(
                    [
                        Gf.Vec3f(0.0, 0.0, 0.0),
                        Gf.Vec3f(1.0, 0.0, 0.0),
                        Gf.Vec3f(0.0, 1.0, 0.0),
                        Gf.Vec3f(0.0, 0.0, 1.0),
                    ]
                )
                mesh.CreateFaceVertexCountsAttr([3, 3, 3, 3])
                mesh.CreateFaceVertexIndicesAttr(
                    [0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3]
                )
                prim = mesh.GetPrim()
                UsdPhysics.CollisionAPI.Apply(prim)
                UsdPhysics.MassAPI.Apply(prim)
                UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr(
                    "convexHull"
                )
            return stage

        for _ in range(10):
            stage = make_stage()
            UsdPhysics.LoadUsdPhysicsFromRange(stage, ["/object"])
        """
    )
    env = os.environ.copy()
    env.pop("PXR_WORK_THREAD_LIMIT", None)
    result = subprocess.run(
        [sys.executable, "-u", "-c", code],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
        env=env,
    )

    assert result.returncode == 0, result.stdout + result.stderr
