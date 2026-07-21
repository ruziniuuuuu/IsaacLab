# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Standalone Isaac Lab reproducer for intermittent OVRTX robot-link loss.

This file intentionally does not import GalbotLab. It runs four synchronized
copies of the bundled robot with Newton/MJWarp physics, OVRTX RGB cameras,
and the Newton visualizer.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import platform
import sys
from collections import deque
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import newton_usd_schemas  # noqa: F401  # Register Newton schemas before USD opens.
import torch
from PIL import Image
from torch.nn import functional as torch_functional

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg

try:
    from isaaclab_tasks.utils import launch_simulation
except ImportError:  # Isaac Lab develop moved the launcher into the core package.
    from isaaclab.app import launch_simulation
from isaaclab_newton.physics import (
    MJWarpSolverCfg,
    NewtonCfg,
    NewtonCollisionPipelineCfg,
    NewtonShapeCfg,
)
from isaaclab_newton.sim.schemas import (
    NewtonArticulationRootPropertiesCfg,
    NewtonMaterialPropertiesCfg,
)
from isaaclab_ov.renderers import OVRTXRendererCfg
from isaaclab_ov.renderers import ovrtx_renderer as ovrtx_renderer_module
from isaaclab_visualizers.newton import NewtonVisualizerCfg

from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedEnv, ManagerBasedEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim import SimulationCfg, UsdFileCfg
from isaaclab.utils.configclass import configclass

HERE = Path(__file__).resolve().parent
ROBOT_USD = HERE / "assets" / "galbot_one_golf" / "galbot_one_golf.usda"
BACKDROP_USD = HERE / "assets" / "backdrop.usda"
CAMERA_PRIM_PATH = "{ENV_REGEX_NS}/Camera"
VIEWER_CAMERA_PRIM_PATH = CAMERA_PRIM_PATH.replace("{ENV_REGEX_NS}", "/World/envs/env_.*")

INITIAL_JOINT_POSITIONS = {
    "leg_joint1": 0.0,
    "leg_joint2": 0.0,
    "leg_joint3": 0.0,
    "leg_joint4": 0.0,
    "leg_joint5": 0.0,
    # Keep the arms outstretched so every arm link is visible and missing
    # geometry cannot hide behind the torso or another arm link.
    "left_arm_joint1": 0.0,
    "left_arm_joint2": 0.0,
    "left_arm_joint3": 0.0,
    "left_arm_joint4": 0.0,
    "left_arm_joint5": 0.0,
    "left_arm_joint6": 0.0,
    "left_arm_joint7": 0.0,
    "right_arm_joint1": 0.0,
    "right_arm_joint2": 0.0,
    "right_arm_joint3": 0.0,
    "right_arm_joint4": 0.0,
    "right_arm_joint5": 0.0,
    "right_arm_joint6": 0.0,
    "right_arm_joint7": 0.0,
    "left_gripper_joint": 0.0,
    "right_gripper_joint": 0.0,
    "wheel1_joint": 0.0,
    "wheel2_joint": 0.0,
    "wheel3_joint": 0.0,
    "wheel4_joint": 0.0,
    "head_joint1": 0.0,
    "head_joint2": 0.30,
}
MOTION_JOINTS = [
    "left_arm_joint1",
    "left_arm_joint2",
    "left_arm_joint3",
    "left_arm_joint4",
    "left_arm_joint5",
    "left_arm_joint6",
    "left_arm_joint7",
    "right_arm_joint1",
    "right_arm_joint2",
    "right_arm_joint3",
    "right_arm_joint4",
    "right_arm_joint5",
    "right_arm_joint6",
    "right_arm_joint7",
    "head_joint1",
    "head_joint2",
]


@configclass
class EmptyManagerCfg:
    """No actions or observations are needed for the rendering reproducer."""

    pass


def make_robot_actuators() -> dict[str, ImplicitActuatorCfg]:
    """Return the robot's source-MJCF drive parameters using Isaac Lab configs."""

    common = {"armature": 0.1, "friction": 0.5}
    return {
        "base_wheels": ImplicitActuatorCfg(
            joint_names_expr=["wheel[1-4]_joint"],
            stiffness=0.0,
            damping=50.0,
            effort_limit_sim=120.0,
            velocity_limit_sim=25.0,
            **common,
        ),
        "legs": ImplicitActuatorCfg(
            joint_names_expr=["leg_joint[1-5]"],
            stiffness={
                "leg_joint[12]": 10000.0,
                "leg_joint3": 6000.0,
                "leg_joint[45]": 1600.0,
            },
            damping={
                "leg_joint[12]": 2000.0,
                "leg_joint3": 600.0,
                "leg_joint[45]": 160.0,
            },
            effort_limit_sim={
                "leg_joint[12]": 420.0,
                "leg_joint3": 280.0,
                "leg_joint4": 200.0,
                "leg_joint5": 100.0,
            },
            velocity_limit_sim=1.0,
            **common,
        ),
        "left_arm": ImplicitActuatorCfg(
            joint_names_expr=["left_arm_joint[1-7]"],
            stiffness={
                ".*joint[12]": 5000.0,
                ".*joint[345]": 3000.0,
                ".*joint[67]": 1000.0,
            },
            damping={
                ".*joint[12]": 500.0,
                ".*joint[345]": 300.0,
                ".*joint[67]": 100.0,
            },
            effort_limit_sim={
                ".*joint[12]": 180.0,
                ".*joint[34]": 60.0,
                ".*joint[567]": 30.0,
            },
            velocity_limit_sim=1.5,
            **common,
        ),
        "right_arm": ImplicitActuatorCfg(
            joint_names_expr=["right_arm_joint[1-7]"],
            stiffness={
                ".*joint[12]": 5000.0,
                ".*joint[345]": 3000.0,
                ".*joint[67]": 1000.0,
            },
            damping={
                ".*joint[12]": 500.0,
                ".*joint[345]": 300.0,
                ".*joint[67]": 100.0,
            },
            effort_limit_sim={
                ".*joint[12]": 180.0,
                ".*joint[34]": 60.0,
                ".*joint[567]": 30.0,
            },
            velocity_limit_sim=1.5,
            **common,
        ),
        "grippers": ImplicitActuatorCfg(
            joint_names_expr=["(left|right)_gripper_joint"],
            stiffness=100.0,
            damping=20.0,
            effort_limit_sim=5.0,
            velocity_limit_sim=2.0,
            **common,
        ),
        "head": ImplicitActuatorCfg(
            joint_names_expr=["head_joint[12]"],
            stiffness=340.0,
            damping=34.0,
            effort_limit_sim=10.0,
            velocity_limit_sim=1.5,
            **common,
        ),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse reproducer-only arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--reset-every", type=int, default=0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--width", type=int, default=298)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument(
        "--viewer",
        choices=("newton", "none"),
        default="newton",
        help="Use 'none' as a control run that keeps OVRTX but removes the viewer.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Create the Newton visualizer offscreen; useful for automated runs.",
    )
    parser.add_argument(
        "--ovrtx-cloning",
        action="store_true",
        help="Enable OVRTX cloning. The reported configuration leaves it disabled.",
    )
    parser.add_argument("--rgb-threshold", type=int, default=40)
    parser.add_argument("--bad-rgb-pixel-threshold", type=int, default=100)
    parser.add_argument("--consecutive-failure-frames", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=HERE / "captures")
    parser.add_argument(
        "--stop-on-failure",
        action="store_true",
        help="Stop immediately after a sustained cross-environment RGB mismatch.",
    )
    args = parser.parse_args(argv)
    if args.num_envs < 2:
        parser.error("--num-envs must be at least 2 for differential detection")
    if args.steps < 1:
        parser.error("--steps must be positive")
    if args.reset_every < 0:
        parser.error("--reset-every cannot be negative")
    if args.consecutive_failure_frames < 1:
        parser.error("--consecutive-failure-frames must be positive")
    return args


def make_env_cfg(args: argparse.Namespace) -> ManagerBasedEnvCfg:
    """Build a small upstream-only Isaac Lab environment config."""

    renderer_kwargs = {"log_file_path": str(args.output_dir / "ovrtx.log")}
    if "use_ovrtx_cloning" in inspect.signature(OVRTXRendererCfg).parameters:
        renderer_kwargs["use_ovrtx_cloning"] = bool(args.ovrtx_cloning)
    elif args.ovrtx_cloning:
        raise RuntimeError(
            "--ovrtx-cloning is not configurable in this Isaac Lab version; "
            "the renderer uses the scene clone plan automatically."
        )
    renderer_cfg = OVRTXRendererCfg(**renderer_kwargs)
    visualizer_cfgs = []
    if args.viewer == "newton":
        visualizer_cfgs = [
            NewtonVisualizerCfg(
                headless=bool(args.headless),
                eye=(6.0, -6.0, 4.0),
                lookat=(0.0, 0.0, 1.2),
                tiled_cam_view=True,
                tiled_cam_num=args.num_envs,
                tiled_cam_env_indices=list(range(args.num_envs)),
                tiled_cam_prim_path=VIEWER_CAMERA_PRIM_PATH,
                randomly_sample_visible_envs=False,
            )
        ]

    robot_cfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=UsdFileCfg(
            usd_path=str(ROBOT_USD),
            variants={"Physics": "physics"},
            articulation_props=NewtonArticulationRootPropertiesCfg(self_collision_enabled=False),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            joint_pos=INITIAL_JOINT_POSITIONS,
            joint_vel={".*": 0.0},
        ),
        actuators=make_robot_actuators(),
    )
    camera_cfg = CameraCfg(
        prim_path=CAMERA_PRIM_PATH,
        update_period=0.0,
        update_latest_camera_pose=True,
        data_types=["rgb"],
        width=args.width,
        height=args.height,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            f_stop=0.0,
            horizontal_aperture=20.955,
            clipping_range=(0.05, 15.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(2.4, -2.4, 2.0),
            rot=(
                0.7322313785552979,
                0.3033001124858856,
                -0.23335732519626617,
                -0.5633744597434998,
            ),
            convention="ros",
        ),
        renderer_cfg=renderer_cfg,
    )

    @configclass
    class ReproSceneCfg(InteractiveSceneCfg):
        robot = robot_cfg
        camera = camera_cfg
        backdrop = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Backdrop",
            spawn=UsdFileCfg(usd_path=str(BACKDROP_USD)),
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=(-1.2, 1.2, 2.0),
                rot=(0.0, 0.0, 0.9238795325112867, 0.38268343236508984),
            ),
        )
        light = AssetBaseCfg(
            prim_path="/World/Light",
            spawn=sim_utils.DomeLightCfg(
                intensity=1500.0,
                color=(1.0, 1.0, 1.0),
            ),
        )

    @configclass
    class ReproEnvCfg(ManagerBasedEnvCfg):
        scene: ReproSceneCfg = ReproSceneCfg(
            num_envs=args.num_envs,
            env_spacing=4.0,
            replicate_physics=True,
            lazy_sensor_update=False,
        )
        actions: EmptyManagerCfg = EmptyManagerCfg()
        observations: EmptyManagerCfg = EmptyManagerCfg()
        sim: SimulationCfg = SimulationCfg(
            dt=1.0 / 100.0,
            render_interval=1,
            device=args.device,
            gravity=(0.0, 0.0, 0.0),
            use_newton_actuators=True,
            physics_material=NewtonMaterialPropertiesCfg(
                static_friction=0.8,
                dynamic_friction=0.6,
                restitution=0.0,
            ),
            physics=NewtonCfg(
                solver_cfg=MJWarpSolverCfg(
                    integrator="implicitfast",
                    cone="elliptic",
                    njmax=1024,
                    nconmax=512,
                    iterations=100,
                    ls_iterations=15,
                    impratio=10.0,
                    use_mujoco_contacts=False,
                    ccd_iterations=35,
                ),
                collision_cfg=NewtonCollisionPipelineCfg(),
                default_shape_cfg=NewtonShapeCfg(),
                num_substeps=5,
            ),
            visualizer_cfgs=visualizer_cfgs,
        )

        def __post_init__(self) -> None:
            self.decimation = 1
            self.num_rerenders_on_reset = 1
            self.seed = args.seed

    return ReproEnvCfg()


def package_version(name: str) -> str:
    """Return one installed package version without failing the reproducer."""

    try:
        return version(name)
    except PackageNotFoundError:
        return "not-installed"


def write_run_info(args: argparse.Namespace) -> None:
    """Record versions, hardware, arguments, and the exact robot asset hash."""

    args.output_dir.mkdir(parents=True, exist_ok=True)
    adapter_source_path = inspect.getsourcefile(ovrtx_renderer_module)
    adapter_source = ""
    if adapter_source_path is not None:
        adapter_source = Path(adapter_source_path).read_text(encoding="utf-8")
    read_gpu_transforms_env = "ISAAC_LAB_OVRTX_READ_GPU_TRANSFORMS"
    adapter_honors_env = read_gpu_transforms_env in adapter_source
    if os.environ.get(read_gpu_transforms_env) is not None and not adapter_honors_env:
        print(
            f"[warning] {read_gpu_transforms_env} is set, but the installed "
            "Isaac Lab OVRTX adapter does not reference it; the value has no effect."
        )
    info = {
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "packages": {
            name: package_version(name)
            for name in (
                "isaaclab",
                "isaaclab-newton",
                "isaaclab-ov",
                "isaaclab-visualizers",
                "newton",
                "ovrtx",
                "torch",
                "warp-lang",
            )
        },
        "platform": platform.platform(),
        "python": sys.version,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(args.device) if torch.cuda.is_available() else None,
        "robot_usd": str(ROBOT_USD),
        "robot_usd_sha256": hashlib.sha256(ROBOT_USD.read_bytes()).hexdigest(),
        "ovrtx_adapter": {
            "source_path": adapter_source_path,
            "read_gpu_transforms_env": os.environ.get(read_gpu_transforms_env),
            "honors_read_gpu_transforms_env": adapter_honors_env,
        },
    }
    path = args.output_dir / "run_info.json"
    path.write_text(json.dumps(info, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(info, indent=2, sort_keys=True))


def reset_robot(env: ManagerBasedEnv) -> None:
    """Restore identical root and joint state in every cloned environment."""

    robot = env.scene["robot"]
    root_pose = robot.data.default_root_pose.torch.clone()
    root_pose[:, :3] += env.scene.env_origins
    robot.write_root_pose_to_sim_index(root_pose=root_pose)
    robot.write_root_velocity_to_sim_index(root_velocity=robot.data.default_root_vel.torch.clone())
    robot.write_joint_position_to_sim_index(position=robot.data.default_joint_pos.torch.clone())
    robot.write_joint_velocity_to_sim_index(velocity=robot.data.default_joint_vel.torch.clone())
    robot.reset()
    env.scene.write_data_to_sim()
    env.sim.forward()


def rgb_mismatch(
    rgb: torch.Tensor,
    *,
    rgb_threshold: int,
) -> tuple[int, int, int, torch.Tensor]:
    """Compare denoised robot silhouettes with environment zero."""

    rgb = rgb[..., :3]
    if rgb.dtype.is_floating_point:
        rgb = (rgb.clamp(0.0, 1.0) * 255.0).to(torch.int16)
    else:
        rgb = rgb.to(torch.int16)
    masks = foreground_masks(rgb, background_threshold=rgb_threshold)
    reference = masks[0]
    worst_env = 1
    worst_bad_pixels = 0
    worst_channel_error = 0
    worst_bad_mask = torch.zeros_like(reference)
    for env_index in range(1, rgb.shape[0]):
        difference = torch.abs(rgb[0] - rgb[env_index])
        pixel_error = torch.max(difference, dim=-1).values
        bad_mask = torch.logical_xor(reference, masks[env_index])
        bad_pixels = int(torch.count_nonzero(bad_mask).item())
        max_channel_error = int(torch.max(pixel_error).item())
        if bad_pixels > worst_bad_pixels:
            worst_env = env_index
            worst_bad_pixels = bad_pixels
            worst_channel_error = max_channel_error
            worst_bad_mask = bad_mask
    return worst_env, worst_bad_pixels, worst_channel_error, worst_bad_mask


def foreground_masks(
    rgb: torch.Tensor,
    *,
    background_threshold: int,
) -> torch.Tensor:
    """Extract closed foreground masks against the uniform emissive backdrop."""

    rgb = rgb[..., :3]
    if rgb.dtype.is_floating_point:
        rgb = (rgb.clamp(0.0, 1.0) * 255.0).to(torch.int16)
    else:
        rgb = rgb.to(torch.int16)
    corner = rgb[:, :32, :32].reshape(rgb.shape[0], -1, 3)
    background = torch.median(corner, dim=1).values[:, None, None, :]
    color_distance = torch.max(torch.abs(rgb - background), dim=-1).values
    masks = (color_distance > background_threshold).to(torch.float32).unsqueeze(1)
    dilated = torch_functional.max_pool2d(masks, kernel_size=7, stride=1, padding=3)
    closed = 1.0 - torch_functional.max_pool2d(
        1.0 - dilated,
        kernel_size=7,
        stride=1,
        padding=3,
    )
    return closed[:, 0] > 0.5


def rgb_to_image(rgb: torch.Tensor) -> Image.Image:
    """Convert one Isaac Lab RGB tensor to a Pillow image."""

    array = rgb.detach().cpu()
    if array.dtype.is_floating_point:
        array = (array.clamp(0.0, 1.0) * 255.0).to(torch.uint8)
    else:
        array = array.to(torch.uint8)
    return Image.fromarray(array[..., :3].numpy(), mode="RGB")


def save_newton_viewer_frame(env: ManagerBasedEnv, path: Path) -> None:
    """Save the Newton viewer framebuffer when the viewer is configured."""

    if not env.sim.visualizers:
        return
    viewer = getattr(env.sim.visualizers[0], "_viewer", None)
    if viewer is None:
        return

    frame = viewer.get_frame()
    if torch.is_tensor(frame):
        frame = frame.detach().cpu().numpy()
    else:
        try:
            import warp as wp

            if isinstance(frame, wp.array):
                frame = wp.to_torch(frame).detach().cpu().numpy()
        except (ImportError, TypeError):
            pass

    import numpy as np

    array = np.asarray(frame)
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=-1)
    array = array[..., :3]
    if np.issubdtype(array.dtype, np.floating):
        if array.size and float(np.nanmax(array)) <= 1.0 + 1.0e-6:
            array = array * 255.0
        array = np.nan_to_num(array, nan=0.0, posinf=255.0, neginf=0.0)
    array = np.clip(array, 0, 255).astype(np.uint8)
    Image.fromarray(array, mode="RGB").save(path)


def invisible_robot_prims(stage: Any) -> list[str]:
    """Return authored/computed invisible prims under the cloned robots."""

    from pxr import UsdGeom

    invisible = []
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if not path.startswith("/World/envs/env_") or "/Robot/" not in path:
            continue
        imageable = UsdGeom.Imageable(prim)
        if imageable and imageable.ComputeVisibility() == UsdGeom.Tokens.invisible:
            invisible.append(path)
    return invisible


def simulation_state_metrics(env: ManagerBasedEnv) -> dict[str, float | bool]:
    """Measure whether all Newton articulation states remain synchronized."""

    robot = env.scene["robot"]
    origins = env.scene.env_origins[:, None, :]
    body_positions = robot.data.body_pos_w.torch - origins
    body_quaternions = robot.data.body_quat_w.torch
    joint_positions = robot.data.joint_pos.torch
    all_finite = bool(
        torch.all(torch.isfinite(body_positions))
        and torch.all(torch.isfinite(body_quaternions))
        and torch.all(torch.isfinite(joint_positions))
    )
    body_position_delta = torch.max(torch.abs(body_positions[1:] - body_positions[0:1]))
    joint_position_delta = torch.max(torch.abs(joint_positions[1:] - joint_positions[0:1]))
    quaternion_abs_dot = torch.abs(torch.sum(body_quaternions[1:] * body_quaternions[0:1], dim=-1))
    return {
        "all_finite": all_finite,
        "max_body_position_delta_m": float(body_position_delta.item()),
        "max_joint_position_delta_rad": float(joint_position_delta.item()),
        "min_body_quaternion_abs_dot": float(torch.min(quaternion_abs_dot).item()),
    }


def simulation_state_is_synchronized(metrics: dict[str, float | bool]) -> bool:
    """Return whether a render mismatch can be isolated from Newton state drift."""

    return bool(
        metrics["all_finite"]
        and metrics["max_body_position_delta_m"] < 1.0e-4
        and metrics["max_joint_position_delta_rad"] < 1.0e-4
        and metrics["min_body_quaternion_abs_dot"] > 0.9999
    )


def save_capture(
    env: ManagerBasedEnv,
    *,
    output_dir: Path,
    label: str,
    step: int,
    rgb: torch.Tensor,
    background_threshold: int,
    detection: dict[str, int | float] | None = None,
) -> Path:
    """Save renderer output, Newton viewer, and simulation-state evidence."""

    capture_dir = output_dir / f"{label}_step_{step:06d}"
    capture_dir.mkdir(parents=True, exist_ok=True)
    masks = foreground_masks(rgb, background_threshold=background_threshold)
    for env_index in range(rgb.shape[0]):
        rgb_to_image(rgb[env_index]).save(capture_dir / f"rgb_env_{env_index}.png")
        mask = (masks[env_index].to(torch.uint8) * 255).detach().cpu().numpy()
        Image.fromarray(mask, mode="L").save(capture_dir / f"silhouette_env_{env_index}.png")
    save_newton_viewer_frame(env, capture_dir / "newton_viewer.png")
    visibility = {"invisible_robot_prims": invisible_robot_prims(env.sim.stage)}
    (capture_dir / "stage_visibility.json").write_text(
        json.dumps(visibility, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    camera = env.scene["camera"]
    camera_positions = camera.data.pos_w.torch.detach().cpu()
    relative_positions = camera_positions - env.scene.env_origins.detach().cpu()
    camera_state = {
        "world_positions": camera_positions.tolist(),
        "env_relative_positions": relative_positions.tolist(),
        "quaternions_ros": camera.data.quat_w_ros.torch.detach().cpu().tolist(),
    }
    (capture_dir / "camera_state.json").write_text(
        json.dumps(camera_state, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    robot = env.scene["robot"]
    robot_state = {
        "metrics": simulation_state_metrics(env),
        "body_names": list(robot.body_names),
        "joint_names": list(robot.joint_names),
        "env_relative_body_positions": (
            robot.data.body_pos_w.torch.detach().cpu() - env.scene.env_origins.detach().cpu()[:, None, :]
        ).tolist(),
        "body_quaternions_wxyz": robot.data.body_quat_w.torch.detach().cpu().tolist(),
        "joint_positions": robot.data.joint_pos.torch.detach().cpu().tolist(),
    }
    (capture_dir / "robot_state.json").write_text(
        json.dumps(robot_state, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if detection is not None:
        (capture_dir / "detection.json").write_text(
            json.dumps(detection, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(f"[capture] {capture_dir}")
    return capture_dir


def viewer_is_open(env: ManagerBasedEnv) -> bool:
    """Return whether at least one configured viewer remains open."""

    if not env.sim.visualizers:
        return True
    return any(visualizer.is_running() and not visualizer.is_closed for visualizer in env.sim.visualizers)


def run(args: argparse.Namespace, env_cfg: ManagerBasedEnvCfg) -> bool:
    """Run the synchronized pose sweep and return whether it reproduced."""

    env = ManagerBasedEnv(cfg=env_cfg)
    reproduced = False
    last_rgb: torch.Tensor | None = None
    last_step = 0
    suspected_env: int | None = None
    mismatch_window: deque[torch.Tensor] = deque(maxlen=args.consecutive_failure_frames)
    try:
        env.reset(seed=args.seed)
        reset_robot(env)
        robot = env.scene["robot"]
        camera = env.scene["camera"]
        joint_ids, joint_names = robot.find_joints(MOTION_JOINTS, preserve_order=True)
        if joint_names != MOTION_JOINTS:
            raise RuntimeError(f"Robot joint mismatch: expected {MOTION_JOINTS}, got {joint_names}")
        center = robot.data.default_joint_pos.torch[:, joint_ids].clone()
        root_pose_center = robot.data.default_root_pose.torch.clone()
        root_pose_center[:, :3] += env.scene.env_origins
        zero_root_velocity = torch.zeros_like(robot.data.default_root_vel.torch)
        phase = torch.linspace(
            0.0,
            math.pi,
            len(joint_ids),
            device=env.device,
        ).unsqueeze(0)
        actions = torch.zeros_like(env.action_manager.action)

        print(
            "[run] Newton/MJWarp + OVRTX + "
            f"viewer={args.viewer}, num_envs={args.num_envs}, "
            f"ovrtx_cloning={args.ovrtx_cloning}"
        )
        for step in range(args.steps):
            if not viewer_is_open(env):
                print("[run] Viewer closed by user.")
                break
            if args.reset_every and step and step % args.reset_every == 0:
                env.reset()
                reset_robot(env)
                suspected_env = None
                mismatch_window.clear()
                print(f"[reset] step={step}")

            time_value = step * env.step_dt
            root_pose = root_pose_center.clone()
            root_pose[:, 0] += 0.60 * math.sin(0.35 * time_value)
            root_pose[:, 1] += 0.35 * math.sin(0.51 * time_value)
            robot.write_root_pose_to_sim_index(root_pose=root_pose)
            robot.write_root_velocity_to_sim_index(root_velocity=zero_root_velocity)
            target = center + 0.28 * torch.sin(1.7 * time_value + phase)
            robot.set_joint_position_target_index(
                target=target,
                joint_ids=joint_ids,
            )
            with torch.inference_mode():
                env.step(actions)

            rgb = camera.data.output["rgb"].torch
            last_rgb = rgb.clone()
            last_step = step
            rgb_env, bad_rgb_pixels, max_rgb_error, bad_rgb_mask = rgb_mismatch(
                rgb,
                rgb_threshold=args.rgb_threshold,
            )
            state_metrics = simulation_state_metrics(env)
            state_is_synchronized = simulation_state_is_synchronized(state_metrics)
            if step >= 5 and state_is_synchronized:
                if suspected_env != rgb_env:
                    suspected_env = rgb_env
                    mismatch_window.clear()
                mismatch_window.append(bad_rgb_mask)
            else:
                suspected_env = None
                mismatch_window.clear()
            persistent_bad_pixels = 0
            if len(mismatch_window) == args.consecutive_failure_frames:
                persistent_mask = torch.stack(tuple(mismatch_window)).all(dim=0)
                persistent_bad_pixels = int(torch.count_nonzero(persistent_mask).item())
            if step % 100 == 0:
                print(
                    f"[step {step:05d}] rgb_env={rgb_env} "
                    f"bad_rgb_pixels={bad_rgb_pixels} max_rgb_error={max_rgb_error} "
                    f"persistent_bad_rgb_pixels={persistent_bad_pixels} "
                    f"state={state_metrics}"
                )
            if not reproduced and persistent_bad_pixels >= args.bad_rgb_pixel_threshold:
                reproduced = True
                print(
                    f"[REPRODUCED] step={step} rgb_env={rgb_env} "
                    f"bad_rgb_pixels={bad_rgb_pixels} max_rgb_error={max_rgb_error} "
                    f"persistent_bad_rgb_pixels={persistent_bad_pixels} "
                    f"state={state_metrics}"
                )
                save_capture(
                    env,
                    output_dir=args.output_dir,
                    label="failure",
                    step=step,
                    rgb=rgb,
                    background_threshold=args.rgb_threshold,
                    detection={
                        "affected_env": rgb_env,
                        "bad_silhouette_pixels": bad_rgb_pixels,
                        "persistent_bad_silhouette_pixels": persistent_bad_pixels,
                        "max_rgb_channel_error": max_rgb_error,
                        "persistence_frames": args.consecutive_failure_frames,
                    },
                )
                if args.stop_on_failure:
                    break

        if last_rgb is not None:
            save_capture(
                env,
                output_dir=args.output_dir,
                label="last",
                step=last_step,
                rgb=last_rgb,
                background_threshold=args.rgb_threshold,
            )
    finally:
        env.close()
    return reproduced


def main(argv: list[str] | None = None) -> int:
    """Run the standalone reproducer."""

    args = parse_args(argv)
    if not ROBOT_USD.is_file():
        raise FileNotFoundError(f"Bundled robot asset is missing: {ROBOT_USD}")
    write_run_info(args)
    env_cfg = make_env_cfg(args)
    visualizer = ["newton"] if args.viewer == "newton" else None
    launcher_args = {
        "headless": bool(args.headless),
        "visualizer": visualizer,
        "visualizer_explicit": True,
    }
    with launch_simulation(env_cfg, launcher_args):
        reproduced = run(args, env_cfg)
    print(f"[result] reproduced={reproduced}")
    return 2 if reproduced else 0


if __name__ == "__main__":
    raise SystemExit(main())
