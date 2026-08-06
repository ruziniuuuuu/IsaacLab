# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for Newton control-array bindings used by articulation data."""

from __future__ import annotations

from unittest.mock import patch

import warp as wp
from isaaclab_newton.actuators.physx_wrapper import PhysxActuatorWrapper
from isaaclab_newton.assets.articulation.articulation_data import ArticulationData
from isaaclab_newton.physics import NewtonManager as SimulationManager


class _RecordingArticulationView:
    """Minimal view that records attribute names requested by a binding refresh."""

    count = 1
    joint_dof_count = 1
    link_count = 1
    tendon_count = 0
    is_fixed_base = True

    def __init__(self) -> None:
        self.requested_attributes: list[str] = []

    def get_attribute(self, name: str, source: object) -> wp.array:
        del source
        self.requested_attributes.append(name)
        if name in {"body_inertia", "body_inv_inertia"}:
            return wp.zeros((1, 1, 1), dtype=wp.mat33f, device="cpu")
        if name in {"body_f", "body_parent_f"}:
            return wp.zeros((1, 1, 1), dtype=wp.spatial_vectorf, device="cpu")
        return wp.zeros((1, 1, 1), dtype=wp.float32, device="cpu")

    def get_root_transforms(self, source: object) -> wp.array:
        del source
        return wp.zeros((1, 1), dtype=wp.transformf, device="cpu")

    def get_root_velocities(self, source: object) -> None:
        del source
        return

    def get_link_transforms(self, source: object) -> wp.array:
        del source
        return wp.zeros((1, 1, 1), dtype=wp.transformf, device="cpu")

    def get_link_velocities(self, source: object) -> wp.array:
        del source
        return wp.zeros((1, 1, 1), dtype=wp.spatial_vectorf, device="cpu")

    def get_dof_positions(self, source: object) -> wp.array:
        del source
        return wp.zeros((1, 1, 1), dtype=wp.float32, device="cpu")

    def get_dof_velocities(self, source: object) -> wp.array:
        del source
        return wp.zeros((1, 1, 1), dtype=wp.float32, device="cpu")


def test_articulation_data_uses_canonical_newton_control_targets() -> None:
    """Binding refresh requests the Newton 1.5 control target names."""
    view = _RecordingArticulationView()
    data = object.__new__(ArticulationData)
    data._root_view = view
    data._read_launch_cache = {}
    data.device = "cpu"

    sentinel = object()
    with (
        patch.object(SimulationManager, "get_model", return_value=sentinel),
        patch.object(SimulationManager, "get_state_0", return_value=sentinel),
        patch.object(SimulationManager, "get_control", return_value=sentinel),
    ):
        data._create_simulation_bindings()

    assert "joint_target_q" in view.requested_attributes
    assert "joint_target_qd" in view.requested_attributes
    assert "joint_target_pos" not in view.requested_attributes
    assert "joint_target_vel" not in view.requested_attributes


def test_physx_actuator_wrapper_matches_canonical_newton_control_schema() -> None:
    """The PhysX bridge exposes the same target fields as Newton Control."""
    wrapper = PhysxActuatorWrapper()

    assert hasattr(wrapper, "joint_target_q")
    assert hasattr(wrapper, "joint_target_qd")
    assert not hasattr(wrapper, "joint_target_pos")
    assert not hasattr(wrapper, "joint_target_vel")
