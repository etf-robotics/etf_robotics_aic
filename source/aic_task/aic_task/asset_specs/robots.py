"""Concrete robot asset specs."""

from __future__ import annotations

from dataclasses import dataclass

from .base import (
    AssetIdentity,
    AssetSpec,
    BodyRoleSpec,
    JointGroupSpec,
    NamedBodySpec,
    NamedFrameSpec,
    UsdAssetInterface,
    asset_path,
)


ROBOT_EEF = "eef"
ROBOT_TCP = "tcp"
ROBOT_PLUG_CENTER = "plug_center"


@dataclass(frozen=True)
class RobotAssetSpec(AssetSpec):
    """Asset-level contract for a robot articulation."""

    joint_groups: tuple[JointGroupSpec, ...]
    bodies: tuple[NamedBodySpec, ...]
    frames: tuple[NamedFrameSpec, ...]
    body_roles: tuple[BodyRoleSpec, ...] = ()

    def joint_group(self, name: str) -> JointGroupSpec:
        """Return a named joint group."""
        for group in self.joint_groups:
            if group.name == name:
                return group
        raise KeyError(f"Unknown joint group '{name}' for asset '{self.identity.name}'.")

    def body(self, name: str) -> NamedBodySpec:
        """Return a named body contract."""
        for body in self.bodies:
            if body.name == name:
                return body
        raise KeyError(f"Unknown body '{name}' for asset '{self.identity.name}'.")

    def body_role(self, name: str) -> BodyRoleSpec:
        """Return the concrete body currently assigned to a semantic role."""
        for role in self.body_roles:
            if role.name == name:
                return role
        raise KeyError(f"Unknown body role '{name}' for asset '{self.identity.name}'.")

    def body_name_for_role(self, name: str) -> str:
        """Return only the concrete body name for a semantic role."""
        return self.body_role(name).body_name

    def frame(self, name: str) -> NamedFrameSpec:
        """Return a named frame contract."""
        for frame in self.frames:
            if frame.name == name:
                return frame
        raise KeyError(f"Unknown frame '{name}' for asset '{self.identity.name}'.")


UR5E_ARM = JointGroupSpec(
    name="arm",
    joint_names=(
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    ),
    default_positions={
        "shoulder_pan_joint": 0.1597,
        "shoulder_lift_joint": -1.3542,
        "elbow_joint": -1.6648,
        "wrist_1_joint": -1.6933,
        "wrist_2_joint": 1.5710,
        "wrist_3_joint": 1.4110,
    },
)

UR5E_CABLE_ASSET = RobotAssetSpec(
    identity=AssetIdentity(name="ur5e_cable", role="robot"),
    usd=UsdAssetInterface(kind="articulation", root_prim="aic_unified_robot"),
    usd_path=asset_path("robots", "ur5e_cable", "aic_unified_robot_cable_sdf.usd"),
    joint_groups=(UR5E_ARM,),
    bodies=(
        NamedBodySpec(name="gripper_tcp", purpose="tool center point used by task controllers"),
        NamedBodySpec(name="wrist_3_link", purpose="terminal wrist body for IK diagnostics"),
        NamedBodySpec(name="sfp_module_link", purpose="rigid body at the SFP module center"),
        NamedBodySpec(name="sfp_tip_link", purpose="physical insertion tip carried by the robot"),
        NamedBodySpec(name="sfp_module_visual", purpose="inserted SFP module body"),
    ),
    body_roles=(
        BodyRoleSpec(
            name=ROBOT_EEF,
            body_name="sfp_tip_link",
            purpose="task end-effector used for insertion target pose and success checks",
        ),
        BodyRoleSpec(
            name=ROBOT_TCP,
            body_name="gripper_tcp",
            purpose="body controlled by the Differential IK action",
        ),
        BodyRoleSpec(
            name=ROBOT_PLUG_CENTER,
            body_name="sfp_module_link",
            purpose="module center body used with the eef body to define the plug axis",
        ),
    ),
    frames=(
        NamedFrameSpec(
            name="center_camera",
            path="aic_unified_robot/center_camera_optical/center_camera",
            purpose="center wrist camera optical frame",
        ),
        NamedFrameSpec(
            name="left_camera",
            path="aic_unified_robot/left_camera_optical/left_camera",
            purpose="left wrist camera optical frame",
        ),
        NamedFrameSpec(
            name="right_camera",
            path="aic_unified_robot/right_camera_optical/right_camera",
            purpose="right wrist camera optical frame",
        ),
    ),
)
