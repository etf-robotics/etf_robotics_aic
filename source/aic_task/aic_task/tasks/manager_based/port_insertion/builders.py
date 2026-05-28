"""Build IsaacLab cfg objects from port-insertion assembly specs."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp import DifferentialInverseKinematicsActionCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass

from aic_task.asset_specs import (
    AssetSpec,
    CameraFrameSpec,
    RobotAssetSpec,
    SceneSlotSpec,
    TargetPortSpec,
)

from .mdp.commands import InsertionGoalCommandCfg
from .specs import PortInsertionAssemblySpec


def build_scene_cfg(
    assembly: PortInsertionAssemblySpec,
    *,
    num_envs: int = 1,
    env_spacing: float = 4.0,
    replicate_physics: bool = False,
    filter_collisions: bool = False,
) -> InteractiveSceneCfg:
    """Build the scene cfg for the selected robot, target, and layout."""

    assembly.validate()

    @configclass
    class PortInsertionSceneCfg(InteractiveSceneCfg):
        pass

    scene = PortInsertionSceneCfg(
        num_envs=num_envs,
        env_spacing=env_spacing,
        replicate_physics=replicate_physics,
        filter_collisions=filter_collisions,
    )
    scene.ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    )
    scene.light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
    )

    setattr(scene, assembly.layout.robot_slot.name, _build_robot_cfg(assembly.layout.robot_slot, assembly.robot))
    for slot in _non_robot_slots(assembly):
        setattr(scene, slot.name, _build_asset_cfg(slot))
    for camera in assembly.robot.camera_frames:
        setattr(scene, camera.name, _build_camera_cfg(assembly.layout.robot_slot, camera))

    return scene


def build_action_cfg(assembly: PortInsertionAssemblySpec) -> dict[str, DifferentialInverseKinematicsActionCfg]:
    """Build action terms from ``ControllerSpec`` and ``RobotAssetSpec``."""

    assembly.validate()
    controller = assembly.controller
    if controller.action_type != "diff_ik":
        raise NotImplementedError(f"Unsupported controller action type: {controller.action_type}")

    joint_group = assembly.robot.joint_group(controller.joint_group)
    body_name = assembly.robot.body_name_for_role(controller.controlled_body_role)
    action_cfg = DifferentialInverseKinematicsActionCfg(
        asset_name=controller.robot_slot,
        joint_names=list(joint_group.joint_names),
        body_name=body_name,
        controller=DifferentialIKControllerCfg(
            command_type=controller.command_type,
            use_relative_mode=controller.use_relative_mode,
            ik_method=controller.ik_method,
            ik_params=dict(controller.ik_params),
        ),
        scale=controller.scale,
    )
    return {"arm_action": action_cfg}


def build_command_cfg(assembly: PortInsertionAssemblySpec) -> dict[str, InsertionGoalCommandCfg]:
    """Build the episode-level insertion goal command from target/port specs."""

    assembly.validate()
    goal = assembly.goal
    target = assembly.target
    port = assembly.selected_port()
    command_cfg = InsertionGoalCommandCfg(
        target_scene_name=goal.target_slot,
        target_root_prim=target.usd.root_prim,
        port_name=port.name,
        port_index=port.index,
        port_link_path=port.link_path,
        port_seat_frame_path=port.seat_frame_path,
        port_entrance_frame_path=port.entrance_frame_path,
        insertion_axis_local=port.insertion_axis_local,
        target_xz_offset=goal.target_xz_offset,
        approach_offset_local=goal.approach_offset_local,
        approach_pos_noise_local=goal.approach_pos_noise_local,
        approach_tilt_noise_deg=goal.approach_tilt_noise_deg,
        approach_twist_noise_deg=goal.approach_twist_noise_deg,
        resampling_time_range=goal.resampling_time_range,
        debug_vis=goal.debug_vis,
    )
    _validate_command_cfg(assembly, command_cfg, port)
    return {goal.command_name: command_cfg}


def _build_robot_cfg(slot: SceneSlotSpec, robot: RobotAssetSpec) -> ArticulationCfg:
    spawn = robot.spawn
    return ArticulationCfg(
        prim_path=slot.prim_path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=robot.usd_path,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                max_depenetration_velocity=spawn.max_depenetration_velocity,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=spawn.enabled_self_collisions,
                solver_position_iteration_count=spawn.solver_position_iteration_count,
                solver_velocity_iteration_count=spawn.solver_velocity_iteration_count,
            ),
            activate_contact_sensors=spawn.activate_contact_sensors,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=slot.pose.pos,
            rot=slot.pose.rot,
            joint_pos=_robot_default_joint_positions(robot),
        ),
        actuators=_robot_actuator_cfgs(robot),
    )


def _build_asset_cfg(slot: SceneSlotSpec) -> AssetBaseCfg | RigidObjectCfg:
    spawn = _usd_file_cfg(slot.asset, kinematic_enabled=slot.kinematic)
    if slot.asset.usd.kind == "static":
        return AssetBaseCfg(
            prim_path=slot.prim_path,
            spawn=spawn,
            init_state=AssetBaseCfg.InitialStateCfg(pos=slot.pose.pos, rot=slot.pose.rot),
        )
    if slot.asset.usd.kind == "rigid_object":
        return RigidObjectCfg(
            prim_path=slot.prim_path,
            spawn=spawn,
            init_state=RigidObjectCfg.InitialStateCfg(pos=slot.pose.pos, rot=slot.pose.rot),
        )
    raise ValueError(f"Scene slot '{slot.name}' is not a passive asset: {slot.asset.usd.kind}")


def _usd_file_cfg(asset: AssetSpec, *, kinematic_enabled: bool) -> sim_utils.UsdFileCfg:
    rigid_props = None
    if asset.usd.kind == "rigid_object":
        rigid_props = sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=kinematic_enabled)
    return sim_utils.UsdFileCfg(usd_path=asset.usd_path, rigid_props=rigid_props)


def _build_camera_cfg(robot_slot: SceneSlotSpec, camera: CameraFrameSpec) -> TiledCameraCfg:
    return TiledCameraCfg(
        prim_path=f"{robot_slot.prim_path}/{camera.relative_prim_path.lstrip('/')}",
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=22.48,
            focus_distance=0.0,
            horizontal_aperture=20.955,
            vertical_aperture=18.627,
            clipping_range=(0.07, 20.0),
        ),
        height=camera.height,
        width=camera.width,
        data_types=list(camera.data_types),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
            convention="ros",
        ),
    )


def _robot_default_joint_positions(robot: RobotAssetSpec) -> dict[str, float]:
    joint_pos: dict[str, float] = {}
    for group in robot.joint_groups:
        joint_pos.update(group.default_positions)
    return joint_pos


def _robot_actuator_cfgs(robot: RobotAssetSpec) -> dict[str, ImplicitActuatorCfg]:
    cfgs: dict[str, ImplicitActuatorCfg] = {}
    for actuator in robot.actuators:
        joint_group = robot.joint_group(actuator.joint_group)
        cfgs[actuator.name] = ImplicitActuatorCfg(
            joint_names_expr=list(joint_group.joint_names),
            effort_limit_sim=actuator.effort_limit_sim,
            stiffness=actuator.stiffness,
            damping=actuator.damping,
        )
    return cfgs


def _non_robot_slots(assembly: PortInsertionAssemblySpec) -> tuple[SceneSlotSpec, ...]:
    return tuple(slot for slot in assembly.layout.all_slots() if slot.name != assembly.layout.robot_slot.name)


def _validate_command_cfg(
    assembly: PortInsertionAssemblySpec,
    command_cfg: InsertionGoalCommandCfg,
    port: TargetPortSpec,
) -> None:
    if command_cfg.target_scene_name != assembly.layout.target_slot.name:
        raise ValueError(
            f"Command targets scene slot '{command_cfg.target_scene_name}', "
            f"but layout target slot is '{assembly.layout.target_slot.name}'."
        )
    if command_cfg.target_root_prim != assembly.target.usd.root_prim:
        raise ValueError("Command target root prim does not match the selected target asset.")
    if command_cfg.port_name != port.name or command_cfg.port_index != port.index:
        raise ValueError("Command port identity does not match the selected target port.")
    if not command_cfg.port_seat_frame_path:
        raise ValueError("Command port seat frame path cannot be empty.")
    if not any(abs(value) > 0.0 for value in command_cfg.insertion_axis_local):
        raise ValueError("Command insertion axis cannot be the zero vector.")


__all__ = [
    "build_action_cfg",
    "build_command_cfg",
    "build_scene_cfg",
]
