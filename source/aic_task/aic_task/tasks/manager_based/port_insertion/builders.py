"""Build IsaacLab cfg objects from port-insertion assembly specs."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp import (
    DifferentialInverseKinematicsActionCfg,
    image,
    joint_pos_rel,
    joint_vel_rel,
    last_action,
    reset_joints_by_offset,
    time_out,
)
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass

from aic_task.asset_specs import (
    AssetSpec,
    CameraFrameSpec,
    ROBOT_ROLE_EEF,
    RobotAssetSpec,
    SceneLayoutSpec,
    SceneSlotSpec,
    TargetPortSpec,
)

from .mdp.commands import InsertionGoalCommandCfg
from .mdp.events import randomize_board_and_parts, randomize_dome_light
from .mdp.observations import (
    body_ang_vel_b,
    body_incoming_wrench,
    body_lin_vel_b,
    body_pos_b,
    body_quat_b,
    entrance_pos_b,
    entrance_quat_b,
    insertion_fraction,
    seat_pos_b,
    seat_quat_b,
)
from .mdp.terminations import InsertionGoalReachedSuccess, InsertionGoalStationaryFailure
from .specs import PortInsertionAssemblySpec


_GROUND_SCENE_NAME = "ground"
_LIGHT_SCENE_NAME = "light"


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
    _validate_scene_slots(assembly)

    @configclass
    class PortInsertionSceneCfg(InteractiveSceneCfg):
        pass

    scene = PortInsertionSceneCfg(
        num_envs=num_envs,
        env_spacing=env_spacing,
        replicate_physics=replicate_physics,
        filter_collisions=filter_collisions,
    )
    setattr(
        scene,
        _GROUND_SCENE_NAME,
        AssetBaseCfg(
            prim_path="/World/ground",
            spawn=sim_utils.GroundPlaneCfg(),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
        ),
    )
    setattr(
        scene,
        _LIGHT_SCENE_NAME,
        AssetBaseCfg(
            prim_path="/World/light",
            spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
        ),
    )

    setattr(scene, assembly.layout.robot_slot.name, _build_robot_cfg(assembly.layout.robot_slot, assembly.robot))
    for slot in _support_slots(assembly.layout):
        setattr(scene, slot.name, _build_asset_cfg(slot, slot.asset))
    setattr(scene, assembly.layout.target_slot.name, _build_asset_cfg(assembly.layout.target_slot, assembly.target))
    for camera in assembly.robot.camera_frames:
        setattr(scene, camera.name, _build_camera_cfg(assembly.layout.robot_slot, camera))

    return scene


def build_action_cfg(assembly: PortInsertionAssemblySpec):
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

    @configclass
    class PortInsertionActionsCfg:
        pass

    actions = PortInsertionActionsCfg()
    setattr(actions, controller.action_name, action_cfg)
    return actions


def build_command_cfg(assembly: PortInsertionAssemblySpec):
    """Build the episode-level insertion goal command from target/port specs."""

    assembly.validate()
    goal = assembly.goal
    target = assembly.target
    port = assembly.selected_port()
    command_cfg = InsertionGoalCommandCfg(
        target_scene_name=goal.target_slot,
        target_root_prim=target.usd.root_prim,
        port_name=port.name,
        port_seat_frame_path=port.seat_frame_path,
        port_entrance_frame_path=port.entrance_frame_path,
        eef_pos_in_port_frame=goal.eef_pose_in_port_frame.pos,
        eef_quat_in_port_frame=goal.eef_pose_in_port_frame.rot,
        resampling_time_range=goal.resampling_time_range,
        debug_vis=goal.debug_vis,
    )
    _validate_command_cfg(assembly, command_cfg, port)

    @configclass
    class PortInsertionCommandsCfg:
        pass

    commands = PortInsertionCommandsCfg()
    setattr(commands, goal.command_name, command_cfg)
    return commands


def build_observation_cfg(assembly: PortInsertionAssemblySpec) -> dict[str, ObsGroup]:
    """Build the policy (proprio + vision + last action) and cheatcode obs groups.

    ``policy`` is what the BC policy sees at inference: low-dim proprio,
    split root-frame TCP/EEF position/orientation/velocity, three RGB cameras
    (uint8, normalize off), and the last action. ``cheatcode`` carries
    privileged root-frame command targets and insertion progress for BC-dataset
    analysis and asymmetric critics. Both groups disable concatenation because
    the policy mixes heterogeneous shapes (images + low-dim) and the recorder
    writes each term as its own HDF5 dataset.
    """

    assembly.validate()
    robot_name = assembly.layout.robot_slot.name
    joint_names = list(assembly.robot.joint_group(assembly.controller.joint_group).joint_names)
    tcp_body = assembly.robot.body_name_for_role(assembly.controller.controlled_body_role)
    eef_body = assembly.robot.body_name_for_role(ROBOT_ROLE_EEF)
    action_name = assembly.controller.action_name
    command_name = assembly.goal.command_name
    lateral_threshold_m = assembly.observation.insertion_lateral_threshold_m

    joint_cfg = SceneEntityCfg(robot_name, joint_names=joint_names)
    tcp_cfg = SceneEntityCfg(robot_name, body_names=[tcp_body])
    eef_cfg = SceneEntityCfg(robot_name, body_names=[eef_body])
    wrist_wrench_cfg = SceneEntityCfg(robot_name, body_names=["ati_tool_link"])
    robot_root_cfg = SceneEntityCfg(robot_name)

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=joint_pos_rel, params={"asset_cfg": joint_cfg})
        joint_vel = ObsTerm(func=joint_vel_rel, params={"asset_cfg": joint_cfg})
        tcp_pos_b = ObsTerm(func=body_pos_b, params={"asset_cfg": tcp_cfg})
        tcp_quat_b = ObsTerm(func=body_quat_b, params={"asset_cfg": tcp_cfg})
        eef_pos_b = ObsTerm(func=body_pos_b, params={"asset_cfg": eef_cfg})
        eef_quat_b = ObsTerm(func=body_quat_b, params={"asset_cfg": eef_cfg})
        tcp_lin_vel_b = ObsTerm(func=body_lin_vel_b, params={"asset_cfg": tcp_cfg})
        tcp_ang_vel_b = ObsTerm(func=body_ang_vel_b, params={"asset_cfg": tcp_cfg})
        eef_lin_vel_b = ObsTerm(func=body_lin_vel_b, params={"asset_cfg": eef_cfg})
        eef_ang_vel_b = ObsTerm(func=body_ang_vel_b, params={"asset_cfg": eef_cfg})
        wrist_wrench = ObsTerm(func=body_incoming_wrench, params={"asset_cfg": wrist_wrench_cfg})
        center_camera_rgb = ObsTerm(
            func=image,
            params={
                "sensor_cfg": SceneEntityCfg("center_camera"),
                "data_type": "rgb",
                "normalize": False,
            },
        )
        left_camera_rgb = ObsTerm(
            func=image,
            params={
                "sensor_cfg": SceneEntityCfg("left_camera"),
                "data_type": "rgb",
                "normalize": False,
            },
        )
        right_camera_rgb = ObsTerm(
            func=image,
            params={
                "sensor_cfg": SceneEntityCfg("right_camera"),
                "data_type": "rgb",
                "normalize": False,
            },
        )
        actions = ObsTerm(func=last_action, params={"action_name": action_name})

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class CheatcodeCfg(ObsGroup):
        entrance_pos_b = ObsTerm(
            func=entrance_pos_b,
            params={"command_name": command_name, "asset_cfg": robot_root_cfg},
        )
        entrance_quat_b = ObsTerm(
            func=entrance_quat_b,
            params={"command_name": command_name, "asset_cfg": robot_root_cfg},
        )
        seat_pos_b = ObsTerm(
            func=seat_pos_b,
            params={"command_name": command_name, "asset_cfg": robot_root_cfg},
        )
        seat_quat_b = ObsTerm(
            func=seat_quat_b,
            params={"command_name": command_name, "asset_cfg": robot_root_cfg},
        )
        insertion_fraction = ObsTerm(
            func=insertion_fraction,
            params={
                "command_name": command_name,
                "asset_cfg": eef_cfg,
                "body_name": eef_body,
                "lateral_threshold_m": lateral_threshold_m,
            },
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = False

    return {"policy": PolicyCfg(), "cheatcode": CheatcodeCfg()}


def build_event_cfg(assembly: PortInsertionAssemblySpec) -> dict[str, EventTerm]:
    """Build reset events, including layout-owned scene randomization."""

    assembly.validate()
    robot_slot_name = assembly.layout.robot_slot.name
    events = {
        "reset_robot_joints": EventTerm(
            func=reset_joints_by_offset,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg(robot_slot_name),
                "position_range": (-0.3, 0.3),
                "velocity_range": (0.0, 0.0),
            },
        ),
        "randomize_light": EventTerm(
            func=randomize_dome_light,
            mode="reset",
            params={
                "light_scene_name": _LIGHT_SCENE_NAME,
                "intensity_range": (1500.0, 3500.0),
                "color_range": ((0.5, 0.5, 0.5), (1.0, 1.0, 1.0)),
            },
        ),
    }
    randomization = assembly.layout.randomization
    if randomization is not None:
        board_slot = assembly.layout.slot(randomization.board_slot_name)
        board_relative_parts = []
        for part in randomization.board_relative_parts:
            assembly.layout.slot(part.slot_name)
            board_relative_parts.append(
                {
                    "slot_name": part.slot_name,
                    "board_local_offset": part.board_local_offset,
                    "pose_ranges": {axis_range.axis: axis_range.bounds for axis_range in part.pose_ranges},
                    "snap_steps": {axis_snap.axis: axis_snap.step for axis_snap in part.snap_steps},
                }
            )
        events["randomize_board_and_parts"] = EventTerm(
            func=randomize_board_and_parts,
            mode="reset",
            params={
                "board_slot_name": randomization.board_slot_name,
                "board_default_position": board_slot.pose.pos,
                "board_pose_ranges": {axis_range.axis: axis_range.bounds for axis_range in randomization.board_ranges},
                "board_relative_parts": board_relative_parts,
                "sync_usd_xforms": randomization.sync_usd_xforms,
            },
        )
    return events


def build_termination_cfg(assembly: PortInsertionAssemblySpec) -> dict[str, DoneTerm]:
    """Build command-aware success, stationary failure, and timeout terms."""

    assembly.validate()
    termination = assembly.termination
    eef_body = assembly.robot.body_name_for_role(ROBOT_ROLE_EEF)
    robot_cfg = SceneEntityCfg(assembly.layout.robot_slot.name, body_names=eef_body)

    return {
        "success": DoneTerm(
            func=InsertionGoalReachedSuccess,
            params={
                "asset_cfg": robot_cfg,
                "command_name": assembly.goal.command_name,
                "tip_body": eef_body,
                "position_threshold": termination.success_position_threshold,
                "orientation_threshold": termination.success_orientation_threshold_rad,
                "required_seconds": termination.success_required_seconds,
            },
        ),
        "failed_stationary": DoneTerm(
            func=InsertionGoalStationaryFailure,
            params={
                "asset_cfg": robot_cfg,
                "command_name": assembly.goal.command_name,
                "tip_body": eef_body,
                "movement_threshold": termination.stationary_movement_threshold,
                "success_position_threshold": termination.stationary_success_position_threshold,
                "required_seconds": termination.stationary_required_seconds,
            },
        ),
        "time_out": DoneTerm(func=time_out, time_out=True),
    }


def build_empty_reward_cfg() -> dict:
    """Return an intentionally empty reward manager cfg."""

    return {}


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


def _build_asset_cfg(slot: SceneSlotSpec, asset: AssetSpec) -> AssetBaseCfg | RigidObjectCfg:
    spawn = _usd_file_cfg(asset, kinematic_enabled=slot.kinematic)
    if asset.usd.kind == "static":
        return AssetBaseCfg(
            prim_path=slot.prim_path,
            spawn=spawn,
            init_state=AssetBaseCfg.InitialStateCfg(pos=slot.pose.pos, rot=slot.pose.rot),
        )
    if asset.usd.kind == "rigid_object":
        return RigidObjectCfg(
            prim_path=slot.prim_path,
            spawn=spawn,
            init_state=RigidObjectCfg.InitialStateCfg(pos=slot.pose.pos, rot=slot.pose.rot),
        )
    raise ValueError(f"Scene slot '{slot.name}' is not a passive asset: {asset.usd.kind}")


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


def _support_slots(layout: SceneLayoutSpec) -> tuple[SceneSlotSpec, ...]:
    slots = []
    if layout.workcell_slot is not None:
        slots.append(layout.workcell_slot)
    if layout.board_slot is not None:
        slots.append(layout.board_slot)
    slots.extend(layout.auxiliary_slots)
    return tuple(slots)


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
    if command_cfg.port_name != port.name:
        raise ValueError("Command port identity does not match the selected target port.")
    if command_cfg.port_entrance_frame_path is None:
        raise ValueError("Command port entrance frame path cannot be empty.")
    if command_cfg.port_entrance_frame_path != port.entrance_frame_path:
        raise ValueError("Command port entrance frame path does not match the selected target port.")
    if not command_cfg.port_seat_frame_path:
        raise ValueError("Command port seat frame path cannot be empty.")
    if command_cfg.port_seat_frame_path != port.seat_frame_path:
        raise ValueError("Command port seat frame path does not match the selected target port.")
    if not any(abs(value) > 0.0 for value in command_cfg.eef_quat_in_port_frame):
        raise ValueError("Command EEF orientation cannot be the zero quaternion.")


def _validate_scene_slots(assembly: PortInsertionAssemblySpec) -> None:
    if assembly.layout.robot_slot.role != "robot":
        raise ValueError(f"Robot slot '{assembly.layout.robot_slot.name}' must have role 'robot'.")
    if assembly.layout.target_slot.role != "target":
        raise ValueError(f"Target slot '{assembly.layout.target_slot.name}' must have role 'target'.")
    if assembly.robot.usd.kind != "articulation":
        raise ValueError(f"Selected robot asset '{assembly.robot.name}' must be an articulation.")
    if assembly.target.usd.kind != "rigid_object":
        raise ValueError(f"Selected target asset '{assembly.target.name}' must be a rigid object.")
    for slot in _support_slots(assembly.layout):
        if slot.asset.usd.kind == "articulation":
            raise ValueError(f"Support slot '{slot.name}' cannot contain an articulation asset.")


__all__ = [
    "build_action_cfg",
    "build_command_cfg",
    "build_empty_reward_cfg",
    "build_event_cfg",
    "build_observation_cfg",
    "build_scene_cfg",
    "build_termination_cfg",
]
