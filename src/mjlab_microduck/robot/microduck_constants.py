import os
from pathlib import Path

import mujoco
from mjlab.actuator import XmlActuatorCfg
from mjlab_microduck.actuator import (
    BacklashEncoderBamActuatorCfg,
    FrictionDRBamActuatorCfg,
)
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg


_ROBOT_DIR: Path = Path(os.path.dirname(__file__)) / "microduck"

MICRODUCK_WALK_XML: Path = _ROBOT_DIR / "robot_walk.xml"
# Full-collision model, shared by standup / ground-pick / walk-rollers tasks.
MICRODUCK_ALLCOLLISIONS_XML: Path = _ROBOT_DIR / "robot_allcollisions.xml"
# 70mm / 15g ball prop for the BallKick task.
MICRODUCK_BALL_XML: Path = _ROBOT_DIR / "ball.xml"
# Roller-skate model: 14 actuated joints + passive wheel hinges (passive_*wheel).
MICRODUCK_ALLCOLLISIONS_ROLLERS_XML: Path = _ROBOT_DIR / "robot_allcollisions_rollers.xml"
# Backlash models: every servo joint gets an unactuated passive_<joint>_backlash
# hinge in series (±1° play, 2° total). Exported via
# config_mjcf_{allcollisions,walk}_backlash.json (add_backlash.py post-processor).
MICRODUCK_ALLCOLLISIONS_BACKLASH_XML: Path = _ROBOT_DIR / "robot_allcollisions_backlash.xml"
MICRODUCK_WALK_BACKLASH_XML: Path = _ROBOT_DIR / "robot_walk_backlash.xml"
MICRODUCK_ALLCOLLISIONS_ROLLERS_BACKLASH_XML: Path = _ROBOT_DIR / "robot_allcollisions_rollers_backlash.xml"

# Swing geometry and ideal-string parameters.  The attachment coordinates are
# the eyelet centres of the retained seat designed in microduck_seat_render.
SWING_STRING_LENGTH = 0.38
# Real cord is not perfectly inextensible.  A tension-only dead-band spring
# begins pulling at the nominal 380 mm length, while a soft 395 mm upper limit
# is only a safety catch.  At the 895 g combined robot+seat mass, 2 kN/m per
# string gives about 2.2 mm static extension.
SWING_STRING_STIFFNESS = 2000.0
SWING_STRING_LIMIT = 0.395
SWING_HANG_LENGTH = 0.3822
SWING_ANCHOR_HEIGHT = 0.75
SWING_ATTACHMENT_Y = 0.088
SWING_ATTACHMENT_Z = 0.085
SWING_BOTTOM_TRUNK_Z = SWING_ANCHOR_HEIGHT - SWING_HANG_LENGTH - SWING_ATTACHMENT_Z

assert MICRODUCK_WALK_XML.exists(), f"XML not found: {MICRODUCK_WALK_XML}"
assert MICRODUCK_ALLCOLLISIONS_XML.exists(), f"XML not found: {MICRODUCK_ALLCOLLISIONS_XML}"
assert MICRODUCK_BALL_XML.exists(), f"XML not found: {MICRODUCK_BALL_XML}"
assert MICRODUCK_ALLCOLLISIONS_ROLLERS_XML.exists(), f"XML not found: {MICRODUCK_ALLCOLLISIONS_ROLLERS_XML}"
assert MICRODUCK_ALLCOLLISIONS_BACKLASH_XML.exists(), f"XML not found: {MICRODUCK_ALLCOLLISIONS_BACKLASH_XML}"
assert MICRODUCK_WALK_BACKLASH_XML.exists(), f"XML not found: {MICRODUCK_WALK_BACKLASH_XML}"
assert MICRODUCK_ALLCOLLISIONS_ROLLERS_BACKLASH_XML.exists(), f"XML not found: {MICRODUCK_ALLCOLLISIONS_ROLLERS_BACKLASH_XML}"


def get_walk_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(MICRODUCK_WALK_XML))


def get_standup_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(MICRODUCK_ALLCOLLISIONS_XML))


def get_ground_pick_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(MICRODUCK_ALLCOLLISIONS_XML))


def get_walk_rollers_spec() -> mujoco.MjSpec:
    # NOTE: was loading robot_allcollisions.xml (no wheels) — the roller env
    # silently ran on the wheel-less standup model.
    return mujoco.MjSpec.from_file(str(MICRODUCK_ALLCOLLISIONS_ROLLERS_XML))


def get_ball_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(MICRODUCK_BALL_XML))


def get_backlash_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(MICRODUCK_ALLCOLLISIONS_BACKLASH_XML))


def get_walk_backlash_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(MICRODUCK_WALK_BACKLASH_XML))


def get_rollers_backlash_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(MICRODUCK_ALLCOLLISIONS_ROLLERS_BACKLASH_XML))


def get_swing_spec() -> mujoco.MjSpec:
    """Return MicroDuck rigidly retained in its seat on two ideal strings.

    Each string is a MuJoCo spatial tendon with a tension-only elastic dead
    band plus a soft upper safety limit. It can transmit tension but never
    compression and is therefore a massless cord rather than a rod or a
    decorative capsule.
    The seat and retention kit are fixed to the trunk (the fitted strap model)
    while every head and leg joint remains actuated and free to pump.
    """
    spec = mujoco.MjSpec.from_file(str(MICRODUCK_ALLCOLLISIONS_XML))

    visual_meshes = (
        "swing_seat_retained",
        "swing_retention_strap",
        "swing_retention_buckle",
        "swing_retention_bumper_0",
        "swing_retention_bumper_1",
        "swing_retention_bumper_2",
    )
    for name in visual_meshes:
        spec.add_mesh(name=name, file=f"{name}.stl")

    spec.add_material(name="swing_seat_material", rgba=(0.055, 0.062, 0.068, 1.0))
    spec.add_material(name="swing_strap_material", rgba=(0.075, 0.080, 0.085, 1.0))
    spec.add_material(name="swing_bumper_material", rgba=(0.12, 0.13, 0.14, 1.0))
    spec.add_material(name="swing_buckle_material", rgba=(0.24, 0.25, 0.26, 1.0))
    spec.add_material(name="swing_frame_material", rgba=(0.28, 0.18, 0.11, 1.0))

    # The training A-frame is visual-only so incidental stand strikes do not
    # change the established swing task.  Frontier-failure evaluation can opt
    # into the physically collidable real-stand geometry.  This makes a top-bar
    # impact an actual MuJoCo contact/fall, rather than a scripted animation or
    # a mesh interpenetration.
    collidable_frame = os.environ.get(
        "MICRODUCK_SWING_COLLIDABLE_FRAME", "0"
    ).lower() in {"1", "true", "yes"}
    frame_segments = {
        "swing_frame_front_left": ((0.34, 0.34, 0.0), (0.0, 0.25, 0.755)),
        "swing_frame_back_left": ((-0.34, 0.34, 0.0), (0.0, 0.25, 0.755)),
        "swing_frame_front_right": ((0.34, -0.34, 0.0), (0.0, -0.25, 0.755)),
        "swing_frame_back_right": ((-0.34, -0.34, 0.0), (0.0, -0.25, 0.755)),
        "swing_frame_crossbar": ((0.0, -0.27, 0.755), (0.0, 0.27, 0.755)),
    }
    for name, fromto in frame_segments.items():
        spec.worldbody.add_geom(
            name=name,
            type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            fromto=fromto[0] + fromto[1],
            size=(0.009 if name == "swing_frame_crossbar" else 0.007,),
            material="swing_frame_material",
            contype=1 if collidable_frame else 0,
            conaffinity=1 if collidable_frame else 0,
            group=2,
        )

    trunk = next(body for body in spec.bodies if body.name == "trunk_base")
    # Manager reset events use SWING_SEATED_FRAME, but mjlab exposes the raw
    # compiled qpos0 before the first reset.  Keep the MJCF root at the same
    # hanging equilibrium so rollout step 0 cannot begin with overstretched
    # strings from robot_allcollisions.xml's standing z=0.12 pose.
    trunk.pos = (0.0, 0.0, SWING_BOTTOM_TRUNK_Z)
    payload = trunk.add_body(name="swing_seat_payload")
    # 150 g seat + 8 g retention kit.  Giving the mesh geoms mass makes the
    # payload's distributed inertia part of the actual articulated dynamics.
    payload.add_geom(
        name="swing_seat_visual",
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname="swing_seat_retained",
        material="swing_seat_material",
        contype=0,
        conaffinity=0,
        mass=0.150,
        group=2,
    )
    payload.add_geom(
        name="swing_retention_strap_visual",
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname="swing_retention_strap",
        material="swing_strap_material",
        contype=0,
        conaffinity=0,
        mass=0.004,
        group=2,
    )
    payload.add_geom(
        name="swing_retention_buckle_visual",
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname="swing_retention_buckle",
        material="swing_buckle_material",
        contype=0,
        conaffinity=0,
        mass=0.002,
        group=2,
    )
    for index in range(3):
        payload.add_geom(
            name=f"swing_retention_bumper_{index}_visual",
            type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname=f"swing_retention_bumper_{index}",
            material="swing_bumper_material",
            contype=0,
            conaffinity=0,
            mass=0.002 / 3.0,
            group=2,
        )

    for side, y in (("left", SWING_ATTACHMENT_Y), ("right", -SWING_ATTACHMENT_Y)):
        spec.worldbody.add_site(
            name=f"swing_anchor_{side}",
            pos=(0.0, y, SWING_ANCHOR_HEIGHT),
            size=(0.003,),
            rgba=(0.55, 0.38, 0.22, 1.0),
        )
        trunk.add_site(
            name=f"swing_attach_{side}",
            pos=(0.0, y, SWING_ATTACHMENT_Z),
            size=(0.003,),
            rgba=(0.55, 0.38, 0.22, 1.0),
        )
        string = spec.add_tendon(
            name=f"swing_string_{side}",
            stiffness=SWING_STRING_STIFFNESS,
            springlength=(0.0, SWING_STRING_LENGTH),
            limited=True,
            range=(0.0, SWING_STRING_LIMIT),
            width=0.0016,
            rgba=(0.92, 0.87, 0.70, 1.0),
            solref_limit=(0.02, 1.0),
            solimp_limit=(0.90, 0.95, 0.001, 0.5, 2.0),
        )
        string.wrap_site(f"swing_anchor_{side}")
        string.wrap_site(f"swing_attach_{side}")

    return spec


HOME_FRAME = EntityCfg.InitialStateCfg(
    joint_pos={
        # Lower body — STAND2 pose: trunk shifted ~5mm forward over the feet so
        # the CoM sits over the ankle axis (was ~5mm behind it at the old HOME,
        # which biased the robot backward and made the standup policy droop its
        # head forward as a counterweight). Leg pitch chain leaned forward:
        # hip_pitch 30°→26.24°, ankle 30°→25.95°, knee 0°→0.28°. Matches the
        # STAND keyframe in scene.xml / scene_walk.xml.
        r".*hip_yaw.*": 0.0,
        r".*left_hip_roll.*": -0.0873,
        r".*right_hip_roll.*": 0.0873,
        r".*left_hip_pitch.*": -0.4579,
        r".*right_hip_pitch.*": 0.4579,
        r".*left_knee.*": -0.0049,
        r".*right_knee.*": 0.0049,
        r".*left_ankle.*": 0.4530,
        r".*right_ankle.*": -0.4530,
        # Head
        r".*neck_pitch.*": 0.3491,
        r".*head_pitch.*": 0.3491,
        r".*head_yaw.*": 0.0,
        r".*head_roll.*": 0.0,
    },
    joint_vel={".*": 0.0},
)

SWING_SEATED_FRAME = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, SWING_BOTTOM_TRUNK_Z),
    rot=(1.0, 0.0, 0.0, 0.0),
    lin_vel=(0.0, 0.0, 0.0),
    ang_vel=(0.0, 0.0, 0.0),
    joint_pos={
        r".*hip_yaw.*": 0.0,
        r".*hip_roll.*": 0.0,
        r".*left_hip_pitch.*": -0.4079,
        r".*right_hip_pitch.*": 0.4079,
        r".*left_knee.*": 1.35,
        r".*right_knee.*": -1.35,
        r".*ankle.*": 0.0,
        r".*neck_pitch.*": 0.3491,
        r".*head_pitch.*": 0.3491,
        r".*head_yaw.*": 0.0,
        r".*head_roll.*": 0.0,
    },
    joint_vel={".*": 0.0},
)

FULL_COLLISION = CollisionCfg(
    geom_names_expr=[".*_collision"],
    condim={r"^(left|right)_foot_collision$": 3, ".*_collision": 1},
    priority={r"^(left|right)_foot_collision$": 1},
    friction={r"^(left|right)_foot_collision$": (1.0,)},
)

# -- Old actuator (XML position, MuJoCo built-in PD + friction) --
# actuators = DelayedActuatorCfg(
    # delay_min_lag=0,
    # delay_max_lag=3,
    # base_cfg=XmlPositionActuatorCfg(joint_names_expr=(r".*",)),
# )

# -- BAM M6 actuator (full voltage control + load-dependent friction) --
# Exclude passive_* joints (jaw linkage in the new model has no XML actuator).
# Voltage domain randomization (mirrors mjlab_microban):
#   - vin_range: per-env battery voltage sampled at startup (replaces fixed vin)
#   - vin_drop_gain_range: load-dependent voltage sag V_drop = gain * sum(|tau|)
#   - vin_min: hard floor on the effective voltage after sag
# kp_fw kept at 200 (microduck's preserved firmware stiffness; microban uses 125).
_BAM_ACTUATOR_KWARGS = dict(
    motor_name="xl330",
    model="m6",
    target_names_expr=(r"^(?!passive_).*",),
    kp_fw=200.0,  # microduck's preserved firmware stiffness (microban uses 125)
    # vin_range=(6.9, 7.9),
    vin_range=(6.5, 8.2),
    vin_drop_gain_range=(0.0, 0.2),
    vin_min=6.0,
    # max_current=1.75,
    delay_min_lag=3,
    delay_max_lag=6,
)
actuators = FrictionDRBamActuatorCfg(**_BAM_ACTUATOR_KWARGS)

# Same BAM actuator, but the firmware position loop reads the encoder THROUGH
# the passive_<joint>_backlash hinges (the real encoder is on the output side
# of the gear play). Only for the backlash model; the target regex already
# excludes the passive_* backlash joints from actuation.
backlash_actuators = BacklashEncoderBamActuatorCfg(**_BAM_ACTUATOR_KWARGS)

# -- BAM M4 actuator
# actuators = DelayedActuatorCfg(
    # delay_min_lag=0,
    # delay_max_lag=3,
    # base_cfg=make_bam_m4_actuator_cfg(),
# )

# HOME frame for the backlash model. HOME_FRAME's unanchored patterns
# (e.g. r".*left_hip_roll.*") would also match passive_left_hip_roll_backlash
# and try to initialize it at -0.0873 rad — outside its ±1° range. Pattern
# matching is first-match-wins in declaration order, so the anchored backlash
# rule placed FIRST pins every backlash joint at 0 and the servo joints fall
# through to the normal HOME values.
BACKLASH_HOME_FRAME = EntityCfg.InitialStateCfg(
    joint_pos={r".*_backlash$": 0.0, **HOME_FRAME.joint_pos},
    joint_vel={".*": 0.0},
)

MICRODUCK_WALK_ROBOT_CFG = EntityCfg(
    spec_fn=get_walk_spec,
    init_state=HOME_FRAME,
    collisions=(FULL_COLLISION,),
    articulation=EntityArticulationInfoCfg(
        actuators=(actuators,),
        soft_joint_pos_limit_factor=0.9,
    ),
)

MICRODUCK_STANDUP_ROBOT_CFG = EntityCfg(
    spec_fn=get_standup_spec,
    init_state=HOME_FRAME,
    collisions=(FULL_COLLISION,),
    articulation=EntityArticulationInfoCfg(
        actuators=(actuators,),
        soft_joint_pos_limit_factor=0.9,
    ),
)

MICRODUCK_SWING_ROBOT_CFG = EntityCfg(
    spec_fn=get_swing_spec,
    init_state=SWING_SEATED_FRAME,
    collisions=(FULL_COLLISION,),
    articulation=EntityArticulationInfoCfg(
        actuators=(actuators,),
        soft_joint_pos_limit_factor=0.95,
    ),
)

MICRODUCK_GROUND_PICK_ROBOT_CFG = EntityCfg(
    spec_fn=get_ground_pick_spec,
    init_state=HOME_FRAME,
    collisions=(FULL_COLLISION,),
    articulation=EntityArticulationInfoCfg(
        actuators=(actuators,),
        soft_joint_pos_limit_factor=0.9,
    ),
)

# Backlash robots: base model + ±1° serial backlash hinge per servo.
# Encoder reads through the backlash (BacklashEncoderBamActuator feedback +
# joint_pos/vel_rel_backlash observations — see tasks/backlash.py).
# Allcollisions variant → VelStand/StandUp backlash tasks (mirrors
# MICRODUCK_STANDUP_ROBOT_CFG); walk variant → Velocity backlash
# tasks (mirrors MICRODUCK_WALK_ROBOT_CFG, keeps backlash-vs-base comparisons
# unconfounded by the collision model).
MICRODUCK_BACKLASH_ROBOT_CFG = EntityCfg(
    spec_fn=get_backlash_spec,
    init_state=BACKLASH_HOME_FRAME,
    collisions=(FULL_COLLISION,),
    articulation=EntityArticulationInfoCfg(
        actuators=(backlash_actuators,),
        soft_joint_pos_limit_factor=0.9,
    ),
)

MICRODUCK_WALK_BACKLASH_ROBOT_CFG = EntityCfg(
    spec_fn=get_walk_backlash_spec,
    init_state=BACKLASH_HOME_FRAME,
    collisions=(FULL_COLLISION,),
    articulation=EntityArticulationInfoCfg(
        actuators=(backlash_actuators,),
        soft_joint_pos_limit_factor=0.9,
    ),
)

# Roller-skate backlash robot: wheels stay free (passive_*wheel untouched by
# add_backlash.py). collisions=() mirrors MICRODUCK_WALK_ROLLERS_ROBOT_CFG —
# roller wheel collision geoms have no explicit names; XML defaults apply.
MICRODUCK_ROLLERS_BACKLASH_ROBOT_CFG = EntityCfg(
    spec_fn=get_rollers_backlash_spec,
    init_state=BACKLASH_HOME_FRAME,
    collisions=(),
    articulation=EntityArticulationInfoCfg(
        actuators=(backlash_actuators,),
        soft_joint_pos_limit_factor=0.9,
    ),
)

# Free-floating, non-articulated ball prop for the BallKick task. Position is
# set each episode by the reset_ball_in_front_of_foot event; the init pos here
# only matters for the pristine pre-first-reset state.
MICRODUCK_BALL_CFG = EntityCfg(
    spec_fn=get_ball_spec,
    init_state=EntityCfg.InitialStateCfg(pos=(0.3, 0.0, 0.035)),
)

# Roller skate robot: the 4 passive wheel joints (passive_*wheel) have no XML
# actuators; the BAM cfg's target regex already excludes them, so the action
# space stays 14-dimensional. Uses the SAME canonical BAM actuator as every
# other variant (was a plain XmlActuatorCfg PD — an actuator-physics mismatch
# vs the rest of the family, and joint-friction DR was impossible).
MICRODUCK_WALK_ROLLERS_ROBOT_CFG = EntityCfg(
    spec_fn=get_walk_rollers_spec,
    init_state=HOME_FRAME,
    collisions=(),  # roller wheel collision geoms have no explicit names; XML defaults apply
    articulation=EntityArticulationInfoCfg(
        actuators=(actuators,),
        soft_joint_pos_limit_factor=0.9,
    ),
)

if __name__ == "__main__":
    import mujoco.viewer as viewer
    from mjlab.scene import Scene, SceneCfg
    from mjlab.terrains import TerrainImporterCfg

    SCENE_CFG = SceneCfg(
        terrain=TerrainImporterCfg(terrain_type="plane"),
        entities={"robot": MICRODUCK_WALK_ROBOT_CFG},
    )

    scene = Scene(SCENE_CFG, device="cuda:0")
    viewer.launch(scene.compile())
