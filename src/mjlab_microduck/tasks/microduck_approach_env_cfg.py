"""Microduck corridor APPROACH: walk in off the floor and stop in the pose the
climb takes over from (user 2026-09-19: "a separate policy that starts in front
of the gap and enters the starting position in it, and ideally then triggers a
switch to the next policy").

This is the missing half of the corridor chain.  The climb (w12) now accepts a
standing handover 100 % of the time from a floor start, so what is left is
getting the duck there - and the cheap route was measured first and ruled out:
walker_v1 commanded straight at a 12.5 cm corridor from 32 cm out gains about
10 cm and never once ends standing between the walls.

Built on the VELOCITY recipe rather than the long-jump one, for a reason that
matters: joint actions are offsets from the robot's default pose, and the
climb's default IS the brace.  An approach policy has to walk, so its default
has to be the stance - which also makes walker_v1 a legal warm start.

The corridor is pinned to 12.5 cm by default, the width used in the climb clip
the user asked the entry to work at.  Env knobs:
MICRODUCK_AP_{WIDTH,DISTANCE_MAX,PROGRESS_W,ARRIVE_W,JAM_W,FALL_W,EPISODE_S,
ENTROPY,INIT_STD}.
"""
from __future__ import annotations

import os

from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab_microduck.tasks.bounded_actions import bounded_cfg_from, command_saturation_penalty
from mjlab.managers import EventTermCfg, ObservationTermCfg, RewardTermCfg, TerminationTermCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg

from mjlab_microduck.robot import corridor_stage as cs
from mjlab_microduck.robot.microduck_constants import HOME_FRAME
from mjlab_microduck.tasks import approach_mdp as ap
from mjlab_microduck.tasks.microduck_velocity_env_cfg import make_microduck_velocity_env_cfg
from mjlab_microduck.tasks.symmetry import SYMMETRY_CFG, PpoWithSymmetryCfg

# the standing pose the climb hands over from (chimney UPRIGHT_OVERRIDES)
STAND_OVERRIDES = {2: -0.4579, 3: -0.0049, 4: 0.4529, 11: 0.4579, 12: 0.0049, 13: -0.4529}

WIDTH = float(os.getenv("MICRODUCK_AP_WIDTH", "0.125"))
DISTANCE_MAX = float(os.getenv("MICRODUCK_AP_DISTANCE_MAX", "0.50"))
# Progress is the metres given up per step, so a whole 50 cm approach pays
# PROGRESS_W * 0.5 in total.  Arrival pays per step at ARRIVE_W until the hold
# completes and the episode ends, so it is bounded at ARRIVE_W * hold_s / dt.
PROGRESS_W = float(os.getenv("MICRODUCK_AP_PROGRESS_W", "300"))
ARRIVE_W = float(os.getenv("MICRODUCK_AP_ARRIVE_W", "4.0"))
JAM_W = float(os.getenv("MICRODUCK_AP_JAM_W", "20.0"))
DASH_W = float(os.getenv("MICRODUCK_AP_DASH_W", "30.0"))
# Orientation, charged per step.  Without it the policy turns and walks in
# nose-first, because progress pays for distance whichever way the duck faces.
YAW_W = float(os.getenv("MICRODUCK_AP_YAW_W", "2.0"))
FALL_W = float(os.getenv("MICRODUCK_AP_FALL_W", "10.0"))
EPISODE_LENGTH_S = float(os.getenv("MICRODUCK_AP_EPISODE_S", "8.0"))
ENTROPY_COEF = float(os.getenv("MICRODUCK_AP_ENTROPY", "0.004"))
INIT_STD = float(os.getenv("MICRODUCK_AP_INIT_STD", "0.5"))
# Reverse-curriculum spawn mix.  Needed once the handover required facing across
# the slot: without it the arrival bonus never fires (a7 scored 0 %).
FRONTIER_PROB = float(os.getenv("MICRODUCK_AP_FRONTIER_PROB", "0.35"))
NEARLY_DONE_PROB = float(os.getenv("MICRODUCK_AP_NEARLY_DONE_PROB", "0.15"))


def make_microduck_approach_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = make_microduck_velocity_env_cfg(play=play)

    # The approach hands over to the CLIMB, so both halves of the chain must run
    # the same body.  The velocity recipe brings the walk model
    # (``get_walk_spec``) while the climb is built on the long-jump lineage and
    # uses the collision-covered model (``get_long_jump_spec``) - a handover
    # between two different robots, which defeats the point.  Take the climb's
    # robot and keep the walking recipe's everything else.
    #
    # The DEFAULT POSE stays the stance, not the brace: joint actions are
    # offsets from it, and this policy has to walk.  Differing offsets between
    # the two policies is expected and is what the runtime swaps along with the
    # network; differing MODELS is not.
    from mjlab_microduck.tasks.microduck_long_jump_env_cfg import make_microduck_long_jump_env_cfg

    _lj_robot = make_microduck_long_jump_env_cfg(play=play).scene.entities["robot"]
    # The approach is deployed as a GUEST inside the exit env, whose action
    # term is BoundedJointPositionAction (hardware command bounds).  Built from
    # the velocity recipe, this cfg kept the plain unbounded term, so a11
    # trained unbounded and was deployed bounded: measured 2026-09-22, 99.8 %
    # handover in its own env vs 84 % as a guest, tumbling in the throat when
    # the side-step saturated the hip bounds.  Train under the bounds it will
    # run with - the standing hardware rule, and the chain needs it.
    cfg.actions["joint_pos"] = bounded_cfg_from(cfg.actions["joint_pos"], bound_margin=0.0)
    cfg.scene.entities["robot"] = replace(_lj_robot, init_state=HOME_FRAME)

    cfg.scene.entities.update(cs.make_corridor_entity_cfgs())
    cfg.scene.env_spacing = max(cfg.scene.env_spacing, 3.0)
    cfg.episode_length_s = EPISODE_LENGTH_S
    cfg.viewer.distance = 1.4

    # ── rewards: the task terms are replaced, the regularizers are kept ──────
    # (the velocity recipe's smoothness/limit terms are what keep a warm-started
    # walker walking; only its command-tracking terms are wrong here)
    for name in list(cfg.rewards):
        if "track" in name or "velocity" in name:
            cfg.rewards.pop(name)
    # The gait reward is gated on the twist COMMAND in the walking recipe.  Here
    # that command is a leftover the task never asks the policy to follow, so
    # re-gate it on distance to the goal - otherwise the policy is paid to keep
    # stepping after it has arrived, which fights the stillness the handover
    # needs.
    if "air_time" in cfg.rewards:
        _air = cfg.rewards["air_time"]
        _base_func, _base_params = _air.func, dict(_air.params)
        _base_params["command_threshold"] = 0.0
        cfg.rewards["air_time"] = RewardTermCfg(
            func=ap.gait_gate, weight=_air.weight,
            params={"base_func": _base_func, "base_params": _base_params, "far_m": 0.12},
        )
    cfg.rewards["ap_progress"] = RewardTermCfg(func=ap.approach_progress, weight=PROGRESS_W)
    cfg.rewards["ap_arrive"] = RewardTermCfg(func=ap.approach_arrive, weight=ARRIVE_W)
    # SELF-NEGATING → POSITIVE weights (mdp sign convention)
    cfg.rewards["ap_jam"] = RewardTermCfg(func=ap.approach_jam_penalty, weight=JAM_W)
    cfg.rewards["ap_dash"] = RewardTermCfg(func=ap.approach_dash_penalty, weight=DASH_W)
    cfg.rewards["ap_yaw"] = RewardTermCfg(func=ap.approach_yaw_penalty, weight=YAW_W)
    cfg.rewards["ap_fall"] = RewardTermCfg(func=ap.approach_fall_cost, weight=FALL_W)

    # ── observations: where the corridor is, and how wide ────────────────────
    for group in ("actor", "critic"):
        cfg.observations[group].terms["body_command"] = ObservationTermCfg(
            func=ap.approach_command_obs, params={"dim": 6},
        )

    # ── terminations ─────────────────────────────────────────────────────────
    cfg.terminations["fell"] = TerminationTermCfg(func=ap.approach_fell, time_out=False)
    # arrival is a SUCCESS, so it bootstraps: zeroing its value would teach the
    # policy that reaching the goal is as bad as falling over
    cfg.terminations["arrived"] = TerminationTermCfg(
        func=ap.approach_arrived, time_out=True, params={"hold_s": 0.4},
    )

    # ── events ───────────────────────────────────────────────────────────────
    for name in list(cfg.events):
        if "reset_base" in name or "reset_robot" in name:
            cfg.events.pop(name)
    cfg.events["set_approach_state"] = EventTermCfg(
        func=ap.reset_approach_state, mode="reset",
        params={
            "width_range": (WIDTH, WIDTH),
            "distance_range": (0.25, DISTANCE_MAX),
            "stand_overrides": STAND_OVERRIDES,
            "frontier_prob": FRONTIER_PROB,
            "nearly_done_prob": NEARLY_DONE_PROB,
        },
    )
    # Contact buffer.  The default 35 OVERFLOWS in this task - the logs read
    # "broadphase overflow - please increase nconmax to 46" - and an overflow
    # silently DROPS contacts, which is exactly the wall support a braced duck
    # depends on.  The sitstand recipe hit the same wall ("legs + head all in
    # close ground/self contact") and uses 200; so do the tug and stilt envs.
    cfg.sim.nconmax = 200
    # nconmax was raised here long ago; contact_sensor_maxmatch was NOT, and it
    # defaults to 64.  a12 logged "increase Option.contact_sensor_maxmatch to
    # 81" (2026-09-21), i.e. contact SENSOR matches were being dropped in this
    # task exactly the way broadphase contacts once were - silently, and with
    # the rewards that read contact quietly going wrong.
    cfg.sim.contact_sensor_maxmatch = 500
    return cfg


MicroduckApproachRlCfg = RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": INIT_STD, "std_type": "scalar"},
    ),
    critic=RslRlModelCfg(hidden_dims=(512, 256, 128), activation="elu", obs_normalization=True),
    algorithm=PpoWithSymmetryCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=ENTROPY_COEF,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        symmetry_cfg=SYMMETRY_CFG,
    ),
    wandb_project="mjlab_microduck",
    experiment_name="approach",
    run_name="approach",
    save_interval=250,
    num_steps_per_env=24,
    max_iterations=8000,
)
