"""Microduck corridor EXIT: leave the top of the vertical corridor onto a
platform (user sketch 2026-09-20).

The third link in the corridor chain.  The approach walks in and stops in a
standing handover pose; the climb goes up; this policy takes the climb's own
terminal state - WEDGED, just above platform level - and turns the corner:
travel out of the slot along +y while still braced between the walls, which
overhang the platform for exactly that purpose, then descend and stand.

Built on the CHIMNEY cfg rather than the walking one, because the exit starts
braced: joint actions are offsets from the default pose, and the brace IS the
chimney's default.  A policy that begins wedged must have the same neutral
action as the policy that hands it the wedge, or the handover moves the legs
before it does anything else.

Env knobs: MICRODUCK_EX_{WIDTH,PLATFORM_TOP,OVERHANG,PLATFORM_W,PROGRESS_W,
LAND_W,DASH_W,FALL_W,EPISODE_S,ENTROPY,INIT_STD}.
"""
from __future__ import annotations

import os

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import EventTermCfg, ObservationTermCfg, RewardTermCfg, TerminationTermCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg

from mjlab_microduck.robot import exit_stage as ex
from mjlab_microduck.tasks import exit_mdp as em
from mjlab_microduck.tasks.microduck_chimney_env_cfg import (
    BRACE_OVERRIDES,
    UPRIGHT_OVERRIDES,
    make_microduck_chimney_env_cfg,
)
from mjlab_microduck.tasks.symmetry import SYMMETRY_CFG, PpoWithSymmetryCfg

WIDTH = float(os.getenv("MICRODUCK_EX_WIDTH", "0.125"))
# The whole exit is at most ~50 cm of travel, so progress is weighted well below
# the climb's 800: the corner is short and the LANDING is the hard part.
PROGRESS_W = float(os.getenv("MICRODUCK_EX_PROGRESS_W", "400"))
# The intermediate must not out-pay the finish.  At land 3 / clear 8 the
# arithmetic over a 10 s episode was: sit under the overhang collecting
# 3 x 400 = 1200, or walk out and collect 3 x 100 + 8 x 25 = 500 and END the
# episode, forfeiting the rest.  Loitering paid 2.4x the finish, and both runs
# duly stopped at the wall end.  A per-step intermediate reward competes with a
# terminating goal for the whole remaining episode, so it has to be small.
LAND_W = float(os.getenv("MICRODUCK_EX_LAND_W", "0.5"))     # intermediate: on the platform
CLEAR_W = float(os.getenv("MICRODUCK_EX_CLEAR_W", "30.0"))  # the finish: out on the open part
DASH_W = float(os.getenv("MICRODUCK_EX_DASH_W", "120.0"))
FALL_W = float(os.getenv("MICRODUCK_EX_FALL_W", "30.0"))
EPISODE_LENGTH_S = float(os.getenv("MICRODUCK_EX_EPISODE_S", "8.0"))
ENTROPY_COEF = float(os.getenv("MICRODUCK_EX_ENTROPY", "0.005"))
INIT_STD = float(os.getenv("MICRODUCK_EX_INIT_STD", "0.6"))
# Reverse-curriculum spawn mix.  Without it the landing bonus never fires: x1
# and x2 both logged Episode_Reward/ex_land = 0.0000 after 1250 iterations.
FRONTIER_PROB = float(os.getenv("MICRODUCK_EX_FRONTIER_PROB", "0.40"))
NEARLY_DONE_PROB = float(os.getenv("MICRODUCK_EX_NEARLY_DONE_PROB", "0.20"))
# A bank of REAL climb-arrival states and how often to spawn from it.
CARRY_W = float(os.getenv("MICRODUCK_EX_CARRY_W", "2.0"))
COLLAPSE_W = float(os.getenv("MICRODUCK_EX_COLLAPSE_W", "200.0"))  # must exceed the rest of the episode
LEAN_W = float(os.getenv("MICRODUCK_EX_LEAN_W", "2000.0"))  # one-shot; dt-scaled -> 40 units
STILL_W = float(os.getenv("MICRODUCK_EX_STILL_W", "3.0"))
LEVEL_W = float(os.getenv("MICRODUCK_EX_LEVEL_W", "3.0"))
LANDED_BONUS_W = float(os.getenv("MICRODUCK_EX_LANDED_BONUS_W", "1500.0"))  # one-shot; dt-scaled -> 30 units
ARRIVAL_BANK = os.getenv("MICRODUCK_EX_ARRIVAL_BANK", "")
ARRIVAL_PROB = float(os.getenv("MICRODUCK_EX_ARRIVAL_PROB", "0.0"))
# With the goal moved out past the wall end, the nearly-done spawns must land
# THERE - otherwise the finish bonus never fires, which is the failure this
# task already hit once.


def make_microduck_exit_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = make_microduck_chimney_env_cfg(play=play)
    # the walls must be REBUILT with the overhang: corridor_stage fixes its
    # depth at import time, so the chimney's 30 cm walls would not reach over
    # the platform even though every helper says they do
    cfg.scene.entities.update(ex.make_overhang_corridor_entity_cfgs())
    cfg.scene.entities.update(ex.make_exit_stage_entity_cfgs())
    cfg.episode_length_s = EPISODE_LENGTH_S
    cfg.viewer.distance = 1.3

    # ── rewards: the climb's terms go, the corner's terms replace them ───────
    for name in list(cfg.rewards):
        if name.startswith("ch_"):
            cfg.rewards.pop(name)
    cfg.rewards["ex_progress"] = RewardTermCfg(func=em.exit_progress, weight=PROGRESS_W)
    cfg.rewards["ex_land"] = RewardTermCfg(func=em.exit_landed, weight=LAND_W)
    cfg.rewards["ex_clear"] = RewardTermCfg(func=em.exit_cleared, weight=CLEAR_W)
    # SELF-NEGATING → POSITIVE weights (mdp sign convention)
    cfg.rewards["ex_dash"] = RewardTermCfg(func=em.exit_dash_penalty, weight=DASH_W)
    cfg.rewards["ex_fall"] = RewardTermCfg(func=em.exit_fall_cost, weight=FALL_W)
    # Standing is a ROUTE the old stack never paid for, so shape it per step
    # rather than only gating success on it.
    cfg.rewards["ex_carry"] = RewardTermCfg(func=em.carriage_reward, weight=CARRY_W)
    cfg.rewards["ex_collapse"] = RewardTermCfg(func=em.collapse_cost, weight=COLLAPSE_W)
    cfg.rewards["ex_lean"] = RewardTermCfg(func=em.lean_cost, weight=LEAN_W)
    cfg.rewards["ex_still"] = RewardTermCfg(func=em.stillness_reward, weight=STILL_W)
    cfg.rewards["ex_level"] = RewardTermCfg(func=em.level_reward, weight=LEVEL_W)
    cfg.rewards["ex_landed_bonus"] = RewardTermCfg(func=em.landed_bonus, weight=LANDED_BONUS_W)

    # ── observations ─────────────────────────────────────────────────────────
    for group in ("actor", "critic"):
        cfg.observations[group].terms["body_command"] = ObservationTermCfg(
            func=em.exit_command_obs, params={"dim": 6},
        )

    # ── terminations ─────────────────────────────────────────────────────────
    cfg.terminations.pop("fell", None)
    cfg.terminations["fell"] = TerminationTermCfg(func=em.exit_fell, time_out=False)
    # Down on the platform used to score as a WIN; make it an explicit loss.
    cfg.terminations["collapsed"] = TerminationTermCfg(func=em.collapsed, time_out=False)
    cfg.terminations["leaning"] = TerminationTermCfg(func=em.leaning, time_out=False)
    # landing is a SUCCESS and must bootstrap, or arriving is valued like falling
    cfg.terminations["landed"] = TerminationTermCfg(
        # Hold for TWO seconds, not half a one.  With a 0.5 s hold the episode
        # ended almost the moment the pose was first struck, so standing
        # afterwards was never trained - filmed without the termination, the
        # duck clears the walls and then topples onto the platform.  A success
        # that ends the episode only trains the instant it fires.
        func=em.exit_landed_done, time_out=True, params={"hold_s": 2.0},
    )

    # ── events ───────────────────────────────────────────────────────────────
    cfg.events.pop("set_chimney_state", None)
    cfg.events["set_exit_state"] = EventTermCfg(
        func=em.reset_exit_state, mode="reset",
        params={
            "width_range": (WIDTH, WIDTH),
            "brace_overrides": BRACE_OVERRIDES,
            "stand_overrides": UPRIGHT_OVERRIDES,
            "frontier_prob": FRONTIER_PROB,
            "nearly_done_prob": NEARLY_DONE_PROB,
            "handover_bank": ARRIVAL_BANK,
            "handover_prob": ARRIVAL_PROB,
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


MicroduckExitRlCfg = RslRlOnPolicyRunnerCfg(
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
    experiment_name="exit",
    run_name="exit",
    save_interval=250,
    num_steps_per_env=24,
    max_iterations=20000,
)
