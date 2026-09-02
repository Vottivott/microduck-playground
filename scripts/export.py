"""Script to play RL agent with RSL-RL."""

import json
import math
import os
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import tyro
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.utils.os import get_checkpoint_path, get_wandb_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder
from rsl_rl.runners import OnPolicyRunner

from mjlab_microduck.onnx_policy_contract import bake_action_clip


@dataclass(frozen=True)
class ExportConfig:
    onnx_file: str = "output.onnx"
    agent: Literal["zero", "random", "trained"] = "trained"
    registry_name: str | None = None
    wandb_run_path: str | None = None
    checkpoint: int | None = None      # Select checkpoint by iteration number (e.g. 3000)
    checkpoint_file: str | None = None
    seed: int | None = None
    motion_file: str | None = None
    num_envs: int | None = None
    device: str | None = None
    video: bool = False
    video_length: int = 200
    video_height: int | None = None
    video_width: int | None = None
    episode_length_s: float | None = None
    camera: int | str | None = None
    viewer: Literal["auto", "native", "viser"] = "auto"
    running_speed: float | None = None
    camera_azimuth_offset_deg: float = 0.0
    camera_follow_yaw_tau_s: float | None = None
    camera_follow_position_tau_s: float | None = None
    camera_follow_position_gain: float = 1.0
    camera_follow_shadow_light: bool = False
    disable_shadows: bool = False
    contact_log_file: str | None = None

    # Internal flag used by demo script.
    _demo_mode: tyro.conf.Suppress[bool] = False


def run_export(task_id: str, cfg: ExportConfig):
    configure_torch_backends()

    if cfg.camera_follow_yaw_tau_s is not None:
        if cfg.camera_follow_yaw_tau_s <= 0.0:
            raise ValueError("camera_follow_yaw_tau_s must be positive")
        if not cfg.video:
            raise ValueError("camera_follow_yaw_tau_s requires video recording")
    if cfg.camera_follow_shadow_light and not cfg.video:
        raise ValueError("camera_follow_shadow_light requires video recording")
    if cfg.camera_follow_position_tau_s is not None:
        if cfg.camera_follow_position_tau_s <= 0.0:
            raise ValueError("camera_follow_position_tau_s must be positive")
        if not cfg.video:
            raise ValueError("camera_follow_position_tau_s requires video recording")
        if cfg.camera_follow_position_gain < 0.0:
            raise ValueError("camera_follow_position_gain must be non-negative")
    if cfg.contact_log_file is not None and not cfg.video:
        raise ValueError("contact_log_file requires video recording")

    device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    env_cfg = load_env_cfg(task_id, play=True)
    agent_cfg = load_rl_cfg(task_id)
    if cfg.episode_length_s is not None:
        if cfg.episode_length_s <= 0.0:
            raise ValueError("episode_length_s must be positive")
        env_cfg.episode_length_s = cfg.episode_length_s
    if cfg.seed is not None:
        env_cfg.seed = cfg.seed

    # Keep the configured follow-camera orbit intact while allowing recordings
    # from the opposite side of the tracked body.
    env_cfg.viewer.azimuth = (
        env_cfg.viewer.azimuth + cfg.camera_azimuth_offset_deg
    ) % 360.0

    if cfg.running_speed is not None:
        if cfg.running_speed < 0.0:
            raise ValueError("running_speed must be non-negative")
        if env_cfg.commands is None or "twist" not in env_cfg.commands:
            raise ValueError("running_speed requires a twist command")
        command = env_cfg.commands["twist"]
        command.ranges.lin_vel_x = (cfg.running_speed, cfg.running_speed)
        command.ranges.lin_vel_y = (0.0, 0.0)
        command.ranges.ang_vel_z = (0.0, 0.0)
        command.rel_standing_envs = 0.0
        if hasattr(command, "rel_turn_in_place_envs"):
            command.rel_turn_in_place_envs = 0.0

    DUMMY_MODE = cfg.agent in {"zero", "random"}
    TRAINED_MODE = not DUMMY_MODE

    # Check if this is a motion tracking task.
    is_motion_tracking = (
        env_cfg.commands is not None
        and "motion" in env_cfg.commands
        and isinstance(env_cfg.commands["motion"], MotionCommandCfg)
    )
    is_tracking_task = is_motion_tracking

    if is_tracking_task and cfg._demo_mode:
        # Demo mode: use uniform sampling to see more diversity with num_envs > 1.
        assert env_cfg.commands is not None
        motion_cmd = env_cfg.commands["motion"]
        assert isinstance(motion_cmd, MotionCommandCfg)
        motion_cmd.sampling_mode = "uniform"

    if is_tracking_task:
        assert env_cfg.commands is not None
        motion_cmd = env_cfg.commands["motion"]
        assert isinstance(motion_cmd, MotionCommandCfg)

        # Check if motion file is already set and exists
        motion_file_already_set = (
            hasattr(motion_cmd, 'motion_file')
            and motion_cmd.motion_file is not None
            and Path(motion_cmd.motion_file).exists()
        )

        if DUMMY_MODE:
            if not cfg.registry_name:
                raise ValueError(
                    "Tracking tasks require `registry_name` when using dummy agents."
                )
            # Check if the registry name includes alias, if not, append ":latest".
            registry_name = cfg.registry_name
            if ":" not in registry_name:
                registry_name = registry_name + ":latest"
            import wandb

            api = wandb.Api()
            artifact = api.artifact(registry_name)
            motion_cmd.motion_file = str(Path(artifact.download()) / "motion.npz")
        else:
            if cfg.motion_file is not None:
                print(f"[INFO]: Using motion file from CLI: {cfg.motion_file}")
                motion_cmd.motion_file = cfg.motion_file
            elif motion_file_already_set:
                print(f"[INFO]: Using motion file from env config: {motion_cmd.motion_file}")
            else:
                # Try to download from wandb artifacts
                import wandb

                api = wandb.Api()
                if cfg.wandb_run_path is None and cfg.checkpoint_file is not None:
                    raise ValueError(
                        "Tracking tasks require `motion_file` when using `checkpoint_file`, "
                        "or provide `wandb_run_path` so the motion artifact can be resolved."
                    )
                if cfg.wandb_run_path is not None:
                    wandb_run = api.run(str(cfg.wandb_run_path))
                    art = next(
                        (a for a in wandb_run.used_artifacts() if a.type == "motions"),
                        None,
                    )
                    if art is None:
                        raise RuntimeError("No motion artifact found in the run.")
                    motion_cmd.motion_file = str(Path(art.download()) / "motion.npz")

    log_dir: Path | None = None
    resume_path: Path | None = None
    if TRAINED_MODE:
        log_root_path = (Path("logs") / "rsl_rl" / agent_cfg.experiment_name).resolve()
        if cfg.checkpoint_file is not None:
            resume_path = Path(cfg.checkpoint_file)
            if not resume_path.exists():
                raise FileNotFoundError(f"Checkpoint file not found: {resume_path}")
            print(f"[INFO]: Loading checkpoint: {resume_path.name}")
        elif cfg.checkpoint is not None:
            # Select a specific checkpoint iteration, from wandb or local.
            checkpoint_filename = f"model_{cfg.checkpoint}.pt"
            if cfg.wandb_run_path is not None:
                import wandb
                api = wandb.Api()
                wandb_run = api.run(str(cfg.wandb_run_path))
                run_id = cfg.wandb_run_path.split("/")[-1]
                download_dir = log_root_path / "wandb_checkpoints" / run_id
                resume_path = download_dir / checkpoint_filename
                if resume_path.exists():
                    print(f"[INFO]: Loading checkpoint: {checkpoint_filename} (run: {run_id}, cached)")
                else:
                    available = [f.name for f in wandb_run.files() if "model" in f.name]
                    if checkpoint_filename not in available:
                        raise FileNotFoundError(
                            f"Checkpoint '{checkpoint_filename}' not found in wandb run. "
                            f"Available: {sorted(available)}"
                        )
                    wandb_run.file(checkpoint_filename).download(str(download_dir), replace=True)
                    print(f"[INFO]: Loading checkpoint: {checkpoint_filename} (run: {run_id}, downloaded)")
            else:
                resume_path = get_checkpoint_path(
                    log_root_path, checkpoint=re.escape(checkpoint_filename)
                )
                print(f"[INFO]: Loading checkpoint: {resume_path.name}")
        else:
            if cfg.wandb_run_path is None:
                raise ValueError(
                    "`wandb_run_path` is required when `checkpoint_file` is not provided."
                )
            resume_path, was_cached = get_wandb_checkpoint_path(
                log_root_path, Path(cfg.wandb_run_path)
            )
            # Extract run_id and checkpoint name from path for display.
            run_id = resume_path.parent.name
            checkpoint_name = resume_path.name
            cached_str = "cached" if was_cached else "downloaded"
            print(
                f"[INFO]: Loading checkpoint: {checkpoint_name} (run: {run_id}, {cached_str})"
            )
        log_dir = resume_path.parent

    if cfg.num_envs is not None:
        env_cfg.scene.num_envs = cfg.num_envs
    if cfg.video_height is not None:
        env_cfg.viewer.height = cfg.video_height
    if cfg.video_width is not None:
        env_cfg.viewer.width = cfg.video_width

    render_mode = "rgb_array" if (TRAINED_MODE and cfg.video) else None
    if cfg.video and DUMMY_MODE:
        print(
            "[WARN] Video recording with dummy agents is disabled (no checkpoint/log_dir)."
        )
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)

    if TRAINED_MODE and cfg.video:
        print("[INFO] Recording videos during play")
        assert log_dir is not None  # log_dir is set in TRAINED_MODE block
        env = VideoRecorder(
            env,
            video_folder=log_dir / "videos" / "play",
            step_trigger=lambda step: step == 0,
            video_length=cfg.video_length,
            disable_logger=True,
        )

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    if DUMMY_MODE:
        action_shape: tuple[int, ...] = env.unwrapped.action_space.shape  # type: ignore
        if cfg.agent == "zero":

            class PolicyZero:
                def __call__(self, obs) -> torch.Tensor:
                    del obs
                    return torch.zeros(action_shape, device=env.unwrapped.device)

            policy = PolicyZero()
        else:

            class PolicyRandom:
                def __call__(self, obs) -> torch.Tensor:
                    del obs
                    return 2 * torch.rand(action_shape, device=env.unwrapped.device) - 1

            policy = PolicyRandom()
    else:
        runner_cls = load_runner_cls(task_id) or OnPolicyRunner
        runner = runner_cls(env, asdict(agent_cfg), device=device)
        runner.load(str(resume_path), map_location=device)
        policy = runner.get_inference_policy(device=device)

    if TRAINED_MODE and cfg.video:
        # VideoRecorder captures only while stepping.  Keep rollout separate
        # from ONNX export so --video produces a real checkpoint video.
        obs = env.get_observations()
        camera_azimuth: float | None = None
        base_env = env.unwrapped
        renderer = base_env._offline_renderer
        assert renderer is not None
        if cfg.disable_shadows:
            # Mesa/OSMesa can intermittently corrupt MuJoCo's shadow map into
            # long black bars during large articulated motion.  Disabling only
            # shadow casting preserves scene lighting/materials while removing
            # that unstable render pass for deterministic artifact-free video.
            renderer._model.light_castshadow[:] = False
        tracked_entity = None
        if (
            cfg.camera_follow_yaw_tau_s is not None
            or cfg.camera_follow_position_tau_s is not None
            or cfg.camera_follow_shadow_light
        ):
            entity_name = env_cfg.viewer.entity_name
            if entity_name is None:
                raise ValueError("camera following requires a viewer entity_name")
            tracked_entity = base_env.scene[entity_name]

        position_initial_root: np.ndarray | None = None
        position_base_lookat: np.ndarray | None = None
        position_smoothed_lookat: np.ndarray | None = None
        if cfg.camera_follow_position_tau_s is not None:
            assert tracked_entity is not None
            position_initial_root = (
                tracked_entity.data.root_link_pos_w[env_cfg.viewer.env_idx]
                .detach()
                .cpu()
                .numpy()
                .copy()
            )
            position_base_lookat = np.asarray(renderer._cam.lookat).copy()
            position_smoothed_lookat = position_base_lookat.copy()

        frame_contact_samples: list[dict[str, object]] = []

        for video_step in range(cfg.video_length):
            if cfg.camera_follow_position_tau_s is not None:
                assert tracked_entity is not None
                assert position_initial_root is not None
                assert position_base_lookat is not None
                assert position_smoothed_lookat is not None
                root_pos_np = (
                    tracked_entity.data.root_link_pos_w[env_cfg.viewer.env_idx]
                    .detach()
                    .cpu()
                    .numpy()
                )
                target_lookat = position_base_lookat + cfg.camera_follow_position_gain * (
                    root_pos_np - position_initial_root
                )
                alpha = 1.0 - math.exp(
                    -base_env.step_dt / cfg.camera_follow_position_tau_s
                )
                position_smoothed_lookat += alpha * (
                    target_lookat - position_smoothed_lookat
                )
                renderer._cam.lookat[:] = position_smoothed_lookat

            if cfg.camera_follow_yaw_tau_s is not None:
                assert tracked_entity is not None
                quat = tracked_entity.data.root_link_quat_w[
                    env_cfg.viewer.env_idx
                ]
                qw, qx, qy, qz = (float(value) for value in quat)
                yaw = math.degrees(
                    math.atan2(
                        2.0 * (qw * qz + qx * qy),
                        1.0 - 2.0 * (qy * qy + qz * qz),
                    )
                )
                target_azimuth = yaw + env_cfg.viewer.azimuth
                if camera_azimuth is None:
                    camera_azimuth = target_azimuth
                else:
                    alpha = 1.0 - math.exp(
                        -base_env.step_dt / cfg.camera_follow_yaw_tau_s
                    )
                    delta = (
                        target_azimuth - camera_azimuth + 180.0
                    ) % 360.0 - 180.0
                    camera_azimuth += alpha * delta
                renderer._cam.azimuth = camera_azimuth % 360.0

            if cfg.camera_follow_shadow_light:
                assert tracked_entity is not None
                root_pos = tracked_entity.data.root_link_pos_w[
                    env_cfg.viewer.env_idx
                ]
                # The directional light's position still anchors MuJoCo's
                # finite shadow-map region. Keep that region centered on the
                # tracked robot without changing light direction or intensity.
                renderer._model.light_pos[:, 0] = float(root_pos[0])
                renderer._model.light_pos[:, 1] = float(root_pos[1])

            with torch.inference_mode():
                actions = policy(obs)
            obs, _, _, _ = env.step(actions)

            if cfg.contact_log_file is not None:
                render_model = renderer._model
                render_data = renderer._data
                contacts: list[dict[str, object]] = []
                for contact_index in range(render_data.ncon):
                    contact = render_data.contact[contact_index]
                    geom1 = render_model.geom(int(contact.geom1)).name
                    geom2 = render_model.geom(int(contact.geom2)).name
                    if not (
                        geom1.startswith("swing_frame_")
                        or geom2.startswith("swing_frame_")
                    ):
                        continue
                    contacts.append(
                        {
                            "geom1": geom1,
                            "geom2": geom2,
                            "distance_m": float(contact.dist),
                        }
                    )
                if contacts:
                    frame_contact_samples.append(
                        {
                            "step": video_step + 1,
                            "time_s": (video_step + 1) * base_env.step_dt,
                            "contacts": contacts,
                        }
                    )

        if cfg.contact_log_file is not None:
            contact_log_path = Path(cfg.contact_log_file)
            contact_log_path.parent.mkdir(parents=True, exist_ok=True)
            contact_log_path.write_text(
                json.dumps(frame_contact_samples, indent=2) + "\n"
            )

    # mjlab 1.3.0: ONNX export + metadata moved to mjlab.rl.exporter_utils and
    # the runner's built-in export_policy_to_onnx. Observation normalization is
    # baked into the exported graph automatically — EmpiricalNormalization is a
    # submodule of the policy's MLPModel (obs_normalization=True in RslRlModelCfg),
    # so export_policy_to_onnx emits actor(normalizer(obs)). No manual normalizer
    # handling needed (the old export_velocity_policy_as_onnx path is gone).
    from mjlab.rl.exporter_utils import attach_metadata_to_onnx, get_base_metadata

    onnx_path = os.path.abspath(cfg.onnx_file)
    path = os.path.dirname(onnx_path)
    filename = os.path.basename(onnx_path)

    runner.export_policy_to_onnx(path, filename)

    # Training steps through RslRlVecEnvWrapper, which clamps actor outputs.
    # Deployment runtimes consume ONNX outputs directly, so preserve that
    # behavior inside the graph rather than relying on each caller to know it.
    bake_action_clip(onnx_path, agent_cfg.clip_actions)

    metadata = get_base_metadata(runner.env.unwrapped, run_path=cfg.checkpoint_file)
    attach_metadata_to_onnx(onnx_path, metadata)

    print(f"Written {onnx_path}")

    env.close()


def main():
    # Parse first argument to choose the task.
    # Import tasks to populate the registry.
    import mjlab.tasks  # noqa: F401

    all_tasks = list_tasks()
    chosen_task, remaining_args = tyro.cli(
        tyro.extras.literal_type_from_choices(all_tasks),
        add_help=False,
        return_unknown_args=True,
    )

    # Parse the rest of the arguments + allow overriding env_cfg and agent_cfg.
    agent_cfg = load_rl_cfg(chosen_task)

    args = tyro.cli(
        ExportConfig,
        args=remaining_args,
        default=ExportConfig(),
        prog=sys.argv[0] + f" {chosen_task}",
        config=(
            tyro.conf.AvoidSubcommands,
            tyro.conf.FlagConversionOff,
        ),
    )
    del remaining_args, agent_cfg

    run_export(chosen_task, args)


if __name__ == "__main__":
    main()
