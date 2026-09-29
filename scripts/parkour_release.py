"""Pinned simulation profiles for the elevated-jump research releases.

Examples (from repository root):
  uv run python scripts/parkour_release.py long-jump export --checkpoint-file checkpoint.pt --onnx-file policy.onnx
  uv run python scripts/parkour_release.py long-jump verify --checkpoint checkpoint.pt --onnx policy.onnx --output parity.json
  uv run python scripts/parkour_release.py backflip render --checkpoint checkpoint.pt --output rollout.mp4
  uv run python scripts/parkour_release.py backflip smoke

Checkpoints use pickle: load only files from trusted sources.
Profiles are reproducible evaluation settings, not a claim that an edited
montage is a deterministic replay. Do not use this script on real hardware.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys

PROFILES = {
    'long-jump': ('PlatformJump', {'MICRODUCK_PJ_GAP':'0.30', 'MICRODUCK_PJ_DROP':'0.25'}),
    'backflip': ('Flip', {'MICRODUCK_FLIP_DIR':'back', 'MICRODUCK_FLIP_H_MIN':'0.8',
        'MICRODUCK_FLIP_H_MAX':'0.8', 'MICRODUCK_FLIP_MAT_VISUAL':'0', 'MICRODUCK_FLIP_HOP_FATAL':'0'}),
}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('profile', choices=PROFILES)
    parser.add_argument('operation', choices=['export','smoke','verify','render','evaluate'])
    args, rest=parser.parse_known_args()
    name, settings=PROFILES[args.profile]
    # Prevent a shell's old experimental overrides silently changing this profile.
    for key in list(os.environ):
        if key.startswith(('MICRODUCK_PK_', 'MICRODUCK_PJ_', 'MICRODUCK_FLIP_', 'MICRODUCK_LJ_')):
            del os.environ[key]
    os.environ.update(settings)
    task='Mjlab-'+name+'-MicroDuck'
    if args.operation=='export':
        subprocess.run([sys.executable,'scripts/export.py',task,'--num-envs','1',*rest],check=True)
        return
    if args.operation=='smoke':
        subprocess.run([sys.executable,'-m','mjlab_microduck.train_cli',task,
            '--env.scene.num-envs','64','--agent.max-iterations','5',
            '--agent.logger','tensorboard','--agent.run-name','release-smoke-'+args.profile,*rest],check=True)
        return
    sub=argparse.ArgumentParser()
    sub.add_argument('--checkpoint',required=True)
    sub.add_argument('--onnx')
    sub.add_argument('--output',required=True)
    sub.add_argument('--seed',type=int,default=28)
    sub.add_argument('--num-envs',type=int,default=32)
    sub.add_argument('--duration',type=float,default=12.)
    cfg=sub.parse_args(rest)
    import numpy as np
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg,load_rl_cfg,load_runner_cls
    import mjlab_microduck.tasks  # noqa: F401
    env_cfg=load_env_cfg(task,play=True)
    env_cfg.scene.num_envs=cfg.num_envs if args.operation=='evaluate' else 1
    env_cfg.seed=cfg.seed
    env_cfg.episode_length_s=cfg.duration
    render=args.operation=='render'
    if render:
        from chimney_presentation import configure_video_cfg,fix_render_shadows,recolored_robot_cfg
        env_cfg.scene.entities['robot']=recolored_robot_cfg(env_cfg.scene.entities['robot'],'cream')
        configure_video_cfg(env_cfg,width=720,height=720)
        env_cfg.viewer.azimuth=145 if name=='Flip' else 110
        env_cfg.viewer.elevation=-12
        env_cfg.viewer.distance=1.7 if name=='Flip' else 2.5
    device='cuda:0' if torch.cuda.is_available() else 'cpu'
    raw=ManagerBasedRlEnv(env_cfg,device=device,render_mode='rgb_array' if render else None)
    agent=load_rl_cfg(task)
    env=RslRlVecEnvWrapper(raw,clip_actions=agent.clip_actions)
    runner=load_runner_cls(task)(env,asdict(agent),device=device)
    runner.load(cfg.checkpoint,map_location=device)
    actor=runner.get_inference_policy(device=device)
    term=raw.action_manager.get_term('joint_pos')
    lo,hi=term.raw_action_bounds()
    obs=env.get_observations()
    output=Path(cfg.output);output.parent.mkdir(parents=True,exist_ok=True)
    if args.operation=='verify':
        import onnx
        import onnxruntime as ort
        if not cfg.onnx: sub.error('--onnx is required for verify')
        onnx.checker.check_model(onnx.load(cfg.onnx))
        session=ort.InferenceSession(cfg.onnx,providers=['CPUExecutionProvider'])
        errors=[];rng=np.random.default_rng(cfg.seed)
        with torch.inference_mode():
            for i in range(200):
                inp=obs.clone()
                if i<100:
                    inp['actor']=torch.as_tensor(rng.normal(size=(1,61)).astype(np.float32),device=device)
                expected=actor(inp)
                if agent.clip_actions is not None: expected=expected.clamp(-agent.clip_actions,agent.clip_actions)
                expected=torch.clamp(expected,lo,hi)
                actual=session.run(None,{session.get_inputs()[0].name:inp['actor'].cpu().numpy()})[0]
                assert actual.shape==(1,14) and np.isfinite(actual).all()
                errors.append(float(np.abs(actual-expected.cpu().numpy()).max()))
                if i>=100: obs,_,_,_=env.step(torch.as_tensor(actual,device=device))
                assert torch.isfinite(raw.sim.data.qpos).all()
        assert max(errors)<1e-4, max(errors)
        result={'task':task,'profile':args.profile,'environment':settings,'observations':200,'live_steps':100,
            'max_abs_action_error':max(errors),'normalizer_baked_into_onnx':True,
            'joint_offset_rad':term._offset[0].cpu().tolist(),'action_scale':float(term._scale),
            'action_bounds_lo':lo.cpu().tolist(),'action_bounds_hi':hi.cpu().tolist(),
            'joint_names':list(raw.scene['robot'].joint_names),'input_name':session.get_inputs()[0].name,
            'output_name':session.get_outputs()[0].name,'observation_size':61,'action_size':14,
            'control_frequency_hz':50,'action_filter':'none'}
    else:
        active=torch.ones(raw.num_envs,dtype=torch.bool,device=device)
        best=torch.zeros(raw.num_envs,device=device)
        events=[];writer=None
        if render:
            import imageio.v2 as iio
            fix_render_shadows(raw,light_dir=(-0.42,0.11,-1.))
            writer=iio.get_writer(output,fps=50,codec='libx264',macro_block_size=None)
            writer.append_data(raw.render())
        with torch.inference_mode():
            for step in range(round(cfg.duration/raw.step_dt)):
                obs,_,done,_=env.step(actor(obs))
                assert torch.isfinite(raw.sim.data.qpos).all()
                if hasattr(raw,'_pk'):
                    best=torch.where(active,torch.maximum(best,raw._pk['link'].float()),best)
                ended=done.bool() & active
                for idx in ended.nonzero().flatten().tolist():
                    events.append({'env':idx,'time_s':(step+1)*raw.step_dt,
                        'terms':[n for n in raw.termination_manager.active_terms if bool(raw.termination_manager.get_term(n)[idx])]})
                active &= ~done.bool()
                if not active.any(): break
                if writer: writer.append_data(raw.render())
        if writer: writer.close()
        result={'task':task,'profile':args.profile,'seed':cfg.seed,'num_envs':raw.num_envs,
            'duration_s':cfg.duration,'first_episode_end_events':events,'best_link_lower_bound':best.cpu().tolist(),
            'note':'Small fixed-profile diagnostic, not a hardware or robustness validation. Terminal-step state may reset before the link diagnostic is read.'}
    target=output.with_suffix('.json') if render else output
    target.write_text(json.dumps(result,indent=2)+'\n')
    env.close()
    print('RELEASE_CHECK_OK',str(target))

if __name__=='__main__': main()
