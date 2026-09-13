import argparse,json,hashlib,os,subprocess,sys,traceback
from pathlib import Path
from dataclasses import asdict
import numpy as np,torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.utils.torch import configure_torch_backends
import mjlab_microduck.tasks
from mjlab_microduck.tasks import MicroduckOnPolicyRunner,mdp
from mjlab_microduck.tasks.microduck_long_stair_env_cfg import make_long_stairs,LongStairRlCfg
p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--out',required=True);p.add_argument('--mode',choices=['eval','train','parity'],required=True);p.add_argument('--envs',type=int,default=64);p.add_argument('--iterations',type=int,default=5);p.add_argument('--seed',type=int,default=123);p.add_argument('--seconds',type=float,default=30.);p.add_argument('--cued',action='store_true');a=p.parse_args()
torch.set_num_threads(1);configure_torch_backends();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
cfg=make_long_stairs(play=a.mode!='train',blind=not a.cued);cfg.seed=a.seed;cfg.scene.num_envs=a.envs
if a.mode!='train':cfg.episode_length_s=a.seconds+1
agent=asdict(LongStairRlCfg);agent.update(logger='tensorboard',upload_model=False,save_interval=50,max_iterations=a.iterations,seed=a.seed);agent['algorithm'].update(learning_rate=5e-6,schedule='fixed');agent['algorithm']['symmetry_cfg']=None
raw=ManagerBasedRlEnv(cfg,device='cuda:0');env=RslRlVecEnvWrapper(raw,clip_actions=agent.get('clip_actions'));runner=MicroduckOnPolicyRunner(env,agent,str(out),'cuda:0')
saved=torch.load(a.source,map_location='cpu',weights_only=False);runner.load(a.source,map_location='cuda:0')
for key,v in saved['actor_state_dict'].items():assert torch.equal(v.cpu(),runner.alg.actor.state_dict()[key].cpu()),key
assert len(runner.alg.optimizer.state)==len(saved['optimizer_state_dict']['state'])
# Full policy/critic/Adam resume; new geometry starts a new curriculum/episode clock.
raw.common_step_counter=0;runner.current_learning_iteration=int(saved['iter'])+1
for g in runner.alg.optimizer.param_groups:g['lr']=5e-6
runner.alg.learning_rate=5e-6
raw.reset(seed=a.seed);obs=env.get_observations();assert obs['actor'].shape==(a.envs,61)
if not a.cued:assert torch.count_nonzero(obs['actor'][:,48:])==0
manifest={**vars(a),'source_sha256':hashlib.sha256(Path(a.source).read_bytes()).hexdigest(),'actor_dimensions':61,'zero_command_slots':not a.cued,'clock_or_stair_actor_inputs':a.cued,'riser_m':.0231,'angle_deg':62,'treads':64,'landings':0,'dt':raw.step_dt,'resume_iter':saved['iter'],'optimizer_entries':len(runner.alg.optimizer.state),'new_environment_counter':0,'lr':5e-6}
(out/'manifest.json').write_text(json.dumps(manifest,indent=2))
if a.mode=='train':
    runner.learn(num_learning_iterations=a.iterations,init_at_random_ep_len=True)
    runner.save(str(out/'final.pt'));(out/'done.json').write_text(json.dumps({'complete':True,'final':'final.pt'}));raw.close();sys.exit(0)
policy=runner.get_inference_policy(device='cuda:0')
if a.mode=='parity':
    import onnxruntime as ort
    import copy
    cpu_policy=copy.deepcopy(runner.alg.actor).cpu().eval()
    gpu_errors=[]
    path=out/'policy.onnx';sess=ort.InferenceSession(str(path),providers=['CPUExecutionProvider']);errors=[]
    for i in range(20):
        with torch.inference_mode():act=policy(obs)
        pred=sess.run(None,{sess.get_inputs()[0].name:obs['actor'][:1].cpu().numpy()})[0]
        with torch.inference_mode():reference=cpu_policy({k:v[:1].cpu() for k,v in obs.items()}).numpy()
        assert np.allclose(pred,reference,rtol=2e-5,atol=1e-4), (pred,reference)
        errors.append(float(np.max(np.abs(pred-reference))));gpu_errors.append(float(np.max(np.abs(pred-act[:1].cpu().numpy()))))
        obs,_,done,_=env.step(act)
        assert torch.count_nonzero(obs['actor'][:,48:])==0 and torch.isfinite(obs['actor']).all()
        if i==9:raw.reset(seed=a.seed+1);obs=env.get_observations()
    assert max(errors)<5e-4,errors
    (out/'parity.json').write_text(json.dumps({'max_error':max(errors),'reference':'CPU Torch with baked actor normalizer','tolerance':{'rtol':2e-5,'atol':1e-4,'absolute_cap':5e-4},'gpu_tf32_max_difference':max(gpu_errors),'reset_checked':True,'normalizer_baked':True}));raw.close();sys.exit(0)
alive=torch.ones(a.envs,dtype=torch.bool,device=raw.device);best=torch.full((a.envs,),-1,device=raw.device,dtype=torch.long);first=[None]*a.envs;traj=[];max_action=0.;nonfinite=False
for step in range(round(a.seconds/raw.step_dt)):
    with torch.inference_mode():act=policy(obs)
    if not a.cued:assert torch.count_nonzero(obs['actor'][:,48:])==0
    assert torch.isfinite(act).all();max_action=max(max_action,float(act.abs().max()))
    contacts=mdp._stair_contacts(raw);tread=contacts['foot_tread'].max(dim=1).values;best=torch.where(alive,torch.maximum(best,tread),best)
    robot=raw.scene['robot'];pos=robot.data.root_link_pos_w;finite=torch.isfinite(robot.data.joint_pos).all() & torch.isfinite(pos).all();nonfinite|=not bool(finite)
    if step%5==0:traj.append({'t':step*raw.step_dt,'root':pos[0].cpu().tolist(),'feet':contacts['foot_tread'][0].cpu().tolist(),'qpos':robot.data.joint_pos[0].cpu().tolist(),'alive':bool(alive[0])})
    obs,_,done,_=env.step(act);failed=alive & done.bool()
    terms={name:raw.termination_manager.get_term(name).clone() for name in cfg.terminations}
    for idx in failed.nonzero().flatten().tolist():first[idx]={'seconds':(step+1)*raw.step_dt,'terms':[n for n,v in terms.items() if bool(v[idx])],'highest_tread':int(best[idx])}
    alive &= ~done.bool()
result={'manifest':manifest,'survivors':int(alive.sum()),'trials':a.envs,'highest_treads':best.cpu().tolist(),'first_end':first,'nonfinite':nonfinite,'max_abs_action':max_action,'trajectory_env0':traj,'first_episode_only':True}
(out/'eval.json').write_text(json.dumps(result));assert not nonfinite;raw.close()
