import argparse,os,json,hashlib,types,copy
from pathlib import Path
from dataclasses import asdict
import torch,numpy as np
import mjlab_microduck.tasks
import geometry_patch
from mjlab_microduck.tasks import mdp,MicroduckOnPolicyRunner
from mjlab_microduck.tasks.microduck_desk_recovery_env_cfg import make_desk_recovery,DeskRecoveryRlCfg
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.managers import RewardTermCfg,TerminationTermCfg
P=Path(__file__).parent
pa=argparse.ArgumentParser();pa.add_argument('--out',required=True);pa.add_argument('--mode',default='train');pa.add_argument('--envs',type=int,default=64);pa.add_argument('--iterations',type=int,default=5);pa.add_argument('--cost',type=float,default=0);pa.add_argument('--checkpoint');pa.add_argument('--offset',type=int,default=0);a=pa.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1)
bankfile=P/('train-bank.npz' if a.mode=='train' else 'validation-bank.npz');bank=dict(np.load(bankfile));cfg=make_desk_recovery(play=a.mode!='train');geometry_patch.configure_cfg(cfg);cfg.scene.num_envs=a.envs;cfg.seed=273;cfg.auto_reset=a.mode=='train';cfg.episode_length_s=20.;cfg.events['reset_stair_ladder'].params.update(min_start_tread=0,max_start_tread=0,floor_spawn_prob=1.,desk_probability=0.)
original=cfg.events['reset_stair_ladder'].func

def reset_bank(env,env_ids,**kw):
 original(env,env_ids,**kw)
 ids=env_ids.to(env.device,dtype=torch.long);n=len(ids)
 pick=torch.randint(len(bank['qpos']),(n,),device=env.device) if a.mode=='train' else (ids+a.offset)%len(bank['qpos'])
 idx=pick.cpu().numpy();qp=torch.tensor(bank['qpos'][idx],device=env.device);qv=torch.tensor(bank['qvel'][idx],device=env.device);mp=torch.tensor(bank['mocap_pos'][idx],device=env.device);mq=torch.tensor(bank['mocap_quat'][idx],device=env.device)
 shift=env.sim.data.mocap_pos[ids,0]-mp[:,0];assert shift[:,2].abs().max()<1e-4
 qp[:,:3]+=shift;mp+=shift[:,None,:];env.sim.data.qpos[ids]=qp;env.sim.data.qvel[ids]=qv;env.sim.data.mocap_pos[ids]=mp;env.sim.data.mocap_quat[ids]=mq
 if not hasattr(env,'_local_anchor'):
  env._local_anchor=torch.zeros(env.num_envs,2,device=env.device);env._local_raw=torch.zeros(env.num_envs,14,device=env.device);env._local_exec=torch.zeros_like(env._local_raw)
 env._local_anchor[ids]=qp[:,:2];env._local_raw[ids]=torch.tensor(bank['actor'][idx,34:48],device=env.device);env._local_exec[ids]=torch.tensor(bank['previous_executed'][idx],device=env.device)
cfg.events['reset_stair_ladder'].func=reset_bank

def stand(env):return mdp.floor_desk_status(env)[-1].float()
def travel(env):return ((env.scene['robot'].data.root_link_pos_w[:,:2]-env._local_anchor).norm(dim=1)-.08).clamp(min=0).square()/.01
def failed(env):return env.scene['robot'].data.root_link_pos_w[:,2]<.40
# Positive goal requires actual supported upright low-velocity standing; lying pays zero.
cfg.rewards={'standing':RewardTermCfg(func=stand,weight=10.),'travel':RewardTermCfg(func=travel,weight=-a.cost)}
cfg.terminations={k:v for k,v in cfg.terminations.items() if k in ['nan_state','time_out']};cfg.terminations['below_table']=TerminationTermCfg(func=failed)
raw=ManagerBasedRlEnv(cfg,device='cuda:0');env=RslRlVecEnvWrapper(raw);robot=raw.scene['robot'];servo=mdp._servo_joint_ids(raw,robot);head,_=robot.find_joints('^(neck_pitch|head_pitch|head_yaw|head_roll)$');alpha=torch.full((14,),.7,device=raw.device);alpha[[i for i,j in enumerate(servo) if j in head]]=.5
baseg=[v.kp_scale.clone() for v in robot.actuators]
oldstep=env.step

def feedback(obs):
 obs['actor'][:,34:48]=raw._local_raw
 return obs

def filtered_step(actions):
 actions=actions.clone();executed=alpha*actions+(1-alpha)*raw._local_exec;raw._local_raw=actions;raw._local_exec=executed
 for v,g in zip(robot.actuators,baseg):v.kp_scale.copy_(g*.8)
 raw._manual_reset_pending.zero_();obs,reward,done,info=oldstep(executed)
 return feedback(obs),reward,done,info
env.step=filtered_step
agent=asdict(DeskRecoveryRlCfg);agent.update(logger='tensorboard',upload_model=False,seed=273,save_interval=128);agent['algorithm'].update(learning_rate=1e-6,schedule='fixed');agent['algorithm']['symmetry_cfg']=None
runner=MicroduckOnPolicyRunner(env,agent,str(out),'cuda:0')
if a.checkpoint:runner.load(a.checkpoint,map_location='cuda:0')
else:
 runner.alg.actor.load_state_dict(torch.load(P/'official-actor.pt',weights_only=False,map_location='cuda:0'),strict=True)
 assert len(runner.alg.optimizer.state)==0
normalizer=runner.alg.actor.obs_normalizer;before={k:v.clone() for k,v in normalizer.state_dict().items()};normalizer.update=types.MethodType(lambda self,x:None,normalizer)
raw.reset(seed=273);obs=feedback(env.get_observations());original_get_observations=env.get_observations;env.get_observations=lambda:feedback(original_get_observations())
manifest={'actor_source_sha256':hashlib.sha256((P/'alpha_stand.onnx').read_bytes()).hexdigest(),'bank_sha256':hashlib.sha256(bankfile.read_bytes()).hexdigest(),'fresh_critic_Adam':not bool(a.checkpoint),'cost':a.cost,'deadzone_m':.08,'gain':.8,'filter_head':.5,'filter_legs':.7,'history':'raw policy action','norm_frozen':True,'blind':True,'historical_BAM_DR_restored':False,'seed':273}
(out/'contract.json').write_text(json.dumps(manifest,indent=2))
def audit(module,args):
 x=args[0]['actor'];assert torch.isfinite(x).all() and torch.count_nonzero(x[:,48:])==0
h=runner.alg.actor.register_forward_pre_hook(audit)
if a.mode=='train':
 runner.learn(num_learning_iterations=a.iterations,init_at_random_ep_len=False)
 for k,v in before.items():assert torch.equal(v,normalizer.state_dict()[k]),k
 runner.save(str(out/'final.pt'));(out/'frozen-normalizer-check.json').write_text(json.dumps({'exact':True}));raw.close();raise SystemExit
policy=runner.get_inference_policy(device='cuda:0')
if a.mode=='parity':
 import onnxruntime as ort
 sess=ort.InferenceSession(str(out/'policy.onnx'),providers=['CPUExecutionProvider']);errors=[]
 for i in range(20):
  with torch.inference_mode():act=policy(obs)
  x=obs['actor'][:1].cpu().numpy();y=sess.run(None,{sess.get_inputs()[0].name:x})[0];errors.append(float(abs(y-act[:1].cpu().numpy()).max()));assert np.allclose(y,act[:1].cpu().numpy(),atol=5e-4,rtol=2e-5)
  obs,_,_,_=env.step(act)
  if i==9:raw.reset(seed=274);obs=feedback(env.get_observations())
 (out/'parity.json').write_text(json.dumps({'max_error':max(errors),'reset':True}));raw.close();raise SystemExit
records={k:[] for k in ['qpos','qvel','actor','action','executed','safe','drift']};first=[None]*a.envs
for t in range(1000):
 with torch.inference_mode():act=policy(obs)
 assert torch.isfinite(raw.sim.data.qpos).all() and torch.isfinite(raw.sim.data.qvel).all();assert raw.sim.data.xfrc_applied.abs().max()==0
 for k,v in [('qpos',raw.sim.data.qpos),('qvel',raw.sim.data.qvel),('actor',obs['actor']),('action',act),('executed',alpha*act+(1-alpha)*raw._local_exec),('safe',stand(raw)),('drift',(robot.data.root_link_pos_w[:,:2]-raw._local_anchor).norm(dim=1))]:records[k].append(v.detach().cpu().numpy().copy())
 obs,_,done,_=env.step(act)
 for i in range(a.envs):
  if bool(done[i]) and first[i] is None:first[i]=t*.02
np.savez_compressed(out/'rollout.npz',**{k:np.asarray(v) for k,v in records.items()});(out/'eval.json').write_text(json.dumps({'first_flags':first,'finite':True,'zero13':True,'wrench':0,'contract':manifest}));raw.close()
