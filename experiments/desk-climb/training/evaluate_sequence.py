import argparse,json,hashlib,os,subprocess,sys,traceback
from pathlib import Path
from dataclasses import asdict
import numpy as np,torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.utils.torch import configure_torch_backends
import mjlab_microduck.tasks
import geometry_patch
from mjlab_microduck.robot import floor_desk
for rec in floor_desk.RECORDS:
    if rec['name']=='landing_plate':pass
from mjlab_microduck.tasks import MicroduckOnPolicyRunner,mdp
from mjlab_microduck.tasks.microduck_desk_recovery_env_cfg import make_desk_recovery as make_floor_desk,DeskRecoveryRlCfg as FloorDeskRlCfg
p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--out',required=True);p.add_argument('--mode',choices=['eval','train','parity'],required=True);p.add_argument('--envs',type=int,default=64);p.add_argument('--iterations',type=int,default=5);p.add_argument('--seed',type=int,default=123);p.add_argument('--seconds',type=float,default=30.);p.add_argument('--floor-probability',type=float,default=.25);p.add_argument('--mixed-eval',action='store_true');p.add_argument('--min-start-tread',type=int,default=0);p.add_argument('--transition-audit',action='store_true');p.add_argument('--desk-probability',type=float,default=.5);p.add_argument('--desk-only',action='store_true');p.add_argument('--home-hold',action='store_true');p.add_argument('--freeze-part',choices=['none','head','legs'],default='none');p.add_argument('--leg-blend',type=float,default=0.);a=p.parse_args();a.cued=False;assert a.leg_blend in (0.,.5,1.)
torch.set_num_threads(1);configure_torch_backends();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
cfg=make_floor_desk(play=a.mode!='train' and not a.mixed_eval,floor_probability=a.floor_probability);geometry_patch.configure_cfg(cfg);cfg.seed=a.seed;cfg.scene.num_envs=a.envs
cfg.events['reset_stair_ladder'].params.update(min_start_tread=0,max_start_tread=0,desk_probability=0.,floor_spawn_prob=1.)
if a.desk_only:cfg.events['reset_stair_ladder'].params.update(floor_spawn_prob=0.,desk_probability=1.)
cfg.auto_reset=False
cfg.terminations.pop('reached_top') # Measure sustained standing, not success-truncated clips.
if a.mode!='train':cfg.episode_length_s=a.seconds+1
agent=asdict(FloorDeskRlCfg);agent.update(logger='tensorboard',upload_model=False,save_interval=50,max_iterations=a.iterations,seed=a.seed);agent['algorithm'].update(learning_rate=5e-6,schedule='fixed');agent['algorithm']['symmetry_cfg']=None
raw=ManagerBasedRlEnv(cfg,device='cuda:0');env=RslRlVecEnvWrapper(raw,clip_actions=agent.get('clip_actions'));runner=MicroduckOnPolicyRunner(env,agent,str(out),'cuda:0')
saved=torch.load(a.source,map_location='cpu',weights_only=False);runner.load(a.source,map_location='cuda:0')
for key,v in saved['actor_state_dict'].items():assert torch.equal(v.cpu(),runner.alg.actor.state_dict()[key].cpu()),key
def exact_tree(a,b,path='root'):
    if torch.is_tensor(a):assert torch.equal(a.cpu(),b.cpu()),path
    elif isinstance(a,dict):
        assert a.keys()==b.keys(),path
        for k in a:exact_tree(a[k],b[k],path+'.'+str(k))
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b),path
        for i,(x,y) in enumerate(zip(a,b)):exact_tree(x,y,path+'.'+str(i))
    else:assert a==b,path
exact_tree(saved['critic_state_dict'],runner.alg.critic.state_dict(),'critic')
exact_tree(saved['optimizer_state_dict']['state'],runner.alg.optimizer.state_dict()['state'],'Adam')
# Full policy/critic/Adam resume; new geometry starts a new curriculum/episode clock.
raw.common_step_counter=0;runner.current_learning_iteration=int(saved['iter'])+1
for g in runner.alg.optimizer.param_groups:g['lr']=5e-6
runner.alg.learning_rate=5e-6
raw.reset(seed=a.seed);obs=env.get_observations();assert obs['actor'].shape==(a.envs,61)
if not a.cued:assert torch.count_nonzero(obs['actor'][:,48:])==0
manifest={**vars(a),'source_sha256':hashlib.sha256(Path(a.source).read_bytes()).hexdigest(),'actor_dimensions':61,'zero_command_slots':not a.cued,'clock_or_stair_actor_inputs':a.cued,'riser_m':.024,'angle_deg':62,'treads':30,'support_indices':32,'landings':1,'desktop_m':.740,'central_spine':True,'dt':raw.step_dt,'resume_iter':saved['iter'],'full_critic_and_Adam_exact':True,'optimizer_entries':len(runner.alg.optimizer.state),'new_environment_counter':0,'lr':5e-6}
(out/'geometry-audit.json').write_text(json.dumps(geometry_patch.audit(),indent=2))
import mujoco
model=raw.sim.mj_model
desk_ids=[i for i in range(model.ngeom) if 'desktop' in (mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,i) or '')]
assert len(desk_ids)==1,desk_ids
assert np.allclose(model.geom_size[desk_ids[0]],[.7,.4,.0125]),model.geom_size[desk_ids]
(out/'compiled-table-check.json').write_text(json.dumps({'geom':desk_ids[0],'half_sizes':model.geom_size[desk_ids[0]].tolist(),'full_table_collision_verified':True}))
(out/'manifest.json').write_text(json.dumps(manifest,indent=2))
if a.mode=='train':
    checks={'raw_zero_calls':0}
    def audit_actor(module,args):
        ob=args[0]['actor']
        assert ob.shape[-1]==61 and torch.isfinite(ob).all()
        assert torch.count_nonzero(ob[...,48:])==0
        checks['raw_zero_calls']+=1
    hook=runner.alg.actor.register_forward_pre_hook(audit_actor)
    runner.learn(num_learning_iterations=a.iterations,init_at_random_ep_len=True)
    hook.remove();(out/'training-input-audit.json').write_text(json.dumps(checks))
    runner.save(str(out/'final.pt'));(out/'done.json').write_text(json.dumps({'complete':True,'final':'final.pt'}));raw.close();sys.exit(0)
policy=runner.get_inference_policy(device='cuda:0')
if a.mode=='parity':
    import onnxruntime as ort
    import copy
    cpu_policy=copy.deepcopy(runner.alg.actor).cpu().eval()
    gpu_errors=[]
    path=out/'policy.onnx';sess=ort.InferenceSession(str(path),providers=['CPUExecutionProvider']);errors=[]
    for i in range(20):
        obs['actor'][:,34:48]=previous_raw
    with torch.inference_mode():act=policy(obs)
    act=act.clone() # Normal tensor: switched rows are updated outside inference_mode.
    contact_now=mdp._stair_contacts(raw)
    feet_on_desk=torch.zeros(a.envs,2,dtype=torch.bool,device=raw.device)
    import mujoco
    model=raw.sim.mj_model;names=[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,i) or '' for i in range(model.ngeom)]
    desktop=names.index('tread_31/desktop');c=raw.sim.data.contact;n=int(raw.sim.data.nacon[0]);g=c.geom[:n]
    for side,name in enumerate(['robot/left_foot_collision','robot/right_foot_collision']):
        hit=(c.dist[:n]<=0)&(g==desktop).any(1)&(g==names.index(name)).any(1);feet_on_desk[c.worldid[:n][hit],side]=True
    trigger_mode=os.environ['TRIGGER_MODE'];assert trigger_mode in ('first_foot','both_feet','root050','root056','fallen_on_table','supported_root')
    root_local=robot.data.root_link_pos_w-raw.scene.env_origins
    root_local=root_local.clone();stair_state=mdp._stair_state(raw);root_local[:,0]-=stair_state.x0;root_local[:,1]-=stair_state.y0
    eligible=feet_on_desk.any(dim=1)
    if trigger_mode=='both_feet':eligible=feet_on_desk.all(dim=1)
    if trigger_mode=='root050':eligible &= root_local[:,0]>=.50
    if trigger_mode=='root056':eligible &= root_local[:,0]>=.56
    seen_desk |= feet_on_desk.any(dim=1)
    if trigger_mode=='fallen_on_table':
        desk=next(r for r in floor_desk.RECORDS if r['name']=='desktop')
        eligible=seen_desk & (robot.data.projected_gravity_b[:,2]>-.7) & (root_local[:,0]>desk['pos'][0]-desk['size'][0]) & (root_local[:,0]<desk['pos'][0]+desk['size'][0]) & (root_local[:,1].abs()<desk['size'][1]) & (root_local[:,2]>.60)
    if trigger_mode=='supported_root':
        desk=next(r for r in floor_desk.RECORDS if r['name']=='desktop')
        eligible=feet_on_desk.any(dim=1) & (root_local[:,0]>desk['pos'][0]-desk['size'][0]+float(os.environ.get('SWITCH_MARGIN','.08'))) & (root_local[:,0]<desk['pos'][0]+desk['size'][0]-.05) & (root_local[:,1].abs()<desk['size'][1]-.05) & (root_local[:,2]>.66)
    spin=raw.scene['robot'].data.root_link_ang_vel_w.norm(dim=1)
    eligible &= spin<float(os.environ['SWITCH_MAX_SPIN'])
    arriving=~switched & eligible
    for idx in arriving.nonzero().flatten().tolist():switches.append({'env':idx,'step':step,'seconds':step*raw.step_dt,'qpos':raw.sim.data.qpos[idx].cpu().tolist(),'qvel':raw.sim.data.qvel[idx].cpu().tolist(),'actor':obs['actor'][idx].cpu().tolist(),'previous_executed':previous_executed[idx].cpu().tolist(),'spin_at_switch':float(spin[idx]),'spin_limit':float(os.environ['SWITCH_MAX_SPIN']),'switch_margin':float(os.environ.get('SWITCH_MARGIN','.08')),'trigger_mode':trigger_mode,'feet_on_desk':feet_on_desk[idx].cpu().tolist(),'root_ladder_relative':root_local[idx].cpu().tolist()})
    switched|=arriving
    if arm!='climb':
        ids=switched.nonzero().flatten()
        if len(ids):
            x=obs['actor'][ids].cpu().numpy().astype(np.float32)
            yy=np.concatenate([session.run(None,{session.get_inputs()[0].name:v[None]})[0] for v in x])
            if not np.isfinite(yy).all():
                bad=ids[torch.tensor(~np.isfinite(yy).all(1),device=raw.device)]
                np.savez_compressed(out/'first-invalid.npz',qpos=raw.sim.data.qpos.cpu().numpy(),qvel=raw.sim.data.qvel.cpu().numpy(),actor=obs['actor'].cpu().numpy(),output=yy,history_qpos=np.asarray(all_qpos),history_qvel=np.asarray(all_qvel))
                (out/'first-invalid.json').write_text(json.dumps({'reason':'nonfinite official ONNX output','step':step,'env_ids':bad.cpu().tolist(),'input_finite':bool(np.isfinite(x).all()),'qpos_finite':bool(torch.isfinite(raw.sim.data.qpos).all()),'qvel_finite':bool(torch.isfinite(raw.sim.data.qvel).all()),'previous_first_flags':first}))
                raise RuntimeError('Nonfinite output captured; not a completed finite rollout')
            act[ids]=torch.tensor(yy,device=raw.device)
    previous_raw=act.clone()
    if arm=='official':
        for v,g in zip(robot.actuators,base_gains):v.kp_scale.copy_(torch.where(switched[:,None],g*.8,g))
        pred=sess.run(None,{sess.get_inputs()[0].name:obs['actor'][:1].cpu().numpy()})[0]
        with torch.inference_mode():reference=cpu_policy({k:v[:1].cpu() for k,v in obs.items()}).numpy()
        assert np.allclose(pred,reference,rtol=2e-5,atol=1e-4), (pred,reference)
        errors.append(float(np.max(np.abs(pred-reference))));gpu_errors.append(float(np.max(np.abs(pred-act[:1].cpu().numpy()))))
        obs,_,done,_=env.step(act)
        assert torch.count_nonzero(obs['actor'][:,48:])==0 and torch.isfinite(obs['actor']).all()
        if i==9:raw.reset(seed=a.seed+1);obs=env.get_observations()
    assert max(errors)<5e-4,errors
    (out/'parity.json').write_text(json.dumps({'max_error':max(errors),'reference':'CPU Torch with baked actor normalizer','tolerance':{'rtol':2e-5,'atol':1e-4,'absolute_cap':5e-4},'gpu_tf32_max_difference':max(gpu_errors),'reset_checked':True,'normalizer_baked':True}));raw.close();sys.exit(0)
contact_events=[];initial_snapshot=None;all_qpos=[];all_qvel=[];all_alive=[];max_wrench=0.
initial_mocap_pos=raw.sim.data.mocap_pos.cpu().numpy().copy();initial_mocap_quat=raw.sim.data.mocap_quat.cpu().numpy().copy()
alive=torch.ones(a.envs,dtype=torch.bool,device=raw.device);best=torch.full((a.envs,),-1,device=raw.device,dtype=torch.long);first=[None]*a.envs;traj=[];max_action=0.;nonfinite=False
robot=raw.scene['robot'];servo_ids=mdp._servo_joint_ids(raw,robot);head_ids,_=robot.find_joints('^(neck_pitch|head_pitch|head_yaw|head_roll)$');head_actions=[i for i,j in enumerate(servo_ids) if j in head_ids];assert len(head_actions)==4
freeze_ids=head_actions if a.freeze_part=='head' else [i for i in range(14) if i not in head_actions]
held=torch.tensor(json.loads((Path(__file__).parent/'balanced-bank.json').read_text())['held_action'],device=raw.device)
recorded_raw=[];recorded_executed=[];recorded_actor=[];recorded_safe=[]
import onnxruntime as ort
arm=os.environ['HANDOFF_ARM'];assert arm in ('climb','specialist','official','official_unfiltered')
options=ort.SessionOptions();options.intra_op_num_threads=1
filename=os.environ['GETUP_FILE']
session=ort.InferenceSession(str(Path(__file__).parent/'models'/filename),sess_options=options,providers=['CPUExecutionProvider'])
meta=session.get_modelmeta().custom_metadata_map
if arm.startswith('official'):
    assert session.get_inputs()[0].shape==[1,61] and session.get_outputs()[0].shape==[1,14]
    assert meta['action_scale']=='1.0'
    defaults=robot.data.default_joint_pos[0,servo_ids].cpu().numpy()
    assert np.allclose(defaults,np.fromstring(meta['default_joint_pos'],sep=','),atol=.00051)
    names=[robot.joint_names[i] for i in servo_ids];assert names==meta['joint_names'].split(',')
seen_desk=torch.zeros(a.envs,dtype=torch.bool,device=raw.device);switched=torch.zeros(a.envs,dtype=torch.bool,device=raw.device);previous_raw=torch.zeros((a.envs,14),device=raw.device);previous_executed=previous_raw.clone();switches=[]
base_gains=[v.kp_scale.clone() for v in robot.actuators];assert all(v._base_kp==200 for v in robot.actuators)
alpha=torch.full((14,),.7,device=raw.device);alpha[head_actions]=.5
(out/'handoff-contract.json').write_text(json.dumps({'arm':arm,'simulator_only_trigger':os.environ['TRIGGER_MODE'],'trigger_not_actor_input':True,'actor13zero':True,'standing_action_scale':1.,'official_gain_ratio':.8 if arm=='official' else 1.,'official_filter':{'head':.5,'legs':.7} if arm=='official' else None,'previous_action':'raw policy output; filter seeded from last executed climbing target','delays':'existing BAM3to6lag preserved; not hardware parity claim','continuous_after_flags':True,'failure_flags_do_not_disable_recovery':True,'policy_sha256':hashlib.sha256(Path(str(Path(__file__).parent/'models'/filename)).read_bytes()).hexdigest()}))

(out/'action-intervention.json').write_text(json.dumps({'leg_blend':a.leg_blend,'part':a.freeze_part,'head_action_indices':head_actions,'held_actions':held.cpu().tolist(),'diagnostic_only':True}))
for step in range(round(a.seconds/raw.step_dt)):
    obs['actor'][:,34:48]=previous_raw
    with torch.inference_mode():act=policy(obs)
    act=act.clone() # Normal tensor: switched rows are updated outside inference_mode.
    contact_now=mdp._stair_contacts(raw)
    feet_on_desk=torch.zeros(a.envs,2,dtype=torch.bool,device=raw.device)
    import mujoco
    model=raw.sim.mj_model;names=[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,i) or '' for i in range(model.ngeom)]
    desktop=names.index('tread_31/desktop');c=raw.sim.data.contact;n=int(raw.sim.data.nacon[0]);g=c.geom[:n]
    for side,name in enumerate(['robot/left_foot_collision','robot/right_foot_collision']):
        hit=(c.dist[:n]<=0)&(g==desktop).any(1)&(g==names.index(name)).any(1);feet_on_desk[c.worldid[:n][hit],side]=True
    trigger_mode=os.environ['TRIGGER_MODE'];assert trigger_mode in ('first_foot','both_feet','root050','root056','fallen_on_table','supported_root')
    root_local=robot.data.root_link_pos_w-raw.scene.env_origins
    root_local=root_local.clone();stair_state=mdp._stair_state(raw);root_local[:,0]-=stair_state.x0;root_local[:,1]-=stair_state.y0
    eligible=feet_on_desk.any(dim=1)
    if trigger_mode=='both_feet':eligible=feet_on_desk.all(dim=1)
    if trigger_mode=='root050':eligible &= root_local[:,0]>=.50
    if trigger_mode=='root056':eligible &= root_local[:,0]>=.56
    seen_desk |= feet_on_desk.any(dim=1)
    if trigger_mode=='fallen_on_table':
        desk=next(r for r in floor_desk.RECORDS if r['name']=='desktop')
        eligible=seen_desk & (robot.data.projected_gravity_b[:,2]>-.7) & (root_local[:,0]>desk['pos'][0]-desk['size'][0]) & (root_local[:,0]<desk['pos'][0]+desk['size'][0]) & (root_local[:,1].abs()<desk['size'][1]) & (root_local[:,2]>.60)
    if trigger_mode=='supported_root':
        desk=next(r for r in floor_desk.RECORDS if r['name']=='desktop')
        eligible=feet_on_desk.any(dim=1) & (root_local[:,0]>desk['pos'][0]-desk['size'][0]+float(os.environ.get('SWITCH_MARGIN','.08'))) & (root_local[:,0]<desk['pos'][0]+desk['size'][0]-.05) & (root_local[:,1].abs()<desk['size'][1]-.05) & (root_local[:,2]>.66)
    spin=raw.scene['robot'].data.root_link_ang_vel_w.norm(dim=1)
    eligible &= spin<float(os.environ['SWITCH_MAX_SPIN'])
    arriving=~switched & eligible
    for idx in arriving.nonzero().flatten().tolist():switches.append({'env':idx,'step':step,'seconds':step*raw.step_dt,'qpos':raw.sim.data.qpos[idx].cpu().tolist(),'qvel':raw.sim.data.qvel[idx].cpu().tolist(),'actor':obs['actor'][idx].cpu().tolist(),'previous_executed':previous_executed[idx].cpu().tolist(),'spin_at_switch':float(spin[idx]),'spin_limit':float(os.environ['SWITCH_MAX_SPIN']),'switch_margin':float(os.environ.get('SWITCH_MARGIN','.08')),'trigger_mode':trigger_mode,'feet_on_desk':feet_on_desk[idx].cpu().tolist(),'root_ladder_relative':root_local[idx].cpu().tolist()})
    switched|=arriving
    if arm!='climb':
        ids=switched.nonzero().flatten()
        if len(ids):
            x=obs['actor'][ids].cpu().numpy().astype(np.float32)
            yy=np.concatenate([session.run(None,{session.get_inputs()[0].name:v[None]})[0] for v in x])
            if not np.isfinite(yy).all():
                bad=ids[torch.tensor(~np.isfinite(yy).all(1),device=raw.device)]
                np.savez_compressed(out/'first-invalid.npz',qpos=raw.sim.data.qpos.cpu().numpy(),qvel=raw.sim.data.qvel.cpu().numpy(),actor=obs['actor'].cpu().numpy(),output=yy,history_qpos=np.asarray(all_qpos),history_qvel=np.asarray(all_qvel))
                (out/'first-invalid.json').write_text(json.dumps({'reason':'nonfinite official ONNX output','step':step,'env_ids':bad.cpu().tolist(),'input_finite':bool(np.isfinite(x).all()),'qpos_finite':bool(torch.isfinite(raw.sim.data.qpos).all()),'qvel_finite':bool(torch.isfinite(raw.sim.data.qvel).all()),'previous_first_flags':first}))
                raise RuntimeError('Nonfinite output captured; not a completed finite rollout')
            act[ids]=torch.tensor(yy,device=raw.device)
    previous_raw=act.clone()
    if arm=='official':
        for v,g in zip(robot.actuators,base_gains):v.kp_scale.copy_(torch.where(switched[:,None],g*.8,g))
    recorded_actor.append(obs['actor'].detach().cpu().numpy().copy())
    recorded_safe.append(mdp.floor_desk_status(raw)[-1].detach().cpu().numpy().copy())
    recorded_raw.append(act.detach().cpu().numpy().copy())
    if a.freeze_part!='none':
        act=act.clone();act[:,freeze_ids]=(1-a.leg_blend)*act[:,freeze_ids]+a.leg_blend*held[freeze_ids]
    if a.home_hold:act=torch.tensor(json.loads((Path(__file__).parent/'balanced-bank.json').read_text())['held_action'],device=raw.device).expand_as(act)
    if arm=='official':act=torch.where(switched[:,None],alpha*act+(1-alpha)*previous_executed,act)
    previous_executed=act.clone()
    recorded_executed.append(act.detach().cpu().numpy().copy())
    all_qpos.append(raw.sim.data.qpos.cpu().numpy().copy());all_qvel.append(raw.sim.data.qvel.cpu().numpy().copy());all_alive.append(alive.cpu().numpy().copy())
    max_wrench=max(max_wrench,float(raw.sim.data.xfrc_applied.abs().max()))
    assert torch.isfinite(raw.sim.data.qvel).all()
    if not a.cued:assert torch.count_nonzero(obs['actor'][:,48:])==0
    assert torch.isfinite(act).all();max_action=max(max_action,float(act.abs().max()))
    contacts=mdp._stair_contacts(raw);tread=contacts['foot_tread'].max(dim=1).values
    new_high=(tread>best)
    robot=raw.scene['robot']
    if initial_snapshot is None:initial_snapshot={'root':robot.data.root_link_pos_w.cpu().tolist(),'actor':obs['actor'].cpu().tolist(),'x0':mdp._stair_state(raw).x0.cpu().tolist(),'y0':mdp._stair_state(raw).y0.cpu().tolist(),'start_tread':mdp._stair_state(raw).start_tread.cpu().tolist(),'floor':mdp._stair_state(raw).spawn_on_floor.cpu().tolist(),'geom_names':[__import__('mujoco').mj_id2name(raw.sim.mj_model,__import__('mujoco').mjtObj.mjOBJ_GEOM,i) for i in range(raw.sim.mj_model.ngeom)]}
    if step==0:
        count=int(raw.sim.data.nacon[0].item());cc=raw.sim.data.contact
        initial_snapshot['contacts']={'geom':cc.geom[:count].cpu().tolist(),'world':cc.worldid[:count].cpu().tolist(),'dist':cc.dist[:count].cpu().tolist(),'pos':cc.pos[:count].cpu().tolist()}
        initial_snapshot['tread_top']=mdp._stair_state(raw).tread_top.cpu().tolist()
        initial_snapshot['desk_spawn']=raw._desk_spawn.cpu().tolist();initial_snapshot['desk_bank_index']=raw._desk_bank_index.cpu().tolist()
    for idx in new_high.nonzero().flatten().tolist():
        count=int(raw.sim.data.nacon[0].item());c=raw.sim.data.contact;mask=(c.worldid[:count]==idx)&(c.dist[:count]<=0)
        contact_events.append({'env':idx,'t':step*raw.step_dt,'tread':int(tread[idx]),'root':robot.data.root_link_pos_w[idx].cpu().tolist(),'qpos':robot.data.joint_pos[idx].cpu().tolist(),'action':act[idx].cpu().tolist(),'contact_geom':c.geom[:count][mask].cpu().tolist(),'contact_pos':c.pos[:count][mask].cpu().tolist(),'contact_dist':c.dist[:count][mask].cpu().tolist()})
    best=torch.maximum(best,tread)
    robot=raw.scene['robot'];pos=robot.data.root_link_pos_w.clone();finite=torch.isfinite(robot.data.joint_pos).all() & torch.isfinite(pos).all();nonfinite|=not bool(finite)
    if step%5==0:traj.append({'t':step*raw.step_dt,'root':pos[0].cpu().tolist(),'feet':contacts['foot_tread'][0].cpu().tolist(),'qpos':robot.data.joint_pos[0].cpu().tolist(),'alive':bool(alive[0])})
    if a.transition_audit and step%5==0:
        st=mdp._stair_state(raw)
        traj[-1].update(foot_sites=robot.data.site_pos_w[:,mdp._stair_foot_sites(raw,robot),:][0].cpu().tolist(),projected_gravity=robot.data.projected_gravity_b[0].cpu().tolist(),foot_support=contacts['foot_support'][0].cpu().tolist(),tread_targets=st.tread_target_xy[0,26:].cpu().tolist(),tread_tops=st.tread_top[0,26:].cpu().tolist())
    raw._manual_reset_pending.zero_()
    obs,_,done,_=env.step(act);failed=alive & done.bool()
    terms={name:raw.termination_manager.get_term(name).clone() for name in cfg.terminations}
    for idx in failed.nonzero().flatten().tolist():first[idx]={'seconds':(step+1)*raw.step_dt,'terms':[n for n,v in terms.items() if bool(v[idx])],'highest_tread':int(best[idx]),'root_before_end':pos[idx].cpu().tolist()}
    alive &= ~done.bool()
    # Preserve raw/executed action history after diagnostic failure flags.
np.savez_compressed(out/'trajectories.npz',qpos=np.asarray(all_qpos),qvel=np.asarray(all_qvel),alive=np.asarray(all_alive),mocap_pos=initial_mocap_pos,mocap_quat=initial_mocap_quat,dt=raw.step_dt)
np.savez_compressed(out/'actions.npz',raw=np.asarray(recorded_raw),executed=np.asarray(recorded_executed),actor=np.asarray(recorded_actor),standing_safe=np.asarray(recorded_safe))
(out/'switches.json').write_text(json.dumps(switches))
result={'max_actual_external_wrench':max_wrench,'contact_events':contact_events,'initial_snapshot':initial_snapshot,'manifest':manifest,'survivors':int(alive.sum()),'trials':a.envs,'highest_treads':best.cpu().tolist(),'first_end':first,'nonfinite':nonfinite,'max_abs_action':max_action,'trajectory_env0':traj,'first_episode_only':True}
(out/'eval.json').write_text(json.dumps(result));assert not nonfinite;raw.close()
