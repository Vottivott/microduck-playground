"""Rigid collision model of the printable floor-to-desk design; SI units."""
from pathlib import Path
import json,mujoco
from mjlab.entity import EntityCfg
from mjlab_microduck.robot.ladder import StairLadderGeometry
class FloorDeskGeometry(StairLadderGeometry):
    def tread_side(self,index):
        return 0 if index>=30 else super().tread_side(index)
GEOMETRY=FloorDeskGeometry(num_treads=32,tread_thickness_m=.006,rail_length_m=.82)
RECORDS=json.loads((Path(__file__).parent/'floor_desk_assets/geometry.json').read_text())
def spec_for(group):
    spec=mujoco.MjSpec();body=spec.worldbody.add_body(name='structure',mocap=True)
    for r in RECORDS:
        name=r['name']
        is_landing=name in ['landing_plate','landing_taper']
        selected=(group=='landing' and is_landing) or (group=='desk' and name=='desktop') or (group=='structure' and not name.startswith('tread_') and not is_landing and name!='desktop')
        if not selected:continue
        kind=getattr(mujoco.mjtGeom,'mjGEOM_'+r['type'].upper())
        kw=dict(name=name,type=kind,pos=r['pos'])
        if r.get('quat'):kw['quat']=r['quat']
        if r['type']=='mesh':
            mesh=spec.add_mesh(name=name+'_mesh');mesh.uservert=[x for v in r['vertices'] for x in v];kw['meshname']=name+'_mesh'
        else:kw['size']=r['size']+[0.]*(3-len(r['size']))
        g=body.add_geom(**kw);g.friction=[1,.005,.0001];g.solref=[.01,1];g.solimp=[.95,.99,.001,.5,2];g.priority=1;g.condim=3
        g.rgba=[.58,.73,.80,1] if ('bridge' in name or 'spine' in name or is_landing) else [.17,.30,.38,1]
        if name=='desktop':g.rgba=[.78,.68,.55,1]
    return spec

def structure_cfg(group):
    return EntityCfg(spec_fn=lambda:spec_for(group),init_state=EntityCfg.InitialStateCfg(pos=(0,0,-2)))
