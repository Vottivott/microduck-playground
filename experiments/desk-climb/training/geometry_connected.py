from geometry_previous import *
import geometry_previous as previous
assert ROLE=='above'
removed_names=[r['name'] for r in floor_desk.RECORDS if r['name']=='landing_crossmember' or r['name'].startswith('landing_support')]
records=[r for r in floor_desk.RECORDS if r['name'] not in removed_names]
# Solid side cheeks overlap both the retained rails and the lowered clamp receivers.
# Kept at the outer side rails, outside the central foot passage.
profile=[(.357,.671),(.391,.697),(.432,.666),(.432,.637),(.394,.637)]
for side in [-.143,.143]:
 records.append({'name':f'connected_side_gusset_{round(side*1000)}','type':'mesh','pos':[0,0,0],'vertices':[[x,side+dy,z] for dy in [-.012,.012] for x,z in profile]})
floor_desk.RECORDS[:]=records

def audit():
 return {**previous.audit(),'records':records,'removed_floating_parts':removed_names,'connected_side_gusset_profile_xz':profile,'side_gusset_width_m':.024,'physical_collision_geometry':True}
