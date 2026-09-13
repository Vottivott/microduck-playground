"""Validate geometry while distinguishing intentional rib interference from collisions."""
from pathlib import Path
import json,hashlib
import numpy as np,trimesh as tm
P=Path(__file__).parent;result={};count=0
for folder in ['friction-fit','friction-fit-3step']:
 p=P/folder;j=json.loads((p/'manifest.json').read_text());c=j['checks']
 assert c['socket_tread_intersection_mm3']<.01 and c['missing_tread_volume_mm3']<.1
 assert max(c['individual_tread_missing_from_single_module_mm3'])<.1
 for key in ['module_overlap_mm3','floor_module_overlap_mm3','module_insertion_unintended_overlap_mm3','floor_insertion_unintended_overlap_mm3','floor_pair_overlap_mm3']:assert max(c[key])<.1,(folder,key,c[key])
 assert all(0<x<10 for x in c['intentional_rib_interference_mm3']),c['intentional_rib_interference_mm3']
 assert not list(p.glob('*-pin-*'))
 for path in p.glob('*.stl'):
  m=tm.load(path);assert np.isfinite(m.vertices).all() and m.is_watertight and m.is_winding_consistent and m.volume>0,path;count+=1
 for name,part in j['parts'].items():
  m=tm.load(p/part['file']);assert len(m.split())==1,name;assert np.allclose(m.bounds,part['bounds_mm'],atol=.001),name
 for name in ['clamp-screw','pressure-pad','clamp-body--143','clamp-body-143','thread-fit-coupon']:assert (p/f'{name}.stl').read_bytes()==(P/'clamps'/f'{name}.stl').read_bytes(),name
 fits=json.loads((p/'bed-orientation.json').read_text())
 for name,fit in fits.items():
  m=tm.load(p/f'{name}-bed-oriented.stl');assert fit['geometric_fit_256cube'] and max(m.extents)<256,name;assert np.allclose(m.extents,fit['bounds_mm'],atol=.001),name
 result[folder]={'manufacturing_parts':len(j['parts']),'max_packed_axis_mm':max(max(x['bounds_mm'])for x in fits.values()),'checks':c}
result.update({'stl_count':count,'source_geometry_sha256':hashlib.sha256((P/'evaluated-geometry.json').read_bytes()).hexdigest(),'threaded_clamp_byte_identical':True,'physical_fit_retention_and_load_tested':False})
(P/'friction-validation.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items()if k not in ['friction-fit','friction-fit-3step']},indent=2))
