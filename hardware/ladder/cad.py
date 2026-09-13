"""Shared CAD helpers for the current friction-fit ladder and threaded clamps; mm.
Prototype fit allowances are explicit, not a claim of physical load validation.
"""
from pathlib import Path
import json, math, hashlib, argparse
import numpy as np
import trimesh as tm
import geometry as evaluated
P=Path(__file__).parent
parser=argparse.ArgumentParser();parser.add_argument('--treads-per-module',type=int,choices=[3,6],default=6);args=parser.parse_args()
O=P/'clamps';O.mkdir(exist_ok=True)
GAP=.30  # per-face socket/thread clearance, mm

def box(size,pos):
 m=tm.creation.box(size);m.apply_translation(pos);return m

def cyl(r,h,z=0):
 m=tm.creation.cylinder(radius=r,height=h,sections=72);m.apply_translation([0,0,z]);return m

def moved(m,v):m=m.copy();m.apply_translation(v);return m

def union(ms):return tm.boolean.union(ms,engine='manifold')
def diff(a,b):return tm.boolean.difference([a,b],engine='manifold')
def intersect(a,b):return tm.boolean.intersection([a,b],engine='manifold')
def vol(m):return abs(m.volume) if m is not None else 0

def helix(length=66,clearance=0):
 # Right-handed 4mm pitch trapezoid; flat crest, broad root. End clipping
 # makes mating threads extend through the entire tapped jaw.
 pitch=4.;r0=6.6+clearance;r1=8.5+clearance
 vertices=[];faces=[];count=int((length+8)/pitch*96)+1
 for t in np.linspace(-4/pitch*2*np.pi,(length+4)/pitch*2*np.pi,count):
  z=pitch*t/(2*np.pi)
  for r,dz in [(r0,-1.45-clearance),(r1,-.45-clearance),(r1,.45+clearance),(r0,1.45+clearance)]:vertices.append([r*np.cos(t),r*np.sin(t),z+dz])
 for i in range(count-1):
  for k in range(4):a=4*i+k;b=4*i+(k+1)%4;faces.extend([[a,b,b+4],[a,b+4,a+4]])
 faces.extend([[0,2,1],[0,3,2],[4*(count-1),4*(count-1)+1,4*(count-1)+2],[4*(count-1),4*(count-1)+2,4*(count-1)+3]])
 ridge=tm.Trimesh(vertices,faces);ridge.fix_normals()
 return intersect(union([ridge,cyl(6.8+clearance,length+8,length/2)]),box([25,25,length],[0,0,length/2]))

manifest={'treads_per_module':args.treads_per_module,'units':'millimetres','fit_clearance_per_face_mm':GAP,'parts':{},'assembly':{},'checks':{}}
scenes={}
def export_checked(m,path):
 m=m.copy();m.vertices=np.round(m.vertices,4);m.merge_vertices(digits_vertex=4);m.update_faces(m.nondegenerate_faces());m.remove_unreferenced_vertices();m.export(path)
 reload=tm.load(path);assert reload.is_watertight and reload.is_winding_consistent,(str(path),'STL roundtrip')

def save(name,m,world=None):
 assert m.is_watertight and m.is_winding_consistent and m.volume>0,name
 assert len(m.split())==1,(name,len(m.split()))
 export=m.copy();export.apply_translation(-export.bounds[0]);export_checked(export,O/(name+'.stl'))
 manifest['parts'][name]={'file':name+'.stl','bounds_mm':export.bounds.tolist(),'volume_cm3':m.volume/1000,'watertight':True,'single_solid':True,'print_translation_mm':(-m.bounds[0]).tolist()}
 if world is not None:scenes[name]=world.copy()
 return m
