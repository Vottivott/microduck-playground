"""Render actual printable meshes, assembled and exploded, in Blender Cycles.
blender -b --python render_clamps.py
"""
import bpy,math,json,os
from pathlib import Path
from mathutils import Vector
P=Path(__file__).parent;O=P/'clamps';R=P/'renders';manifest=json.loads((O/'manifest.json').read_text())

def material(name,rgb):
 m=bpy.data.materials.new(name);m.diffuse_color=(*rgb,1);m.use_nodes=True;n=m.node_tree.nodes['Principled BSDF'];n.inputs['Base Color'].default_value=(*rgb,1);n.inputs['Roughness'].default_value=.36;return m

def render(name):
 bpy.ops.wm.read_factory_settings(use_empty=True)
 oak=material('Warm oak printed polymer',(.54,.34,.17));walnut=material('Walnut printed polymer',(.19,.09,.035));accent=material('Amber printed screw and locks',(.78,.38,.085));pad=material('Cream printed pressure pad',(.81,.77,.63));ground=material('Studio',(.73,.76,.79))
 objects=[]
 def add(file,mat,shift=(0,0,0)):
  bpy.ops.wm.stl_import(filepath=str(O/file));o=bpy.context.object;o.scale=(.001,)*3;o.location=Vector(shift)*.001;o.data.materials.append(mat);objects.append(o);return o
 if name.startswith('clamp'):
  add('clamp-body--143-assembly.stl',walnut)
  add('screw--143-assembly.stl',accent,(0,0,-80 if 'exploded' in name else 0))
  add('pad--143-assembly.stl',pad,(0,-65,20) if 'exploded' in name else (0,0,0))
  if 'assembled' in name:
   bpy.ops.mesh.primitive_cube_add(size=1,location=(.484667,-.12175,.6475));o=bpy.context.object;o.scale=(.10,.0425,.025);o.data.materials.append(oak);objects.append(o)
 else:raise ValueError(name)
 bpy.context.view_layer.update();pts=[o.matrix_world@Vector(v)for o in objects for v in o.bound_box];lo=Vector([min(v[i]for v in pts)for i in range(3)]);hi=Vector([max(v[i]for v in pts)for i in range(3)]);center=(lo+hi)/2;size=max(hi-lo)
 bpy.ops.mesh.primitive_plane_add(size=size*200,location=(center.x,center.y,lo.z-.002));bpy.context.object.data.materials.append(ground)
 sc=bpy.context.scene;sc.render.engine='CYCLES';sc.cycles.samples=64;sc.cycles.use_denoising=True;sc.render.threads_mode='FIXED';sc.render.threads=6;sc.world=bpy.data.worlds.new('Studio');sc.world.color=(.25,.25,.25)
 for pos,power,area in [((-1,-2,3),550,2),((2,1,2),400,1.5),((-2,2,1),260,1.5)]:
  bpy.ops.object.light_add(type='AREA',location=center+Vector(pos)*size);o=bpy.context.object;o.data.energy=power*size*size;o.data.size=area*size;o.rotation_euler=(center-o.location).to_track_quat('-Z','Y').to_euler()
 direction=(-2,-3,1.5) if name.startswith('clamp') else(-2,-3,1.3)
 bpy.ops.object.camera_add(location=center+Vector(direction)*size);cam=bpy.context.object;cam.rotation_euler=(center-cam.location).to_track_quat('-Z','Y').to_euler();cam.data.type='ORTHO';basis=cam.rotation_euler.to_matrix();right=basis@Vector((1,0,0));up=basis@Vector((0,1,0));w=max(v.dot(right)for v in pts)-min(v.dot(right)for v in pts);h=max(v.dot(up)for v in pts)-min(v.dot(up)for v in pts);cam.data.ortho_scale=max(w,h*1600/1300)*1.16;sc.camera=cam
 sc.render.resolution_x=1600;sc.render.resolution_y=1300;sc.render.resolution_percentage=100;sc.view_settings.view_transform='AgX';sc.view_settings.exposure=-.5;sc.render.image_settings.file_format='PNG';sc.render.filepath=str(R/(name+'.png'))
 bpy.ops.wm.save_as_mainfile(filepath=str(R/(name+'.blend')));bpy.ops.render.render(write_still=True)
for name in os.environ.get('RENDER_NAMES','clamp-assembled,clamp-exploded').split(','):render(name)
