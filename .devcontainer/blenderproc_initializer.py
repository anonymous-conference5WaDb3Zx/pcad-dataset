import blenderproc as bproc
bproc.init()

import bpy

obj = bproc.object.create_primitive("MONKEY")
cam = bpy.context.scene.objects['Camera']
cam_pose = bproc.math.build_transformation_mat(cam.location, cam.rotation_euler)
bproc.camera.add_camera_pose(cam_pose)

data = bproc.renderer.render()
