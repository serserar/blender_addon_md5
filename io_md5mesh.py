import bmesh
import logging
import os
import mathutils
from .shared import * # for brevity use star import, also imports modules

logging.basicConfig(style="{", level=logging.WARNING)

#-------------------------------------------------------------------------------
# Classes
#-------------------------------------------------------------------------------

class Vert:
	def __init__(self, mobj=None):
		if mobj is not None:
			self.from_mobj(mobj)
		else:
			self.index = -1
			self.uv = None
			self.fwi = -1
			self.nof_weights = 0
			self.bmv = None

	def from_mobj(self, mobj):
		self.index = int(mobj.group(1))
		self.uv = unpack_tuple(mobj, 2, 3)
		self.fwi = int(mobj.group(4)) # first weight index
		self.nof_weights  = int(mobj.group(5))
		self.bmv = None

		#fmt = "\tvert {index:d} {uv:s} {fwi:d} {nof_weights:d}\n"
		#print(fmt.format(
		#	index = self.index,
		#	uv = "( "+nums(self.uv.x)+" "+nums(self.uv.y)+" )",
		#	fwi = self.fwi,
		#	nof_weights = self.nof_weights
		#))

		self.uv.y = 1.0 - self.uv.y

	def get_weights(self, weights):
		return weights[self.fwi: self.fwi + self.nof_weights]

	def calc_position(self, weights, matrices):
		return sum((matrices[weight.joint_index][1] * weight.offset * weight.value
					for weight in self.get_weights(weights)), Vector())

	def serialize(self, stream):
		self.uv.y = 1.0 - self.uv.y
		fmt = "\tvert {index:d} {uv:s} {fwi:d} {nof_weights:d}\n"

		stream.write(fmt.format(
			index = self.index,
			uv = "( "+nums(self.uv.x)+" "+nums(self.uv.y)+" )",
			fwi = self.fwi,
			nof_weights = self.nof_weights
		))


class Weight:
	def __init__(self, mobj=None):
		if mobj is not None:
			self.from_mobj(mobj)
		else:
			self.index = -1
			self.joint_index = -1
			self.value = 0.0
			self.offset = None

	def from_mobj(self, mobj):
		self.index = int(mobj.group(1))
		self.joint_index = int(mobj.group(2))
		self.value = float(mobj.group(3))
		self.offset = unpack_tuple(mobj, 4, 6)
		#fmt = "\tweight {index:d} {joint_index:d} {value} {offset:s}\n"
		#print(fmt.format(
		#	index = self.index,
		#	joint_index = self.joint_index,
		#	value = self.value,
		#	offset = "( "+nums(self.offset.x)+" "+nums(self.offset.y)+" "+nums(self.offset.z)+" )"
		#))

	def serialize(self, stream):
		fmt = "\tweight {index:d} {joint_index:d} {value} {offset:s}\n"
		stream.write(fmt.format(
			index = self.index,
			joint_index = self.joint_index,
			value = nums(self.value),
			offset = "( "+numr(self.offset.x)+" "+numr(self.offset.y)+" "+numr(self.offset.z)+" )"
		))


class Mesh:
	def __init__(self, mesh_obj, arm_obj):
		mesh = mesh_obj.data

		self.mesh_obj = mesh_obj
		self.weights = []
		self.shader = (mesh.materials[0].name if mesh.materials
					   else "")

		self.bm = bmesh.new()
		self.bm.from_mesh(mesh)
		# Carl: When we import an MD5, the meshes are children of the armature.
		# We want to export the MD5 with the rotations and positions of the meshes
		# preserved relative to the armature.
		if mesh_obj.parent == arm_obj:
			self.bm.transform(mesh_obj.matrix_local)
		self.process_for_export()
		self.bm.verts.index_update()
		self.tris = [[v.index for v in f.verts]
							  for f in self.bm.faces]
		nof_verts = len(self.bm.verts)

		self.verts = [Vert() for i in range(nof_verts)]

	def process_for_export(self):
		bm = self.bm

		def vec_equals(a, b):
			return (a - b).magnitude < 5e-2

		# split vertices with multiple uv coordinates
		seams = []
		tag_verts = set()
		layer_uv = bm.loops.layers.uv.active

		for edge in bm.edges:
			if not edge.is_manifold: continue

			uvs   = [None] * 2
			loops = [None] * 2

			loops[0] = list(edge.link_loops)
			loops[1] = [loop.link_loop_next for loop in loops[0]]

			for i in range(2):
				uvs[i] = list(map(lambda l: l[layer_uv].uv, loops[i]))

			results = (vec_equals(uvs[0][0], uvs[1][1]),
					   vec_equals(uvs[0][1], uvs[1][0]))

			if not all(results):
				if results[0]: tag_verts.add(loops[0][0].vert)
				if results[1]: tag_verts.add(loops[0][1].vert)
				seams.append(edge)

		tag_verts = list(tag_verts)
		bmesh.ops.split_edges(bm, edges=seams, verts=tag_verts, use_verts=True)

		# triangulate
		bmesh.ops.triangulate(bm, faces=bm.faces[:], quad_method=0, ngon_method=0)

		# flip normals
		bmesh.ops.reverse_faces(bm, faces=bm.faces[:], flip_multires=False)

	def set_weights(self, joints, lut):
		vertex_groups = self.mesh_obj.vertex_groups
		layer_deform = self.bm.verts.layers.deform.active
		layer_uv = self.bm.loops.layers.uv.active
		first_index = 0
		nof_weights = 0

		for v, bmv in zip(self.verts, self.bm.verts):
			v.index = bmv.index
			first_index = first_index + nof_weights
			nof_weights = 0
			weights = []

			for key, value in bmv[layer_deform].items():
				if value < 5e-4:
					logging.warning("Skipping weight with value %.2f of vertex %d" % (value, bmv.index))
					continue

				vertex_group = vertex_groups[key]
				joint_index = lut[vertex_group.name]

				weight = Weight()
				weight.index = first_index + nof_weights
				weight.joint_index = joint_index
				weight.value = value
				weights.append(weight)
				nof_weights += 1

			v.fwi = first_index
			v.nof_weights = nof_weights
			self.weights.extend(weights)

			# r = Σ mi wi ri, ensure Σ wi = 1.0 and choose mi^-1 r
			co = (1 / sum(weight.value for weight in weights)) * bmv.co

			for weight in weights:
				weight.offset = joints[weight.joint_index].mat_inv * co

		for face in self.bm.faces:
			for loop in face.loops:
				vert = self.verts[loop.vert.index]
				vert.uv = loop[layer_uv].uv

	def serialize(self, stream, index, version):
		if (version=="6"):
			stream.write("\nmesh %d {\n" % (index))
		else:
			stream.write("\nmesh {\n")
		if version!="6":
			stream.write("\t// meshes: %s\n"   % self.mesh_obj.name) # Sauerbraten reads this field, as do Blender importers
		if version=="10":
			stream.write("\tname \"%s\"\n"   % self.mesh_obj.name) # Doom 3 reads this field if present, Sauerbraten ignores it. Mesh files don't normally contain it.
		elif version=="SB":
			stream.write("\n")
		if version=="6":
			stream.write("\tshader \"%s\"\n" % add_ext(os.path.join("P:/Doom/base", self.shader).replace("\\","/"),".tga"))
		else:
			stream.write("\tshader \"%s\"\n" % self.shader) # This field MUST be present or Doom 3 won't load the mesh!
		stream.write("\n\tnumverts %d\n"   % len(self.verts)) # MUST be present, must be >= 0 and must be accurate

		for vert in self.verts:
			vert.serialize(stream)

		stream.write("\n\tnumtris %d\n"    % len(self.tris))
		for index, tri in enumerate(self.tris):
			stream.write("\ttri {:d} {:d} {:d} {:d}\n".format(index, *tri))

		stream.write("\n\tnumweights %d\n" % len(self.weights))
		for weight in self.weights:
			weight.serialize(stream)

		stream.write("}\n")

	def finish(self):
		self.bm.free()
		self.verts = None
		self.weights = None
		self.tris = None

class Joint:
	def __init__(self):
		self.name   = ""
		self.index  = -1
		self.parent_index = -1
		self.parent_name = ""

		self.mat     = None
		self.mat_inv = None

		self.loc = None
		self.rot = None

	def serialize(self, stream):
		fmt = "\t\"{name:s}\"\t{pindex:d} {loc:s} {rot:s}\t\t// {pname:s}\n"
		stream.write(fmt.format(
			name   = self.name,
			pindex = self.parent_index,
			loc = "( "+nums(self.loc.x)+" "+nums(self.loc.y)+" "+nums(self.loc.z)+" )",
			rot = "( "+nums(self.rot[1])+" "+nums(self.rot[2])+" "+nums(self.rot[3])+" )",
			pname  = self.parent_name
		))

	def serialize6(self, stream):
		if self.parent_index == -1:
			fmt = "bone {index:d} {{\n\tname \"{name:s}\"\n\tbindpos {loc:s}\n\tbindmat {rot:s}\n}}\n\n"
		else:
			fmt = "bone {index:d} {{\n\tname \"{name:s}\"\n\tbindpos {loc:s}\n\tbindmat {rot:s}\n\tparent \"{pname:s}\"\n}}\n\n"
		stream.write(fmt.format(
			index = self.index,
			name   = self.name,
			loc = num6(self.loc.x)+" "+num6(self.loc.y)+" "+num6(self.loc.z),
			rot = num6(self.mat[0][0])+" "+num6(self.mat[1][0])+" "+num6(self.mat[2][0])+" "+num6(self.mat[0][1])+" "+num6(self.mat[1][1])+" "+num6(self.mat[2][1])+" "+num6(self.mat[0][2])+" "+num6(self.mat[1][2])+" "+num6(self.mat[2][2]),
			pname  = self.parent_name
		))

	def from_bone(self, bone, index, lut):
		self.name = bone.name
		self.index = lut[bone.name] = index
		self.parent_index = self.get_parent_index(bone, lut)
		self.parent_name = bone.parent.name if bone.parent is not None else ""
		self.mat = bone.matrix_local.copy()
		self.mat_inv = self.mat.inverted()
		self.loc, self.rot, scale = self.mat.decompose()
		self.rot *= -1.0

	@classmethod
	def get_parent_index(cls, bone, lut):
		if bone.parent is None: return -1
		return lut[bone.parent.name]

#-------------------------------------------------------------------------------
# Read md5mesh
#-------------------------------------------------------------------------------

def read_md5mesh(filepath):
	t_Int   = r"(-?\d+)"
	t_Float = r"([+-]?\d+\.?\d*[eE]?[+-]?\d*)"
	t_Word  = r"(\S+)"
	t_QuotedString = '"([^"]*)"' # Doom 3 does not allow escaping \"
	t_token = r"(?:\"([^\"]*)\"|'([^']*)'|([a-zA-Z_/\\][a-zA-Z_0-9:/\\.]*|0[xX][0-9a-fA-F]+|0[bB][01]+|[0-9]*[.][0-9]*(?:e[-+]?[0-9]*|#(?:INF|IND|NAN|QNAN|SNAN|NaN)|[0-9.]*)|0[0-7]+|[0-9]+|[~`!@#$%^&*()-=+|{}\[\];:,<>?]))"
	#t_Tuple2f = "\\s+".join(("\\(?", t_Float, t_Float, "\\)?"))
	#t_Tuple3f = "\\s+".join(("\\(?", t_Float, t_Float, t_Float, "\\)?"))
	t_Tuple2f = r"[(]?\s*" + (t_Float + r"\s+") + t_Float + r"\s*[)]?"
	t_Tuple3f = r"[(]?\s*" + (t_Float + r"\s+") * 2 + t_Float + r"\s*[)]?"
	t_Tuple9f = r"[(]?\s*" + (t_Float + r"\s+") * 8 + t_Float + r"\s*[)]?"

	re_commandline = construct("commandline", t_QuotedString)

	re_joint  = construct(t_QuotedString, t_Int, t_Tuple3f, t_Tuple3f)
	re_vert   = construct("vert", t_Int, t_Tuple2f, t_Int, t_Int)
	re_tri    = construct("tri", t_Int, t_Int, t_Int, t_Int)
	re_weight = construct("weight", t_Int, t_Int, t_Float, t_Tuple3f)
	re_end    = construct("\\}")
	re_joints = construct("joints", "\\{")
	re_nverts = construct("numverts", t_Int)
	re_mesh   = construct(r"mesh(?:\s+" + t_Int + r")?\s*\{")
	re_shader = construct("shader", t_token)
	re_mesh_label = construct(".*?// meshes: (.*)$") # comment, used by sauerbraten

	re_nbones = construct("numbones", t_Int)
	re_bone   = construct("bone", t_Int, r"\{")
	re_name   = construct("name", t_token)
	re_parent_name = construct("parent", t_token)
	re_bindpos = construct("bindpos", t_Tuple3f)
	re_bindmat = construct("bindmat", t_Tuple9f)
	re_nmeshes = construct("nummeshes", t_Int) # only matches the alpha, normal Doom 3 uses numMeshes

	with open(filepath, "r") as fobj:
		lines = iter(fobj.readlines())

	filename, file_extension = os.path.splitext(os.path.basename(filepath))

	commandline = ""
	numbones = -1
	nummeshes = 0
	for line in lines:
		# stop when if get to the start of the joints
		mobj = re_joints.match(line)
		if mobj:
			break
		# In Doom 3 alpha, nummeshes marks the end of the bones (which it has instead of joints)
		mobj = re_nmeshes.match(line)
		if mobj:
			nummeshes = int(mobj.group(1))
			if numbones > -1:
				break
			else:
				continue
		mobj = re_commandline.match(line)
		if mobj:
			commandline = mobj.group(1)
			continue
		mobj = re_nbones.match(line)
		if mobj:
			numbones = int(mobj.group(1))
			reg_exprs = re_bone, re_name, re_bindpos, re_bindmat, re_parent_name, re_end, re_nmeshes
			arm_obj, matrices = do_bones(lines, reg_exprs, filename)
			break

	if numbones == -1:
		arm_obj, matrices = do_joints(lines, re_joint, re_end, filename)
	arm_obj['commandline'] = commandline # save commandline in a c

	results = []
	reg_exprs = re_shader, re_vert, re_tri, re_weight, re_end, re_nverts, re_mesh_label, re_name
	n = 0

	while True:
		results.append(do_mesh(lines, reg_exprs, matrices))
		n += 1

		if skip_until(re_mesh, lines) is None:
			break

	for label, shader, bm in results:
		mesh = bpy.data.meshes.new(label)
		bm.to_mesh(mesh)
		bm.free()

		mesh.auto_smooth_angle = math.radians(45)
		mesh.use_auto_smooth = True

		mesh_obj = bpy.data.objects.new(label, mesh)
		for joint_name, mat in matrices:
			mesh_obj.vertex_groups.new(name=joint_name)

		mesh_obj.parent = arm_obj
		arm_mod = mesh_obj.modifiers.new(type='ARMATURE', name="MD5_skeleton")
		arm_mod.object = arm_obj
		arm_mod.use_deform_preserve_volume = True

		bpy.context.scene.objects.link(mesh_obj)

		# apply texture in Blender Render
		mat_name = get_doom3_shader_name(shader)
		mat = (bpy.data.materials.get(mat_name) or
			   bpy.data.materials.new(mat_name))
		tex = bpy.data.textures.new(os.path.basename(shader), 'IMAGE')
		imagefilename = get_image_filename(filepath, shader)
		if imagefilename:
			tex.image = bpy.data.images.load(filepath = imagefilename)
		slot = mat.texture_slots.add()
		slot.texture = tex
		# apply texture in 3D View's Texture View
		slot.uv_layer = "UVMap"
		for uv_face in mesh.uv_textures.active.data:
			uv_face.image = tex.image
		# apply texture in Cycles (Now Blender Render won't work until you turn off Use Node Tree)
		mat.use_nodes = True
		nt = mat.node_tree
		nodes = nt.nodes
		links = nt.links
		while(nodes): nodes.remove(nodes[0])
		output  = nodes.new("ShaderNodeOutputMaterial")
		diffuse = nodes.new("ShaderNodeBsdfDiffuse")
		texture = nodes.new("ShaderNodeTexImage")
		texture.image = tex.image
		links.new( output.inputs['Surface'], diffuse.outputs['BSDF'])
		links.new(diffuse.inputs['Color'],   texture.outputs['Color'])
		# distribute nodes along the x axis
		for index, node in enumerate((texture, diffuse, output)):
			node.location.x = 200.0 * index
		# apply material
		mesh.materials.append(mat)
		

	return arm_obj

def do_joints(lines, re_joint, re_end, filename):
	joints = gather(re_joint, re_end, lines)

	arm = bpy.data.armatures.new("MD5")
	arm_obj = bpy.data.objects.new(filename, arm)
	arm_obj.select = True
	bpy.context.scene.objects.link(arm_obj)
	bpy.context.scene.objects.active = arm_obj

	matrices = []
	name_to_index = {}
	VEC_Y = Vector((0.0, 1.0, 0.0))
	VEC_Z = Vector((0.0, 0.0, 1.0))

	bpy.ops.object.mode_set(mode='EDIT')
	edit_bones = arm.edit_bones

	for index, mobj in enumerate(joints):
		name = mobj.group(1)
		parent = int(mobj.group(2))
		loc  = unpack_tuple(mobj, 3, 5)
		quat = unpack_tuple(mobj, 6, 8, seq=tuple)
		name_to_index[name] = index

		eb = edit_bones.new(name)
		if parent >= 0:
			eb.parent = edit_bones[parent]

		quat = restore_quat(*quat)
		mat = Matrix.Translation(loc) * quat.to_matrix().to_4x4()
		matrices.append((name, mat))

		eb.head = loc
		eb.tail = loc + quat * VEC_Y
		eb.align_roll(quat * VEC_Z)

	for eb in arm.edit_bones:
		if len(eb.children) == 1:
			child = eb.children[0]
			head_to_head = child.head - eb.head
			projection = head_to_head.project(eb.y_axis)
			if eb.y_axis.dot(projection) > 5e-2:
				eb.tail = eb.head + projection

	bpy.ops.object.mode_set()
	arm_obj['name_to_index'] = name_to_index
	return arm_obj, matrices

def do_binary_joints(file, filename):
	arm = bpy.data.armatures.new("MD5")
	arm_obj = bpy.data.objects.new(filename, arm)
	arm_obj.select = True
	bpy.context.scene.objects.link(arm_obj)
	bpy.context.scene.objects.active = arm_obj

	matrices = []
	name_to_index = {}
	VEC_Y = Vector((0.0, 1.0, 0.0))
	VEC_Z = Vector((0.0, 0.0, 1.0))

	bpy.ops.object.mode_set(mode='EDIT')
	edit_bones = arm.edit_bones
	
	joints = []
	quats = []

	numJoints = read_big_int(file)
	print ("numJoints %d\nnumMeshes\n\njoints {" % numJoints)
	for index in range(numJoints):
		name = read_string(file)
		parent = read_big_int(file)
		joints.append((name, parent))
		name_to_index[name] = index

	numDefaultPoses = read_big_int(file)
	#print ("%d poses", numDefaultPoses)
	for index in range(numDefaultPoses):
		if index < numJoints:
			name, parent = joints[index]
		else:
			name = "bone%d" % (index)
			parent = -1
			name_to_index[name] = index

		quat = read_quat(file)
		#if parent >= 0:
		#	quat = quats[parent] * quat
		#quat.normalize()
		#quats.append(quat)
		loc = read_vec3(file)

		eb = edit_bones.new(name)
		if parent >= 0:
			eb.parent = edit_bones[parent]

		mat = Matrix.Translation(loc) * quat.to_matrix().to_4x4()

		if parent < 0:
			parent_name = ""
			parent_mat = Matrix()
		else:
			parent_name = joints[parent][0]
			parent_mat = matrices[parent][1]

		mat = parent_mat * mat
		matrices.append((name, mat))

		print("\t\"%s\"\t%d ( %.5f %.5f %.5f ) ( %.5f %.5f %.5f )\t\t// %s" % (name, parent, loc.x, loc.y, loc.z, quat[0], quat[1], quat[2], parent_name))
		
		eb.head = loc
		eb.tail = loc + quat * VEC_Y
		eb.align_roll(quat * VEC_Z)

	numInvertedDefaultPoses = read_big_int(file)
	print ("%d inverted poses", numInvertedDefaultPoses)
	for index in range(numInvertedDefaultPoses):
		file.read( 4 * 3 * 4 )
		
	for eb in arm.edit_bones:
		if len(eb.children) == 1:
			child = eb.children[0]
			head_to_head = child.head - eb.head
			projection = head_to_head.project(eb.y_axis)
			if eb.y_axis.dot(projection) > 5e-2:
				eb.tail = eb.head + projection

	bpy.ops.object.mode_set()
	arm_obj['name_to_index'] = name_to_index
	return arm_obj, matrices

def do_bones(lines, reg_exprs, filename):
	(re_bone,
	 re_name,
	 re_bindpos,
	 re_bindmat,
	 re_parent_name,
	 re_end_bone,
	 re_end) = reg_exprs

	arm = bpy.data.armatures.new("MD5 alpha")
	arm_obj = bpy.data.objects.new(filename, arm)
	arm_obj.select = True
	bpy.context.scene.objects.link(arm_obj)
	bpy.context.scene.objects.active = arm_obj

	matrices = []
	name_to_index = {}
	VEC_Y = Vector((0.0, 1.0, 0.0))
	VEC_Z = Vector((0.0, 0.0, 1.0))

	bpy.ops.object.mode_set(mode='EDIT')
	edit_bones = arm.edit_bones

	for line in lines:
		mobj = re_bone.match(line)
		if mobj:
			index = int(mobj.group(1))
			parent = -1
			for s in lines:
				mobj = re_name.match(s)
				if mobj:
					name = token_value(mobj, "bone%d" % (index))
					name_to_index[name] = index
					continue
				mobj = re_parent_name.match(s)
				if mobj:
					parent_name = token_value(mobj, "")
					parent = name_to_index[parent_name]
					continue
				mobj = re_bindpos.match(s)
				if mobj:
					loc  = unpack_tuple(mobj, 1, 3)
					continue
				mobj = re_bindmat.match(s)
				if mobj:
					# we need to convert this 3x3 matrix into something useable
					rot = mathutils.Matrix()
					rot[0][0], rot[1][0], rot[2][0] = float(mobj.group(1)), float(mobj.group(2)), float(mobj.group(3))
					rot[0][1], rot[1][1], rot[2][1] = float(mobj.group(4)), float(mobj.group(5)), float(mobj.group(6))
					rot[0][2], rot[1][2], rot[2][2] = float(mobj.group(7)), float(mobj.group(8)), float(mobj.group(9))
					#rot[0][0], rot[0][1], rot[0][2] = float(mobj.group(1)), float(mobj.group(2)), float(mobj.group(3))
					#rot[1][0], rot[1][1], rot[1][2] = float(mobj.group(4)), float(mobj.group(5)), float(mobj.group(6))
					#rot[2][0], rot[2][1], rot[2][2] = float(mobj.group(7)), float(mobj.group(8)), float(mobj.group(9))
					continue
				mobj = re_end_bone.match(s)
				if mobj:
					break
			eb = edit_bones.new(name)
			if parent >= 0:
				eb.parent = edit_bones[parent]

			mat = Matrix.Translation(loc) * rot.to_4x4()
			matrices.append((name, mat))

			eb.head = loc
			eb.tail = loc + rot * VEC_Y
			eb.align_roll(rot * VEC_Z)

			continue
		mobj = re_end.match(line)
		if mobj:
			break

	for eb in arm.edit_bones:
		if len(eb.children) == 1:
			child = eb.children[0]
			head_to_head = child.head - eb.head
			projection = head_to_head.project(eb.y_axis)
			if eb.y_axis.dot(projection) > 5e-2:
				eb.tail = eb.head + projection

	bpy.ops.object.mode_set()
	arm_obj['name_to_index'] = name_to_index
	return arm_obj, matrices

def do_mesh(lines, reg_exprs, matrices):
	(re_shader,
	 re_vert,
	 re_tri,
	 re_weight,
	 re_end,
	 re_nverts,
	 re_label,
	 re_name) = reg_exprs

	mobjs__label, mobjs___name, mobjs_shader = gather_multi([re_label, re_name, re_shader], re_nverts, lines)
	label  = mobjs__label[0].group(1) if len(mobjs__label) > 0 else ""
	if len(mobjs___name) > 0:
		label = token_value(mobjs___name[0], label)
	shader = ""
	if len(mobjs_shader) > 0:
		shader = token_value(mobjs_shader[0], shader)

	if shader == "" and label == "":
		label = "md5mesh"
		shader = ""
	elif shader == "":
		shader = label
	elif label == "":
		label, file_extension = os.path.splitext(os.path.basename(shader))

	verts, tris, weights = gather_multi(
		[re_vert, re_tri, re_weight],
		re_end,
		lines
	)

	print("do_mesh", label, shader)
	
	bm = bmesh.new()
	process_match_objects(verts,   Vert)
	process_match_objects(weights, Weight)

	layer_weight = bm.verts.layers.deform.verify()
	layer_uv	 = bm.loops.layers.uv.verify()

	for index, vert in enumerate(verts):
		vert.bmv = bm.verts.new(vert.calc_position(weights, matrices))
		for weight in vert.get_weights(weights):
			vert.bmv[layer_weight][weight.joint_index] = weight.value

	for mobj_tri in tris:
		vertex_indices = unpack_tuple(mobj_tri, 2, 4, int, list)
		bm_verts = [verts[vertex_index].bmv for vertex_index in vertex_indices]
		# bm_verts.reverse() - use bmesh operator instead
		try:
			face = bm.faces.new(bm_verts)
		except ValueError: # some models contain duplicate faces
			continue
		face.smooth = True

	for vert in verts:
		for loop in vert.bmv.link_loops:
			loop[layer_uv].uv = vert.uv
			vert.bmv = None

	# flip normals
	bmesh.ops.reverse_faces(bm, faces=bm.faces[:], flip_multires=False)

	return label, shader, bm

def do_binary_mesh(file, matrices):

	shader = read_string(file)
	#print("mesh shader = ", shader)
	label, file_extension = os.path.splitext(os.path.basename(shader))
	numVerts = read_big_int(file)
	numTris = read_big_int(file)
	numMeshJoints = read_big_int(file)
	meshJoints = file.read(numMeshJoints)
	maxJointVertDist = read_big_float(file)
	numSourceVerts = read_big_int(file)
	numOutputVerts = read_big_int(file)
	numIndexes = read_big_int(file)
	numMirroredVerts = read_big_int(file)
	numDupVerts = read_big_int(file)
	numSilEdges = read_big_int(file)

	bm = bmesh.new()
	layer_weight = bm.verts.layers.deform.verify()
	layer_uv	 = bm.loops.layers.uv.verify()

	verts = []
	for vert_index in range(numOutputVerts):
		vert = Vert()
		vert.index = vert_index
		x = read_big_float(file)
		y = read_big_float(file)
		z = read_big_float(file)
		vert.bmv = bm.verts.new(Vector((x,y,z)))
		u = F16toF32(read_big_short(file))
		v = F16toF32(read_big_short(file))
		vert.uv = Vector((u, v))
		normal = file.read(4)
		tangent = file.read(4) # [3] is texture polarity sign
		finalJointIndecies = file.read(4) # joint indexes for skinning
		finalWeights = file.read(4) # weights for skinning
		for i in range(4):
			if finalJointIndecies[i] >= 0:
				vert.bmv[layer_weight][finalJointIndecies[i]] = finalWeights[i] / 255.0
		verts.append(vert)

	for tri_index in range(numIndexes // 3):
		v1 = read_big_short(file)
		v2 = read_big_short(file)
		v3 = read_big_short(file)
		vertex_indices = [v1, v2, v3]
		bm_verts = [verts[vertex_index].bmv for vertex_index in vertex_indices]
		# bm_verts.reverse() - use bmesh operator instead
		try:
			face = bm.faces.new(bm_verts)
		except ValueError: # some models contain duplicate faces
			continue
		face.smooth = True		

	file.read(numIndexes * 2) # silIndexes
	file.read(numMirroredVerts * 4)
	file.read(numDupVerts * 2 * 4)
	file.read(numSilEdges * 8)
	surfaceNum = read_big_int(file)

	for vert in verts:
		for loop in vert.bmv.link_loops:
			loop[layer_uv].uv = vert.uv
			vert.bmv = None

	# flip normals
	bmesh.ops.reverse_faces(bm, faces=bm.faces[:], flip_multires=False)

	return label, shader, bm

#-------------------------------------------------------------------------------
# Write md5mesh
#-------------------------------------------------------------------------------

def on_active_layer(scene, obj):
	layers_scene = scene.layers
	layers_obj   = obj.layers

	for i in range(20):
		if layers_scene[i] and layers_obj[i]:
			return True
	return False

def write_md5mesh(filepath, scene, arm_obj, version):
	meshes = []

	for mesh_obj in filter(is_mesh_object, scene.objects):
		if (on_active_layer(scene, mesh_obj) and
			has_armature_modifier(mesh_obj, arm_obj)):
			meshes.append(Mesh(mesh_obj, arm_obj))

	bones = arm_obj.data.bones
	joints = [Joint() for i in range(len(bones))]
	name_to_index = get_name_to_index_dict(arm_obj)

	for bone in bones:
		index = name_to_index[bone.name]
		joints[index].from_bone(bone, index, name_to_index)

	for mesh in meshes:
		mesh.set_weights(joints, name_to_index)

	with open(filepath, "w") as stream:
		if (version == "SB"):
			stream.write("MD5Version 10\n")
		else:
			stream.write("MD5Version %s\n" % (version))

		commandline = arm_obj.get("commandline")
		if commandline is None or version=="SB":
			commandline = ""
		stream.write("commandline \"%s\"\n" % (commandline)) # MUST be present for Doom 3
		if (version=="6"):
			stream.write("numbones %d\n\n" % len(joints))
			for joint in joints:
				joint.serialize6(stream)
			stream.write("nummeshes %d\n" % len(meshes))
		else:
			stream.write("\nnumJoints %d\n" % len(joints))
			stream.write("numMeshes %d\n" % len(meshes))

			stream.write("\njoints {\n")
			for joint in joints:
				joint.serialize(stream)
			stream.write("}\n")

		index = 0
		for mesh in meshes:
			mesh.serialize(stream, index, version)
			mesh.finish()
			index += 1

#-------------------------------------------------------------------------------
# Read bmd5mesh
#-------------------------------------------------------------------------------

def read_bmd5mesh(filepath):
	file = open(filepath, 'rb')

	# BRM header
	magic = read_big_int(file)
	if magic != 0x42524D6C:
		print('\tFatal Error: %d Not a valid binary render model file: %r' % (magic, filepath))
		file.close()
		return
	
	timeStamp = read_big_int64(file)
	numSurfaces = read_big_int(file)
	for index in range(numSurfaces):
		id = read_big_int(file)
		materialName = read_string(file)
		isGeometry = read_bool(file)
		if isGeometry:
			print("unsupported!")
			file.close()
			return
	bounds0 = read_vec3(file)
	bounds1 = read_vec3(file)
	overlaysAdded = read_big_int(file)
	lastModifiedFrame = read_big_int(file)
	lastArchivedFrame = read_big_int(file)
	modelName = read_string(file)
	isStaticWorldModel = read_bool(file)
	defaulted = read_bool(file)
	purged = read_bool(file)
	fastLoad = read_bool(file)
	reloadable = read_bool(file)
	levelLoadReferenced = read_bool(file)
	hasDrawingSurfaces = read_bool(file)
	hasInteractingSurfaces = read_bool(file)
	hasShadowCastingSurfaces = read_bool(file)
	
	# Actual MD5 file
	magic = read_big_int(file)
	if magic != 0x35444D6A:
		print('\tFatal Error: %d Not a valid bmd5mesh file: %r' % (magic, filepath))
		file.close()
		return
	
	filename, file_extension = os.path.splitext(os.path.basename(filepath))

	commandline = ""
	arm_obj, matrices = do_binary_joints(file, filename)
	arm_obj['commandline'] = commandline # save commandline in a c

	results = []
	n = 0

	numMeshes = read_big_int(file)
	for n in range(numMeshes):
		results.append(do_binary_mesh(file, matrices))
	
	file.close()

	for label, shader, bm in results:
		mesh = bpy.data.meshes.new(label)
		bm.to_mesh(mesh)
		bm.free()

		mesh.auto_smooth_angle = math.radians(45)
		mesh.use_auto_smooth = True

		mesh_obj = bpy.data.objects.new(label, mesh)
		for joint_name, mat in matrices:
			mesh_obj.vertex_groups.new(name=joint_name)

		mesh_obj.parent = arm_obj
		arm_mod = mesh_obj.modifiers.new(type='ARMATURE', name="MD5_skeleton")
		arm_mod.object = arm_obj
		arm_mod.use_deform_preserve_volume = True

		bpy.context.scene.objects.link(mesh_obj)

		# apply texture in Blender Render
		mat_name = get_doom3_shader_name(shader)
		mat = (bpy.data.materials.get(mat_name) or
			   bpy.data.materials.new(mat_name))
		tex = bpy.data.textures.new(os.path.basename(shader), 'IMAGE')
		imagefilename = get_image_filename(filepath, shader)
		if imagefilename:
			tex.image = bpy.data.images.load(filepath = imagefilename)
		slot = mat.texture_slots.add()
		slot.texture = tex
		# apply texture in 3D View's Texture View
		slot.uv_layer = "UVMap"
		for uv_face in mesh.uv_textures.active.data:
			uv_face.image = tex.image
		# apply texture in Cycles (Now Blender Render won't work until you turn off Use Node Tree)
		mat.use_nodes = True
		nt = mat.node_tree
		nodes = nt.nodes
		links = nt.links
		while(nodes): nodes.remove(nodes[0])
		output  = nodes.new("ShaderNodeOutputMaterial")
		diffuse = nodes.new("ShaderNodeBsdfDiffuse")
		texture = nodes.new("ShaderNodeTexImage")
		texture.image = tex.image
		links.new( output.inputs['Surface'], diffuse.outputs['BSDF'])
		links.new(diffuse.inputs['Color'],   texture.outputs['Color'])
		# distribute nodes along the x axis
		for index, node in enumerate((texture, diffuse, output)):
			node.location.x = 200.0 * index
		# apply material
		mesh.materials.append(mat)

	return arm_obj

#-------------------------------------------------------------------------------
# Test
#-------------------------------------------------------------------------------

def test():
	import os

	filepath = os.path.expanduser(
		"~/Downloads/Games/sauerbraten_2013"
		"/sauerbraten/packages/models"
		"/snoutx10k/snoutx10k.md5mesh")

	output = os.path.expanduser("~/Dokumente/Blender/Scripts/addons/md5/test.md5mesh")

	layer_source   = tuple(i == 0 for i in range(20))
	layer_reimport = tuple(i == 1 for i in range(20))

	scene = bpy.context.scene
	scene.layers = layer_source

	while bpy.data.objects:
		obj = bpy.data.objects[0]
		scene.objects.unlink(obj)
		obj.user_clear()
		bpy.data.objects.remove(obj)

	read_md5mesh(filepath)
	write_md5mesh(output, scene, bpy.context.active_object)

	scene.layers = layer_reimport
	read_md5mesh(output)
