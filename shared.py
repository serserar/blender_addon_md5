import re
import math
import bpy
import os
import struct
from mathutils import Vector, Matrix, Quaternion

fmt_row2f = "( {} {} )"
fmt_row3f = "( {} {} {} )"

def num6(x):
	return ("%.6f" % (x))

def nums(x):
	if abs(x - int(x)) < 0.000005:
		return "%d" % (int(x)) # if it's an integer, return it as an integer
	else:
		return ("%.5f" % (x)).rstrip("0").rstrip(".") # remove trailing zeroes, should be %.10f

def numr(x):
	if abs(x - int(x)) < 0.000005:
		return "%d" % (int(x)) # rounding errors have lead to integers no longer being integers
	else:
		return ("%.5f" % (x)).rstrip("0").rstrip(".") # remove trailing zeroes, should be %.10f

def construct(*args):
	return re.compile("\s*" + "\s+".join(args))

def unpack_tuple(mobj, start, end, conv=float, seq=Vector):
	return seq(conv(mobj.group(i)) for i in range(start, end + 1))
	
def token_value(mobj, default):
	if mobj.group(1) is not None:
		return mobj.group(1)
	if mobj.group(2) is not None:
		return mobj.group(2)
	if mobj.group(3) is not None:
		return mobj.group(3)
	return default

def gather(regex, end_regex, lines):
	return gather_multi([regex], end_regex, lines)[0]

def gather_multi(regexes, end_regex, lines):
	results = [[] for regex in regexes]

	for line in lines:
		if end_regex.match(line):
			break

		for regex, result in zip(regexes, results):
			mobj = regex.match(line)
			if mobj:
				result.append(mobj)
				break

	return results

def skip_until(regex, lines):
	for line in lines:
		if regex.match(line):
			return line
	# iterator exhausted
	return None

def restore_quat(rx, ry, rz):
	EPS = -5e-2
	t = 1.0 - rx*rx - ry*ry - rz*rz
	if EPS > t: 	   raise ValueError
	if EPS < t  < 0.0: return  Quaternion((          0.0, rx, ry, rz))
	else:			   return -Quaternion((-math.sqrt(t), rx, ry, rz))

def is_mesh_object(obj):
	return obj.type == "MESH"

def has_armature_modifier(obj, arm_obj):
	for modifier in obj.modifiers:
		if modifier.type == "ARMATURE":
			if modifier.object == arm_obj:
				return True
	return False

def process_match_objects(mobj_list, cls):
	for index, mobj in enumerate(mobj_list):
		mobj_list[index] = cls(mobj)

def get_name_to_index_dict(arm_obj):
	name_to_index = arm_obj.get("name_to_index")

	if name_to_index is not None:
		name_to_index = name_to_index.to_dict()

	else:
		name_to_index = {}
		root_bones = (pb for pb in arm_obj.pose.bones if pb.parent is None)
		index = 0

		for root_bone in root_bones:
			name_to_index[root_bone.name] = index
			index += 1
			for child in root_bone.children_recursive:
				name_to_index[child.name] = index
				index += 1

		arm_obj["name_to_index"] = name_to_index

	return name_to_index

def get_doom3_base_folder(localfilename):
	norm = os.path.normpath(os.path.abspath(localfilename))
	path = norm.lower().replace("\\", "/")
	i = path.rfind("/models/md5/")
	if i < 0:
		i = path.rfind("/models/")
	if i >= 0:
		dir = norm[:i]
	else:
		dir = os.path.dirname(norm)
	return os.path.join(os.path.normpath(dir.rstrip("/\\")),"")

def get_doom3_base_relative_path(filepath):
	norm = os.path.normpath(filepath)
	path = norm.lower().replace("\\", "/")
	i = path.rfind("/models/md5/")
	if i < 0:
		i = path.rfind("/models/")
	if path.startswith("p:/doom/base/"):
		dir = norm[13:]
	elif i >= 0:
		dir = norm[i:]
	elif os.path.isabs(filepath):
		dir = os.path.basename(norm)
	else:
		dir = norm
	return dir.lstrip("/\\")

def convert_doom3_path(localfilename, givenfilename):
	return os.path.join(get_doom3_base_folder(localfilename), get_doom3_base_relative_path(givenfilename))

def get_doom3_shader_name(givenfilename):
	result, ext = os.path.splitext(get_doom3_base_relative_path(givenfilename))
	return result.replace("\\", "/")

def change_ext(filepath, newext):
	root, ext = os.path.splitext(filepath)
	return root + newext
	
def add_ext(filepath, newext):
	root, ext = os.path.splitext(filepath)
	if (ext == ""):
		return root + newext
	else:
		return filepath

def get_image_filename(localfilename, givenfilename):
	#   base + mat_name + .tga
	imagefilename = add_ext(convert_doom3_path(localfilename, givenfilename), ".tga")
	print("trying "+imagefilename)
	if os.path.isfile(imagefilename) and os.access(imagefilename, os.R_OK):
		return imagefilename
	root, ext = os.path.splitext(imagefilename)
	# try adding _d (for diffuse?)
	fn = root+"_d"+ext
	if fn != imagefilename and os.path.isfile(fn) and os.access(fn, os.R_OK):
		return fn
	# try changing head to _d (for diffuse?)
	if root.endswith("head"):
		fn = root[:-4]+"_d"+ext
		if fn != imagefilename and os.path.isfile(fn) and os.access(fn, os.R_OK):
			return fn
	# if it begins with teeth, try that by itself
	if os.path.basename(root).startswith("teeth"):
		fn = os.path.join(os.path.dirname(root), "teeth.tga")
		if fn != imagefilename and os.path.isfile(fn) and os.access(fn, os.R_OK):
			return fn
	# if it's probably an eye
	if root.find("common/left") >= 0 or root.find("common/right") >= 0 or root.find(r"common\left") >= 0 or root.find(r"common\right") >= 0:
		fn = os.path.join(os.path.dirname(root), "blue.tga")
		if fn != imagefilename and os.path.isfile(fn) and os.access(fn, os.R_OK):
			return fn
	# if it ends with a 2 and isn't found, try it without the 2
	fn = root.rstrip("2")+ext
	if fn != imagefilename and os.path.isfile(fn) and os.access(fn, os.R_OK):
		return fn
	#   path + filename.tga
	fn = change_ext(localfilename, ".tga")
	if fn != imagefilename and os.path.isfile(fn) and os.access(fn, os.R_OK):
		return fn
	#   path + filename(mat_name) + .tga
	fn = add_ext(os.path.join(os.path.dirname(localfilename), os.path.basename(givenfilename)), ".tga")
	if fn != imagefilename and os.path.isfile(fn) and os.access(fn, os.R_OK):
		return fn
	#   base + filepath(mat_name) + filename.tga
	fn = add_ext(os.path.join(os.path.dirname(imagefilename), os.path.basename(localfilename)), ".tga")
	if fn != imagefilename and os.path.isfile(fn) and os.access(fn, os.R_OK):
		return fn
	return None

def read_little_int(file):
	result = struct.unpack("<i", file.read(4))[0]
	#print("read_little_int", result)
	return result

def read_little_float(file):
	result = struct.unpack("<f", file.read(4))[0]
	#print("read_little_float", result)
	return result

def read_big_int(file):
	result = struct.unpack(">i", file.read(4))[0]
	#print("read_big_int", result)
	return result

def read_big_int64(file):
	result = struct.unpack(">q", file.read(8))[0]
	#print("read_big_int64", result)
	return result

def read_big_float(file):
	result = struct.unpack(">f", file.read(4))[0]
	#print("read_big_int", result)
	return result

def read_big_short(file):
	result = struct.unpack(">H", file.read(2))[0]
	#print("read_big_int", result)
	return result

def read_byte(file):
	return int(file.read(1)[0])

def read_bool(file):
	return bool(file.read(1)[0])

def read_vec3(file):
	# Vectors are little-endian, even though individual floats are big-endian!
	result = Vector(struct.unpack("<fff", file.read(3*4)))
	return result

def read_quat(file):
	result = Quaternion(struct.unpack(">ffff", file.read(4*4)))
	#print("read_big_int", result)
	return result

def read_string(file):
	length = read_little_int(file)
	result = "".join(map(chr, file.read(length)))
	#print("read_string", result)
	return result

def F16toF32(x):
	e = (x & 32767) >> 10
	m = x & 1023
	if x & 32768:
		s = 1
	else:
		s = 0
	
	if 0 < e and e < 31:
		return s * pow( 2.0, ( e - 15.0 ) ) * ( 1 + m / 1024.0 )
	elif m == 0:
		return s * 0.0
	return s * pow( 2.0, -14.0 ) * ( m / 1024.0 );

