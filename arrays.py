import numpy
import scipy.ndimage as ndimage

##### ##### ##### ##### ##### 

EPSILON = numpy.finfo(numpy.float16).eps # Use a low rez float to be safe

BYTES = numpy.uint8
FLOATS = numpy.float16 # Note: Change to tune accuracy and performace

FLOATS_INFO	= numpy.finfo(FLOATS)
BYTES_INFO	= numpy.iinfo(BYTES)

##### ##### ##### ##### ##### 

def is_nearly(x, y):
	# Given x and y as numbers, or arrays.
	# Check if all values of x are nearly equal to the corresponding y value
	return numpy.all(numpy.abs(numpy.float128(y) - numpy.float128(x)) <= EPSILON)

def is_nearly_zero(x):
	return is_nearly(x, 0)

##### ##### ##### ##### ##### 

def is_similar_type(x, t):

	# x is either a type or an instance of an object
	# t must be a type
	# 
	# If x is a type,
	# 	Return True if x type and t type are the same
	# 	Otherwise False
	# Otherwise
	#	Return True if x is an instance of t
	#	Otherwise False
	#
	# Examples:
	# 	is_similar_type(numpy.uint8, numpy.uint8)		=> True
	#	is_similar_type(numpy.uint8, numpy.float32)		=> False
	#	is_similar_type(numpy.uint8(), numpy.uint8)		=> True
	#	is_similar_type(numpy.uint16(), numpy.uint8)	=> False

	if isinstance(x, type):
		return x == t
	else:
		return isinstance(x, t)

##### ##### ##### ##### ##### 

def is_bytes(x):
	return is_similar_type(x, BYTES)

def as_bytes(x):
	
	if is_bytes(x):
		return x

	# Reduce data loss...
	x = normalize(x) # Scale to [0, 1)
	x = x*BYTES_INFO.max # Scale to [0, max)

	x = BYTES(x) # Shrink unit capacity to byte
	
	return x

##### ##### ##### ##### ##### 

def is_floats(x):
	return is_similar_type(x, FLOATS)

def as_floats(x):

	if is_floats(x):
		return x

	x = FLOATS(x) # Capacity is assumed to be large enough...

	return x

##### ##### ##### ##### ##### 

def is_equal(a, b):
	
	a = numpy.array(a)
	b = numpy.array(b)

	if numpy.any(a != b): # "If any in a does not equal in b"
		return False
	else:
		return True

##### ##### ##### ##### ##### 

def normalize(array):

	array = as_floats(array)

	array_min = numpy.min(array)
	array_max = numpy.max(array)
	array_range = array_max - array_min

	array -= array_min

	# Prevent division by zero
	if is_nearly_zero(array_range) == False:
		array /= array_range

	return array

##### ##### ##### ##### ##### 

def resample(array, new_shape):

	'''
	order = 0 # nearest
	order = 1 # bilinear
	order = 2 # cubic
	...
	order = 6
	'''

	array = numpy.array(array)

	old_shape = array.shape
	old_shape = as_floats(old_shape)
	new_shape = as_floats(new_shape)

	array = ndimage.interpolation.zoom(input = array, zoom = (new_shape / old_shape), order = 2)

	return array

##### ##### ##### ##### ##### 

def neighbours(array, yy, xx, size, roll): # 2D array

	# Given an image
	# Return an NxN array whose "center" element is arr[y,x]

	(height, width) = array.shape

	yy_start = yy-size
	yy_end = yy+size+1

	xx_start = xx-size
	xx_end = xx+size+1

	# Check if roll operation can be skipped
	# Note: numpy.roll is slow

	if (0 < yy_start) and (yy_end < height):
		if (0 < xx_start) and (xx_end < width):
			roll = False

	if roll:

		array = numpy.roll(array, shift = 1-yy, axis = 0)
		array = numpy.roll(array, shift = 1-xx, axis = 1)

		span = 2*size+1

		return array[:span, :span]

	else:

		return array[yy_start:yy_end, xx_start:xx_end]
