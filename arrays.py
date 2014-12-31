import numpy
import scipy.ndimage as ndimage

##### ##### ##### ##### ##### 

EPSILON = numpy.finfo(numpy.float16).eps # Use a lo-rez float to be safe

FLOATS = numpy.float32 # Note: Change to tune accuracy and performace
FLOATS_INFO	= numpy.finfo(FLOATS)

BYTES = numpy.uint8 # Important: This must never change!
BYTES_INFO = numpy.iinfo(BYTES)

##### ##### ##### ##### ##### 

def is_nearly(x, y):
	# Given x and y as numbers, or arrays.
	# Check if all values of x are nearly equal to the corresponding y value
	return numpy.all(numpy.abs(y - x) <= EPSILON)

def is_nearly_zero(x):
	return numpy.all(numpy.abs(x) <= EPSILON)

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

	if isinstance(x, type) or isinstance(x, numpy.dtype):
		return x == t
	elif type(x) == numpy.ndarray:
		return x.dtype == t
	else:
		return isinstance(x, t)

##### ##### ##### ##### ##### 

def is_bytes(x):
	return is_similar_type(x, BYTES)

def as_bytes(x):

	# Important:
	# This function is not guaranteed to be reversable.
	# The data might be scaled, or translated to reduce data loss.
	# It is most useful for converting float-images to byte-images.

	if is_bytes(x):
		return x

	x = normalize(x) # Scale to [0, 1]
	
	# Story time:
	# When rectifying images,
	# I found that the transformations were inconsistent between FLOATS types.
	# I determined that the best rectification results were obtained with numpy.float16.
	# While debugging I noticed small differences were propagated from within as_bytes.
	# When FLOATS = numpy.float16, numpy.float32, ... small differences in pixel value occured between these types,
	# Which propagated to a feature detector that I used to help determine the transformation.
	# Some of these issues were resolved when I ironed out a few bugs, but the problem was obviously in as_bytes.
	# My solution was to use numpy.around.
	# I understand that when floating operations are done,
	# Errors can accumulate in least-significant decimals and overflow into most-significant decimals.
	# So numpy.around is used to trim off these anomalies at the 4th decimal.
	# The 4th decimal was selected because of the the storage capacity of the unsigned char / uint8 / byte.
	# We need at most 4 decimals for every byte value:
	# 	1/256 ~= 0.0039 => 4 decimal precision

	x = numpy.around(x, decimals = 4) # TODO: find out how to programmatically convert BYTES type to number of decimals
	x = x*BYTES_INFO.max # Scale to [0, max]
	x = numpy.array(x, dtype = BYTES)

	return x

##### ##### ##### ##### ##### 

def is_floats(x):
	return is_similar_type(x, FLOATS)

def as_floats(x):

	if is_floats(x):
		return x

	# Capacity is assumed to be large enough
	# If this is cast down from larger floating point types,
	# Truncation will occur.
	x = numpy.array(x, dtype = FLOATS)

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

def normalize(x):

	x = numpy.array(x)

	x_min = numpy.min(x)
	x_max = numpy.max(x)
	x_range = x_max - x_min

	x -= x_min

	# Prevent division by zero:
	# Simply check if the value is greater
	# Note: x_range guaranteed to be non-negative
	if (x_range > 0):
		# Reduce rounding error:
		# Only mess with the numbers if they are in voilation of domain
		if (x_range < 1):
			return x
		else:
			x = as_floats(x)
			x /= x_range
	
	return x

##### ##### ##### ##### ##### 

def is_vector(x):

	#return (len(x) == len(x.ravel()))
	
	dimension = len(x.shape)

	if (dimension == 1): return True
	if (dimension == 2):
		if (x.shape[0] == 1) or (x.shape[1] == 1):
			return True
		return False
	return False

def as_vector(x):
	return x.ravel()

##### ##### ##### ##### ##### 

def euclidean_to_homogeneous(x):

	"""
	See: OpenCV - convertPointsToHomogeneous
	The function converts points from Euclidean to homogeneous space by appending 1's to the tuple of point coordinates.
	That is, each point (x1, x2, ..., xn) is converted to (x1, x2, ..., xn, 1).
	"""
	
	if is_vector(x):
		return numpy.r_[x, 1]
	else:
		S = (1,) + x.shape[1:]
		return numpy.r_[x, numpy.ones(S)]

def homogeneous_to_euclidean(x):

	"""
	See: OpenCV - convertPointsFromHomogeneous
	The function converts points homogeneous to Euclidean space using perspective projection.
	That is, each point (x1, x2, ... x(n-1), xn) is converted to (x1/xn, x2/xn, ..., x(n-1)/xn).
	When xn=0, the output point coordinates will be (0,0,0,...).
	"""

	# Faster method: avoid dividing everything

	y = as_floats(x) # Make a copy of x and convert to FLOATS

	if is_vector(y):
		y[-1] = numpy.nan if (y[-1] == 0) else y[-1]
		y[:-1] /= y[-1]
		y[numpy.isnan(y)] = 0
		y[-1] = 0 if (y[-1] == 0) else 1
	else:
		y[-1, y[-1] == 0] = numpy.nan	# Replace 0s in end row with nan to prevent division by zero
		y[:-1] /= y[-1]					# Divide all rows excluding end row, by end row
		y[numpy.isnan(y)] = 0			# Replace all nan with 0s
		y[-1, y[-1] != 0] = 1			# Replace end row with 1s if the value is not zero

	return y

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

def zero_pad(x, padding):
	return numpy.pad(x, pad_width = padding, mode = 'constant')
