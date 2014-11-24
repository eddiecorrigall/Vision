from arrays import *

import numpy

# Matching cost functions
# Assumming a and b are the same size
# Returns a scalar representing the cost

def ssd(a, b): # Sum of the the Square Differences

	# Range: non-negative
	# 0.0 is maximal positive correlation

	if (a.size == 0) or (a.shape != b.shape):
		return numpy.nan

	return numpy.sum((b - a)**2)

def sad(a, b): # Sum of the Absolute Differences
	
	# Range: non-negative
	# 0.0 is maximal positive correlation

	if (a.size == 0) or (a.shape != b.shape):
		return numpy.nan

	return numpy.sum(numpy.abs(b - a))

def ncc(a, b): # Normalized Cross-Correlation

	# Range: [-1, 1]
	# 0.0 is zero correlation
	# 1.0 is maximal positive correlation
	# -1.0 is maximal negative correlation

	# MATH:
	# a_centroid = a - mean(a)
	# b_centroid = b - mean(b)
	# dot(a_centroid, b_centroid) / sqrt(ssd(a_centroid)*ssd(b_centroid))

	if (a.size == 0) or (a.shape != b.shape):
		return numpy.nan

	a = a.flatten()
	b = b.flatten()

	a_centroid = a - numpy.mean(a)
	b_centroid = b - numpy.mean(b)

	# numerator = numpy.sum(a_centroid*b_centroid)
	numerator = numpy.dot(a_centroid, b_centroid)

	if is_nearly_zero(numerator):
		return 0

	# denominator = numpy.sqrt(numpy.sum(a_centroid**2)*numpy.sum(b_centroid**2))
	denominator = numpy.sqrt(numpy.dot(a_centroid, a_centroid)*numpy.dot(b_centroid, b_centroid))

	if is_nearly_zero(denominator):
		return numpy.nan

	return (numerator/denominator)
