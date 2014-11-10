import numpy

# Matching cost functions
# Assumming a and b are the same size
# Returns a scalar representing the cost

def ncc(a, b): # Normalized Cross Correlation

	if (a.size == 0) or (a.size != b.size):
		return numpy.nan

	a_sigma = numpy.std(a)
	b_sigma = numpy.std(b)

	denominator = (a_sigma*b_sigma)

	if (0 < numpy.abs(denominator)):
		a_centred = (a - numpy.mean(a))
		b_centred = (b - numpy.mean(b))
		return numpy.mean(a_centred*b_centred) / denominator
	
	return numpy.nan

def ssd(a, b): # Sum of the the Square Difference

	if (a.size == 0) or (a.size != b.size):
		return numpy.nan

	return numpy.sum((a - b)**2)

def sad(a, b): # Sum of the Absolute Differences
	
	if (a.size == 0) or (a.size != b.size):
		return numpy.nan

	return numpy.sum(numpy.abs(a - b))
