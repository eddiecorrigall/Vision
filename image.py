from arrays import *

import numpy
import matplotlib.pyplot as pyplot

GREYSCALE_FILTER_RED		= [1.0, 0.0, 0.0]
GREYSCALE_FILTER_GREEN		= [0.0, 1.0, 0.0]
GREYSCALE_FILTER_BLUE		= [0.0, 0.0, 1.0]

GREYSCALE_FILTER_AVERAGE	= [0.5, 0.5, 0.5]
GREYSCALE_FILTER_LUMINANCE	= [0.2126, 0.7152, 0.0722]

# A simple helper library for computer vision
# Note: image coordinates are (y, x) in (height, width)

def read(path, **keywords):
	image = pyplot.imread(path, **keywords)
	image = as_bytes(image)
	return image

def write(path, image, **keywords):
	pyplot.imsave(path, image, **keywords)

def show(image, title = None, **keywords):

	if title is not None:
		pyplot.title(title)

	# TODO: Is it faster to convert as_bytes, then draw?

	pyplot.imshow(image)
	pyplot.show(**keywords) # ie. show(image, block = True)

def is_greyscale(image):
	return (len(image.shape) == 2)

def as_greyscale(image, rgb_filter = GREYSCALE_FILTER_LUMINANCE):

	# Save image type

	image_type = image.dtype

	# Convert to greyscale

	if (is_greyscale(image) == False):
		image = normalize(image)
		image = numpy.dot(image[..., range(3)], rgb_filter)

	# Restore image type

	if (is_bytes(image_type)):
		image = as_bytes(image)
	
	return image
