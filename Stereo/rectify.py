"""
Python rectification implementation
"""

import cv2
import numpy

import sys
sys.path.append("..")

from arrays import *

##### ##### ##### ##### ##### 
##### VARIABLE DEFINITIONS
##### ##### ##### ##### ##### 

# K		Camera matrix
# d		Distortion parameters
# x1	Feature points in image1
# x2	Corresponding feature points in image2
# F		Fundamental matrix
# H1	Homography matrix transform for image1
# H2	Homography matrix transform for image2
# R1	Rectification matrix transform for image1
# R2	Rectification matrix transform for image2

##### ##### ##### ##### ##### 

def rectify_shearing(H1, H2, image_width, image_height):

	##### ##### ##### ##### ##### 
	##### CREDIT
	##### ##### ##### ##### ##### 

	# Loop & Zhang - via literature
	#	* http://scicomp.stackexchange.com/questions/2844/shearing-and-hartleys-rectification
	# TH. - via stackexchange user
	# 	* http://scicomp.stackexchange.com/users/599/th
	#	* http://scicomp.stackexchange.com/questions/2844/shearing-and-hartleys-rectification

	##### ##### ##### ##### ##### 
	##### PARAMETERS
	##### ##### ##### ##### ##### 

	# Let H1 be the rectification homography of image1 (ie. H1 is a homogeneous space)
	# Let H2 be the rectification homography of image2 (ie. H2 is a homogeneous space)
	# image_width, image_height be the dimensions of both image1 and image2

	##### ##### ##### ##### ##### 

	"""
	Compute shearing transform than can be applied after the rectification transform to reduce distortion.
	Reference:
		http://scicomp.stackexchange.com/questions/2844/shearing-and-hartleys-rectification
		"Computing rectifying homographies for stereo vision" by Loop & Zhang
	"""

	w = image_width
	h = image_height

	'''
	Loop & Zhang use a shearing transform to reduce the distortion
	introduced by the projective transform that mapped the epipoles to infinity
	(ie, that made the epipolar lines parallel).

	Consider the shearing transform:

			| k1 k2 0 |
	S	=	| 0  1  0 |
			| 0  0  1 |

	Let w and h be image width and height respectively.
	Consider the four midpoints of the image edges:
	'''

	a = numpy.array([ (w-1)/2.0,	0,			1 ], dtype = float)
	b = numpy.array([ (w-1),		(h-1)/2.0,	1 ], dtype = float)
	c = numpy.array([ (w-1)/2.0,	(h-1),		1 ], dtype = float)
	d = numpy.array([ 0,			(h-1)/2.0,	1 ], dtype = float)

	'''
	According to Loop & Zhang:
	"... we attempt to preserve perpendicularity and aspect ratio of the lines bd and ca"
	'''

	'''
	Let H be the rectification homography and,
	Let a' = H*a be a point in the affine plane by dividing through so that a'2 = 1
	Note: a'2 is the third component, ie, a' = (a'[0], a'1, a'2))
	'''

	# Note: *.dot is a form of matrix*vector multiplication in numpy
	# So a_prime = H*a such that a_prime[2] = 1 (hence the use of homogeneous_to_euclidean function)

	a_prime = homogeneous_to_euclidean(H1.dot(a))
	b_prime = homogeneous_to_euclidean(H1.dot(b))
	c_prime = homogeneous_to_euclidean(H1.dot(c))
	d_prime = homogeneous_to_euclidean(H1.dot(d))

	''' Let x = b' - d' and y = c' - a' '''

	x = b_prime - d_prime
	y = c_prime - a_prime

	'''
	According to Loop & Zhang:
		"As the difference of affine points, x and y are vectors in the euclidean image plane.
			Perpendicularity is preserved when (Sx)^T(Sy) = 0, and aspect ratio is preserved if [(Sx)^T(Sx)]/[(Sy)^T(Sy)] = (w^2)/(h^2)"
	'''

	''' The real solution presents a closed-form: '''

	k1 = (h*h*x[1]*x[1] + w*w*y[1]*y[1]) / (h*w*(x[1]*y[0] - x[0]*y[1]))
	k2 = (h*h*x[0]*x[1] + w*w*y[0]*y[1]) / (h*w*(x[0]*y[1] - x[1]*y[0]))

	''' Determined by sign (the positive is preferred) '''

	if (k1 < 0): # Why this?
		k1 *= -1
		k2 *= -1

	return numpy.array([
		[k1,	k2,	0],
		[0,		1,	0],
		[0,		0,	1]], dtype = float)

##### ##### ##### ##### ##### 

def rectify_images(image1, image2, x1, x2, F, K, d, shearing = True):

	# Rectification based on found Fundamental matrix

	image_shape = image1.shape
	(height, width) = image_shape
	image_size = (width, height) # Note: image_size is not image_shape

	# Calculate Homogeneous matrix transform given features and fundamental matrix

	retval, H1, H2 = cv2.stereoRectifyUncalibrated(x1.ravel(), x2.ravel(), F, image_size)

	if (retval == False):
		print("ERROR: stereoRectifyUncalibrated failed")
		return None

	# Apply a shearing transform to homography matrices
	if shearing:
		S = rectify_shearing(H1, H2, width, height)
		H1 = S.dot(H1)
	
	# Compute the rectification transform
	K_inverse = numpy.linalg.inv(K)
	R1 = K_inverse.dot(H1).dot(K)
	R2 = K_inverse.dot(H2).dot(K)

	mapx1, mapy1 = cv2.initUndistortRectifyMap(K, d, R2, K, image_size, cv2.CV_16SC2)
	mapx2, mapy2 = cv2.initUndistortRectifyMap(K, d, R1, K, image_size, cv2.CV_16SC2)

	# Find an unused colour to build a border mask
	# Note: Assuming that the union of both image intensity sets do not exhaust the 8 bit range
	# Fortunately, if the set is empty, set.pop() will throw a runtime error

	palette1 = set(image1.flatten())
	palette2 = set(image2.flatten())

	colours = set(range(256))

	key1 = colours.difference(palette1).pop()
	key2 = colours.difference(palette2).pop()

	##### ##### ##### ##### ##### 
	##### Apply Rectification Transform
	##### ##### ##### ##### ##### 

	rectified1 = cv2.remap(image1, mapx1, mapy1,
		interpolation	= cv2.INTER_LINEAR, # cv2.INTER_CUBIC, # cv2.INTER_LINEAR
		borderMode		= cv2.BORDER_CONSTANT,
		borderValue		= key1)
	
	rectified2 = cv2.remap(image2, mapx2, mapy2,
		interpolation	= cv2.INTER_LINEAR,
		borderMode		= cv2.BORDER_CONSTANT,
		borderValue		= key2)

	# Build the mask

	mask = numpy.ones(image_shape, dtype = bool)
	mask[rectified1 == key1] = False
	mask[rectified2 == key2] = False

	return rectified1, rectified2, mask

def rectify_with_sift(image1, image2, K = numpy.eye(3), d = None):

	##### ##### ##### ##### ##### 
	##### CREDIT
	##### ##### ##### ##### ##### 

	# OpenCV-Python Tutorials - Camera Calibration and 3D Reconstruction
	# 	* http://docs.opencv.org/trunk/doc/py_tutorials/py_calib3d/py_epipolar_geometry/py_epipolar_geometry.html

	##### ##### ##### ##### ##### 

	# Let K be the OpenCV Camera Matrix
	# Let d be the OpenCV Distortion coefficients

	if (image1.shape != image2.shape):
		print("ERROR: Image 1 & 2 must have the same shape!")
		return None

	shape = image1.shape
	(height, width) = shape

	print("Camera Matrix:")
	print(str(K))

	print("Distortion parameters:")
	print(str(d))

	##### ##### ##### ##### ##### 
	##### Compute Matching
	##### ##### ##### ##### ##### 

	# Apply a filter to both images to improve feature detection/matching
	# Reference: http://stackoverflow.com/questions/19361448/how-to-improve-features-detection-in-opencv
	# * Equalize Histogram filter (absolutely)
	# * Fast Fourier Transform filter (maybe)

	# Apply Equalize Histogram filter
	# Note:
	# There is an improvement in matching, but also less information...
	# It is hard to conclude this improves GCS

	filtered1 = cv2.equalizeHist(image1)
	filtered2 = cv2.equalizeHist(image2)

	# Find keypoints and descriptors with SIFT

	sift = cv2.SIFT()
	keypoints1, descriptors1 = sift.detectAndCompute(filtered1, mask = None)
	keypoints2, descriptors2 = sift.detectAndCompute(filtered2, mask = None)

	# Match features using FLANN based matching

	FLANN_INDEX_KDTREE = 0
	flann = cv2.FlannBasedMatcher(
		indexParams = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5),
		searchParams = dict(checks = 50))

	matches = flann.knnMatch(descriptors1, descriptors2, k = 2)

	# Ratio test as per Lowe's paper

	x1 = []
	x2 = []

	for i,(m,n) in enumerate(matches):
		if m.distance < 0.8*n.distance:
			x1.append(keypoints1[m.queryIdx].pt)
			x2.append(keypoints2[m.trainIdx].pt)

	x1 = numpy.float64(x1)
	x2 = numpy.float64(x2)

	##### ##### ##### ##### ##### 
	##### Compute Fundamental matrix
	##### ##### ##### ##### ##### 

	F, mask = cv2.findFundamentalMat(x1, x2)
	
	# Select only inlier points
	mask = mask.flatten()
	x1 = x1[mask == 1]
	x2 = x2[mask == 1]

	##### ##### ##### ##### ##### 
	##### Rectify Images
	##### ##### ##### ##### ##### 

	# TOOD,
	# Determine which image is left and which is right...
	# Using keypoints/descriptors?

	return rectify_images(image1, image2, x1, x2, F, K, d)

if (__name__ == "__main__"):

	import os
	import matplotlib.pyplot as pyplot

	##### ##### ##### ##### ##### 
	##### EXAMPLE
	##### ##### ##### ##### ##### 

	example = "Examples/girl/" # So far the only COMPLETE example
	
	if 1 < len(sys.argv):
		example = sys.argv[1]

	for filename in os.listdir(example):
		if filename.startswith("image1"):
			image1_path = filename
		if filename.startswith("image2"):
			image2_path = filename

	image1 = cv2.imread(os.path.join(example, image1_path), cv2.CV_LOAD_IMAGE_GRAYSCALE)
	image2 = cv2.imread(os.path.join(example, image2_path), cv2.CV_LOAD_IMAGE_GRAYSCALE)

	# Load camera matrix

	cameraMatrix = numpy.eye(3)

	cameraMatrix_path = os.path.join(example, "K.npy")
	if os.path.isfile(cameraMatrix_path):
		cameraMatrix = numpy.load(cameraMatrix_path)

	# Load distortion parameters

	distortionParameters = None

	distortionParameters_path = os.path.join(example, "d.npy")
	if os.path.isfile(distortionParameters_path):
		distortionParameters = numpy.load(distortionParameters_path)

	# Rectify images

	rectified1, rectified2, mask = rectify_with_sift(image1, image2,
		K = cameraMatrix,
		d = distortionParameters)

	# Save data

	numpy.save("mask.npy", mask) # Used to crop final image

	imwrite_parameters = (cv2.IMWRITE_PNG_COMPRESSION, 9)
	cv2.imwrite("rectified1.png", rectified1, imwrite_parameters)
	cv2.imwrite("rectified2.png", rectified2, imwrite_parameters)

	# Display data
	
	combined = normalize(numpy.float32(rectified1) + numpy.float32(rectified2))

	pyplot.figure()
	pyplot.imshow(combined)
	pyplot.show()
