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

def rectify_images(img1, img2, x1, x2, K, d, F, shearing = True):

	# Rectification based on found fundamental matrix

	image_shape = image1.shape
	(height, width) = image_shape
	image_size = (width, height) # Note: image_size is not the same as image_shape

	# TODO: Apply a filter on both images to improve feature detection/matching
	# See: http://stackoverflow.com/questions/19361448/how-to-improve-features-detection-in-opencv
	# * Equalize Histogram filter (absolutely)
	# * Fast Fourier Transform filter (maybe)

	# TODO: Implement code for retval
	retval, H1, H2 = cv2.stereoRectifyUncalibrated(x1.ravel(), x2.ravel(), F, image_size)

	# Apply a shearing transform to homography matrices
	if shearing:
		S = rectify_shearing(H1, H2, width, height)
		H1 = S.dot(H1)
	
	# Find the rectify transform
	K_inverse = numpy.linalg.inv(K)
	rH = K_inverse.dot(H1).dot(K)
	lH = K_inverse.dot(H2).dot(K)

	# TODO: Determine the correct order for lH, rH and img1, img2
	map1x, map1y = cv2.initUndistortRectifyMap(K, d, rH, K, image_size, cv2.CV_16SC2)
	map2x, map2y = cv2.initUndistortRectifyMap(K, d, lH, K, image_size, cv2.CV_16SC2)

	# TODO: Rewrite below, we dont need an alpha channel

	# Convert the images to RGBA (add an axis with 4 values)
	img1 = numpy.tile(img1[:, :, numpy.newaxis], [1, 1, 4])
	img1[:, :, 3] = 255

	img2 = numpy.tile(img2[:, :, numpy.newaxis], [1, 1, 4])
	img2[:, :, 3] = 255

	# cv2.INTER_LINEAR, cv2.INTER_CUBIC
	rimg1 = cv2.remap(img1, map1x, map1y,
		interpolation = cv2.INTER_CUBIC,
		borderMode = cv2.BORDER_CONSTANT,
		borderValue = (0, 0, 0, 0))
	
	rimg2 = cv2.remap(img2, map2x, map2y,
		interpolation = cv2.INTER_CUBIC,
		borderMode = cv2.BORDER_CONSTANT,
		borderValue = (0, 0, 0, 0))
	
	# Set the background to be red
	# TODO: Remove aliasing around borders

	color = (255, 0, 0, 255)
	k = len(color)

	# Eddie...
	rmask = numpy.ones(image_shape + (k-1,), dtype = numpy.bool)
	rmask[rimg1[:, :, k-1] == 0,:] = False
	rmask[rimg2[:, :, k-1] == 0,:] = False

	rimg1[rimg1[:, :, k-1] == 0,:] = color
	rimg2[rimg2[:, :, k-1] == 0,:] = color

	return rimg1, rimg2, rmask

def rectify_with_sift(image1, image2, K, d = None):

	##### ##### ##### ##### ##### 
	##### CREDIT
	##### ##### ##### ##### ##### 

	# OpenCV-Python Tutorials - Camera Calibration and 3D Reconstruction
	# 	* http://docs.opencv.org/trunk/doc/py_tutorials/py_calib3d/py_epipolar_geometry/py_epipolar_geometry.html

	##### ##### ##### ##### ##### 

	# Let K be the OpenCV Camera Matrix
	# Let d be the OpenCV Distortion coefficients

	if (image1.shape != image2.shape):
		print("Image 1 & 2 must have the same shape!")
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

	# Find keypoints and descriptors with SIFT

	sift = cv2.SIFT()
	keypoints1, descriptors1 = sift.detectAndCompute(image1, mask = None)
	keypoints2, descriptors2 = sift.detectAndCompute(image2, mask = None)

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
	mask = mask.ravel()
	x1 = x1[mask == 1]
	x2 = x2[mask == 1]

	##### ##### ##### ##### ##### 
	##### Rectify Images
	##### ##### ##### ##### ##### 

	# TOOD,
	# determine which image is left and which is right... using keypoints/descriptors?

	return rectify_images(image1, image2, x1, x2, K, d, F)

if (__name__ == '__main__'):

	##### ##### ##### ##### ##### 
	##### EXAMPLE
	##### ##### ##### ##### ##### 

	import image

	import os
	import matplotlib.pyplot as pyplot

	example = "Examples/unrectified/girl/" # So far the only example
	
	if 1 < len(sys.argv):
		example = sys.argv[1]

	image1 = cv2.imread(os.path.join(example, "image1.png"), cv2.CV_LOAD_IMAGE_GRAYSCALE)
	image2 = cv2.imread(os.path.join(example, "image2.png"), cv2.CV_LOAD_IMAGE_GRAYSCALE)

	cameraMatrix = numpy.load(os.path.join(example + "K.npy"))
	distortionParameters = numpy.load(os.path.join(example + "d.npy"))

	#cameraMatrix = numpy.float64([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
	#distortionParameters = None

	rectified1, rectified2, mask = rectify_with_sift(image1, image2, K = cameraMatrix, d = distortionParameters)

	# Save data

	numpy.save("rectify_mask.npy", mask) # Used for cropping final image
	image.write("rectify_1.png", rectified1)
	image.write("rectify_2.png", rectified2)
	
	# Display data
	
	combined = normalize(numpy.float32(rectified1) + numpy.float32(rectified2))

	pyplot.figure()
	pyplot.imshow(combined)
	pyplot.show()
