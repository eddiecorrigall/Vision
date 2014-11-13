import sys

sys.path.append("..")

import numpy
import matplotlib.pyplot as pyplot

import image
import costs

# Baseline matching...
# Horizonal scanline is assumed to be align in both left and right images

def block_matching(
	L_image,
	R_image,
	disparity = 8,
	window = 3,
	roll = True,
	cost_default = numpy.inf,
	cost_function = costs.ssd,
	censor = True, 
	censor_threshold = 2.0,
	show = True):

	# Brute-force matching algorithm!

	L_image = numpy.array(255*image.greyscale(L_image), dtype = numpy.uint8)
	R_image = numpy.array(255*image.greyscale(R_image), dtype = numpy.uint8)

	# ...

	shape = (height, width) = L_image.shape

	window_span		= 1 + 2*window
	disparity_span	= 1 + 2*disparity
	disparity_range	= (0-disparity) + numpy.array(range(disparity_span))
	disparity_map	= (0-disparity) * numpy.ones(shape)

	if roll:
		yy_range = xrange(height)
		xx_range = xrange(width)
	else:
		yy_range = xrange(window, height-window)
		xx_range = xrange(window+disparity, width-window-disparity)

	if show:
		figure, axis = pyplot.subplots()
		pyplot.plot()
		pyplot.hold(True)

	for yy in yy_range:

		percent_complete = numpy.floor(100*(float(1+yy)/height))
		sys.stdout.write("\rProgress: %d%%" % percent_complete)
		sys.stdout.flush()

		if show:
			image.show(disparity_map, block = False)
			figure.canvas.draw()
			axis.cla()

		for xx in xx_range:

			L_window = image.neighbours(L_image, yy, xx, size = window, roll = roll)

			if (censor): # Censorship: Remove areas of low texture...

				# Censor variance along the horizontal scanline, try: { 2, 8, 16, 32, 48 }

				# IMPORTANT:
				# Rows (xx) is axis=1, ie. numpy.sum(W, axis=1) <=> W[:][0]+W[:][1]+W[:][2]
				# Columns (yy) is axis=0, ie. numpy.sum(W, axis=0) <=> W[0][:]+W[1][:]+W[2][:]
				
				scanline_mean = numpy.mean(L_window, axis = 0) # Horizontal scanline mean
				scanline_variance = numpy.mean((L_window - scanline_mean)**2) # Horizontal scanline mean

				if (scanline_variance < censor_threshold):
					continue # Ensure exture quality

			# Find the best disparity match...

			''' #This one-liner doesn't seem to speed things up...
			disparity_map[yy][xx] = (0-disparity) + numpy.array([
				cost_function(L_window, image.neighbours(R, yy, xx+dd, size = window, roll = roll))
				for dd in disparity_range
			]).argmin()
			'''

			cc_best = cost_default
			
			for dd in disparity_range:

				R_window = image.neighbours(R_image, yy, xx+dd, size = window, roll = roll)

				cc = cost_function(L_window, R_window)

				if (cc < cc_best):
					cc_best = cc
					disparity_map[yy][xx] = dd
	
	sys.stdout.write("\r")

	return disparity_map

def remove_inconsistency(L_disparity_map, R_disparity_map, empty_value):

	(height, width) = shape = L_disparity_map.shape

	disparity_map = empty_value*numpy.ones(shape)

	for yy in xrange(height):
		for xx in xrange(width):

			R_x_estimate = xx + L_disparity_map[yy][xx]

			if (R_x_estimate < 0): continue
			if ((R_x_estimate < width) == False): continue

			L_x_estimate = R_x_estimate + R_disparity_map[yy][R_x_estimate];

			if (xx != L_x_estimate): continue # Inconsistent!

			disparity_map[yy][xx] = L_disparity_map[yy][xx]
	
	return disparity_map

if __name__ == "__main__":

	# Example:
	# > python matching.py pm_L.tif pm_R.tif
	# Find disparity maps,
	# One with left image as reference,
	# The other with right image as reference.
	# Then remove inconsistency between the two.

	# Example:
	# > rm *.map
	# Remove caching mechanism to recompute left and right disparities

	import os
	import pickle

	if len(sys.argv) <= 2:
		print("Requires left and right image path as input!")
		exit()

	L_path = sys.argv[1]
	R_path = sys.argv[2]

	L_disparity_map_path = L_path + ".map"
	R_disparity_map_path = R_path + ".map"

	L_image = image.greyscale(image.read(L_path))
	R_image = image.greyscale(image.read(R_path))

	print("Compute disparity map with LEFT image as reference")
	
	if (os.path.exists(L_disparity_map_path)):
		L_disparity_map = pickle.load(open(L_disparity_map_path, "rb"))
	else:
		L_disparity_map = block_matching(L_image, R_image)
		pickle.dump(L_disparity_map, open(L_disparity_map_path, "wb"))

	image.show(L_disparity_map, title = "Left Disparity Map")

	print("Compute disparity map with RIGHT image as reference")

	if (os.path.exists(R_disparity_map_path)):
		R_disparity_map = pickle.load(open(R_disparity_map_path, "rb"))
	else:
		R_disparity_map = block_matching(R_image, L_image)
		pickle.dump(R_disparity_map, open(R_disparity_map_path, "wb"))

	image.show(R_disparity_map, title = "Right Disparity Map")

	print("Remove any inconsistency between both images")
	
	depth_map = remove_inconsistency(L_disparity_map, R_disparity_map, empty_value = 8)
	depth_map = 0-depth_map

	image.show(depth_map, title = "Depth Map")
