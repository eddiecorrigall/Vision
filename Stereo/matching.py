import sys

sys.path.append("..")

import numpy
import image
import costs

# Baseline matching...
# Horizonal scanline is assumed to be align in both left and right images

def block_matching(L, R):

	# Brute-force matching algorithm!

	L = numpy.array(255*image.greyscale(L), dtype = numpy.uint8)
	R = numpy.array(255*image.greyscale(R), dtype = numpy.uint8)

	cost_function = costs.ssd
	disparity = 8
	window = 2
	sigma = 2 # Try: { 8, 16, 32, 48 }

	# ...

	shape = (height, width) = L.shape

	window_span = 1 + 2*window
	disparity_span = 1 + 2*disparity
	disparity_range = numpy.array(range(disparity_span)) - disparity
	disparity_map = (0-disparity)*numpy.ones(shape)

	for yy in xrange(height):
		
		percent = numpy.floor(100*(float(1+yy)/height))
		sys.stdout.write("\rProgress: %d%%" % percent)
		sys.stdout.flush()

		for xx in xrange(width):

			L_window = image.neighbours(L, yy, xx, N = window_span)

			# Censorship: Remove areas of low texture...

			# IMPORTANT:
			# Rows (xx) is axis=1, ie. numpy.sum(W, axis=1) <=> W[:][0]+W[:][1]+W[:][2]
			# Columns (yy) is axis=0, ie. numpy.sum(W, axis=0) <=> W[0][:]+W[1][:]+W[2][:]
			
			scanline_mean = numpy.mean(L_window, axis = 0) # Horizontal scanline mean
			scanline_variance = numpy.mean((L_window - scanline_mean)**2) # Horizontal scanline mean

			if (scanline_variance < sigma): continue # Ensure exture quality

			# Find the best disparity match...

			cc_best = numpy.inf
			
			for dd in disparity_range:

				zz = xx+dd

				if (zz < 0): continue
				if ((zz < width) == False): break

				R_window = image.neighbours(R, yy, zz, N = window_span)

				cc = cost_function(L_window, R_window)

				if (cc < cc_best):
					cc_best = cc
					disparity_map[yy][xx] = dd
	
	sys.stdout.write("\n")

	##disparity_map = (0 - disparity_map) # Invert

	return disparity_map

def remove_inconsistency(L_disparity_map, R_disparity_map, empty_value):

	(height, width) = shape = L.shape

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

	L = image.greyscale(image.read(L_path))
	R = image.greyscale(image.read(R_path))

	print("Compute disparity map with LEFT image as reference")
	
	if (os.path.exists(L_disparity_map_path)):
		L_disparity_map = pickle.load(open(L_disparity_map_path, "rb"))
	else:
		L_disparity_map = block_matching(L, R)
		pickle.dump(L_disparity_map, open(L_disparity_map_path, "wb"))

	image.show(L_disparity_map)

	print("Compute disparity map with RIGHT image as reference")

	if (os.path.exists(R_disparity_map_path)):
		R_disparity_map = pickle.load(open(R_disparity_map_path, "rb"))
	else:
		R_disparity_map = block_matching(R, L)
		pickle.dump(R_disparity_map, open(R_disparity_map_path, "wb"))

	image.show(R_disparity_map)

	print("Remove any inconsistency between both images")
	
	disparity_map = remove_inconsistency(L_disparity_map, R_disparity_map, empty_value = -8)
	image.show(disparity_map)
