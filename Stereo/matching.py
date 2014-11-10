import sys

sys.path.append("..")

import numpy
import image
import costs

# Baseline matching...
# Horizonal scanline is assumed to be align in both left and right images

def block_matching(L, R):

	# Brute-force matching algorithm!

	L = 255*image.greyscale(L)
	R = 255*image.greyscale(R)

	cost_function = costs.ssd
	disparity = 8
	window = 3
	sigma = 2

	# ...

	shape = (height, width) = L.shape
	disparity_map = disparity*numpy.ones(shape)
	
	disparity_span = 1 + 2*disparity
	disparity_range = numpy.array(range(disparity_span)) - disparity

	window_span = 1 + 2*window
	window_range = numpy.array(range(window_span)) - window

	for yy in xrange(window, (height-window)):
		
		percent = numpy.floor(100*(float(1+yy)/height))
		sys.stdout.write("\rProgress: %d%%" % percent)
		sys.stdout.flush()

		for xx in xrange(window, (width-window)):

			yy_range = yy+window_range
			xx_range = xx+window_range

			L_window = numpy.transpose(L[yy_range])[xx_range] # Left window

			# Censoring: Remove areas of low texture...

			# IMPORTANT:
			# Rows (xx) is axis=1, ie. numpy.sum(W, axis=1) <=> W[:][0]+W[:][1]+W[:][2]
			# Columns (yy) is axis=0, ie. numpy.sum(W, axis=0) <=> W[0][:]+W[1][:]+W[2][:]
			
			scanline_mean = numpy.mean(L_window, axis = 0) # Horizontal scanline mean
			scanline_variance = numpy.mean((L_window - scanline_mean)**2) # Horizontal scanline mean

			if (scanline_variance < sigma): continue # Ensure exture quality

			# Find the best disparity match...

			cc_best = numpy.inf

			for dd in disparity_range:

				dd_range = dd+xx_range

				if dd_range[0] < 0: continue
				if dd_range[-1] >= width: break

				R_window = numpy.transpose(R[yy_range])[dd_range] # Right window

				cc = cost_function(L_window, R_window)

				if (cc < cc_best):
					cc_best = cc
					disparity_map[yy][xx] = dd
	
	disparity_map = (0 - disparity_map) # Invert

	return disparity_map

if __name__ == "__main__":

	# Example...
	# Produce a depth map between to baseline stereo images
	# ie. left and right are aligned to a common horizontal scanline

	# Light shades are closest
	# Dark shades are far
	# Black indicates no information

	import sys
	import costs

	if len(sys.argv) <= 2:
		print("Requires left and right image path as input!")
		exit()

	L_path = sys.argv[1]
	R_path = sys.argv[2]

	L = image.greyscale(image.read(L_path))
	R = image.greyscale(image.read(R_path))

	D = block_matching(L, R)
	D = image.normalize(D)

	#image.write("depth.png", D)
	image.show(D)
