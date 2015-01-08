import sys
import numpy
import matplotlib.pyplot as pyplot

from Queue import PriorityQueue

sys.path.append("..")

import arrays
import costs

MAX_WIDTH = 512

# TODO:
# * Implement GrowingCorrespondenceSeeds class with some kind of StereoMatching interface
# * Set gcs.image1 to the input that covers the largest area in mask
#	- Because the reference image should have the most matching potential?
#	- See Examples/girl, better matching if image1 = Examples/girl/rectified2.png
# * Define preemptiveMatch according to literature
# * Why does "python gcs.py Examples/girl 79 4" work while, "python gcs.py Examples/girl 80 4" does not?

##### ##### ##### ##### ##### 

class Seed:
	
	def __init__(self, y, x1, x2): # y, x, x'
		self.y = y
		self.x1 = x1
		self.x2 = x2

	def getDisparity(self):
		return (self.x2 - self.x1)

	def getNeighbour(self, y_delta, x1_delta, x2_delta):
		return Seed(self.y+y_delta, self.x1+x1_delta, self.x2+x2_delta)

	def __str__(self):
		return "Seed" + '(' + str(self.y) + ',' + str(self.x1) + ',' + str(self.x2) + ')'

##### ##### ##### ##### ##### 

class GrowingCorrespondenceSeeds:

	def __init__(self, shape, disparity, window = 3, initialSeeds = None, tau = 0.6, mu = 0.05):

		# disparity is a non-negative number
		# window is a non-negative number
		# tau is similarity threshold
		# mu is a threshold slope
		# initialSeeds is a list of seeds

		(self.height, self.width) = shape

		self.disparity = disparity
		self.window = window

		self.initialSeeds = initialSeeds
		self.tau = tau
		self.mu = mu

		# Best disparities
		self.D1 = numpy.ndarray(shape)
		self.D2 = numpy.ndarray(shape)

		# Best costs
		self.C1	= numpy.ndarray(shape)
		self.C2 = numpy.ndarray(shape)

		# IMPORTANT:
		# Tau should normally be implimented by a 2D array of ArrayLists
		# However, since python list insert is O(n), then binary search is also O(n), and so it is inefficient in Python
		# Using a matrix that is pre-allocated which exhausts the search space size is very costly in terms of space complexity,
		# But very efficient in terms of runtime complexity

		self.correspondenceCount = 0
		self.correspondenceTable = numpy.ndarray([self.height, self.width, arrays.span(self.disparity)], dtype = bool) # aka TAU[:,:,:] = False
		self.similarityTable = numpy.ndarray([self.height, self.width, arrays.span(self.disparity)])

	def isValidSeed(self, seed):
		tt = self.getTableIndex(seed, validate = False)
		return (0 <= tt) and (tt < arrays.span(self.disparity))

	def getTableIndex(self, seed, validate = True): # Runtime complexity is O(1)
		# Map seed.getDisparity() to [0, DISPARITY_SPAN)
		if validate: assert self.isValidSeed(seed)
		return self.disparity+seed.getDisparity()

	def clearCorrespondenceTable(self):
		self.correspondenceCount = 0
		self.correspondenceTable.fill(False)

	def containsCorrespondence(self, seed): # Runtime complexity is O(1)
		tt = self.getTableIndex(seed)
		return self.correspondenceTable[seed.y][seed.x1][tt]

	def insertCorrespondence(self, seed): # Runtime complexity is O(1)
		tt = self.getTableIndex(seed)
		self.correspondenceTable[seed.y][seed.x1][tt] = True
		self.correspondenceCount += 1

	def clearSimilarityTable(self):
		self.similarityTable.fill(numpy.nan)
	
	def getSimilarity(self, seed):

		tt = self.getTableIndex(seed)
		cc = self.similarityTable[seed.y][seed.x1][tt]

		if numpy.isfinite(cc):
			return cc

		window1 = arrays.neighbours(self.image1, seed.y, seed.x1, size = self.window, roll = False)
		window2 = arrays.neighbours(self.image2, seed.y, seed.x2, size = self.window, roll = False)

		cc = costs.ncc(window1, window2)
		self.similarityTable[seed.y][seed.x1][tt] = cc

		return cc

	def pushSeed(self, seed):
		cc = self.getSimilarity(seed)
		self.seeds.put( (0-cc, seed) ) # IMPORTANT: PriorityQueue is a min priority queue, but the best similarity is maximal

	def popSeed(self):
		return None if self.seeds.empty() else self.seeds.get()[1]

	def preemptiveMatching(self):

		# TODO:
		# This is not the same as described in literature

		print("GCS Preemptive Matching...")

		initial_seed_count = int(4.0*numpy.sqrt(self.height*self.width))

		yy_margin = self.window
		yy_range = numpy.random.randint(
			low		= yy_margin,
			high	= self.height-yy_margin,
			size	= initial_seed_count)
		
		xx_margin = self.window+self.disparity
		xx_range = numpy.random.randint(
			low		= xx_margin,
			high	= self.width-xx_margin,
			size	= initial_seed_count)
		
		dd_range = numpy.array(range(2*self.disparity))-self.disparity

		for ii in xrange(initial_seed_count):

			percent_complete = numpy.floor(100*(float(1+ii)/initial_seed_count))
			sys.stdout.write("\r\tProgress: %d%%" % percent_complete)
			sys.stdout.flush()

			yy = yy_range[ii]
			xx = xx_range[ii]

			cc_range = [ self.getSimilarity( Seed(yy, xx, xx+dd) ) for dd in dd_range ]

			best = numpy.argmax(cc_range)
			best_cc	= cc_range[best]
			best_dd = dd_range[best]

			if (best_cc < 0.9): continue

			self.pushSeed( Seed(yy, xx, xx+best_dd) )

		sys.stdout.write("\r\n")

	def getBestNeighbours(self, seed):

		neighbourhoods = [
			[
				seed.getNeighbour(+0, -1, -1),
				seed.getNeighbour(+0, -2, -1),
				seed.getNeighbour(+0, -1, -2),
			], [
				seed.getNeighbour(+0, +1, +1),
				seed.getNeighbour(+0, +2, +1),
				seed.getNeighbour(+0, +1, +2)
			], [
				seed.getNeighbour(-1, +0, +0),
				seed.getNeighbour(-1, -1, +0),
				seed.getNeighbour(-1, +1, +0),
				seed.getNeighbour(-1, +0, -1),
				seed.getNeighbour(-1, +0, +1)
			], [
				seed.getNeighbour(+1, +0, +0),
				seed.getNeighbour(+1, -1, +0),
				seed.getNeighbour(+1, +1, +0),
				seed.getNeighbour(+1, +0, -1),
				seed.getNeighbour(+1, +0, +1)
			]
		]

		best_neighbours = []

		for neighbourhood in neighbourhoods:

			best_seed = None
			best_similarity = -numpy.inf

			for neighbour_seed in neighbourhood:

				if self.isValidSeed(neighbour_seed) == False: continue

				if (neighbour_seed.y+self.window < 0): continue
				if (self.height <= neighbour_seed.y+self.window): continue

				if (neighbour_seed.x1+self.window < 0): continue
				if (self.width <= neighbour_seed.x1+self.window): continue

				if (neighbour_seed.x2+self.window < 0): continue
				if (self.width <= neighbour_seed.x2+self.window): continue

				neighbour_similarity = self.getSimilarity(neighbour_seed)

				if (best_similarity < neighbour_similarity):
					best_similarity = neighbour_similarity
					best_seed = neighbour_seed

				if best_seed is None:
					continue

				if numpy.isfinite(best_similarity) == False:
					continue

				best_neighbours.append(best_seed)

		return frozenset(best_neighbours)

	def match(self, image1, image2, show = True):

		# Initialize data

		self.image1 = image1
		self.image2 = image2

		self.seeds = PriorityQueue()
		
		self.clearCorrespondenceTable()
		self.clearSimilarityTable()

		self.D1.fill(0-(1+self.disparity))
		self.D2.fill(0-(1+self.disparity))
		self.C1.fill(0-numpy.inf)
		self.C2.fill(0-numpy.inf)

		if self.initialSeeds is None:
			self.preemptiveMatching()
		else:
			for seed in self.initialSeeds:
				self.pushSeed(seed)

		# Setup visuals
		
		if show:
			#figure, axis = pyplot.subplots()
			figure, (axis1, axis2) = pyplot.subplots(1, 2)
			pyplot.plot()
			pyplot.hold(True)

		# Run the algorithm

		print("GCS Algorithm...")

		while True:

			sys.stdout.write('\r')
			sys.stdout.write("[S Size]: %d" % self.seeds.qsize())
			sys.stdout.write('\t')
			sys.stdout.write("[Tau Size]: %d" % self.correspondenceCount)
			sys.stdout.flush()

			ss = self.popSeed()
			if ss is None: break

			for qq in self.getBestNeighbours(ss):
				
				cc = self.getSimilarity(qq)
				dd = qq.getDisparity()

				if (cc < self.tau): continue
				if self.containsCorrespondence(qq): continue
				if (cc+self.mu < numpy.min([ self.C1[qq.y][qq.x1], self.C2[qq.y][qq.x2] ])): continue

				self.insertCorrespondence(qq)
				self.pushSeed(qq)

				if (self.C1[qq.y][qq.x1] < cc):
					self.C1[qq.y][qq.x1] = cc
					self.D1[qq.y][qq.x1] = dd

				if (self.C2[qq.y][qq.x2] < cc):
					self.C2[qq.y][qq.x2] = cc
					self.D2[qq.y][qq.x2] = dd

				if show and (self.correspondenceCount % 1000) == 0:

					axis1.set_title("Left Disparity Map")
					axis2.set_title("Right Disparity Map")
					
					axis1.imshow(self.D1)
					axis2.imshow(self.D2)

					pyplot.show(block = False)
					
					figure.canvas.draw()
					axis1.cla()
					axis2.cla()

		sys.stdout.write('\n')
		
		# Return the disparity maps

		return (self.D1, self.D2, self.C1, self.C2)

##### ##### ##### ##### ##### 
##### RUN PROGRAM
##### ##### ##### ##### ##### 

if (__name__ == "__main__"):

	import image

	rectified1 = image.read(sys.argv[1])
	rectified2 = image.read(sys.argv[2])
	disparity = int(sys.argv[3])
	window = int(sys.argv[4])

	# Convert to greyscale
	
	rectified1 = image.as_greyscale(rectified1)
	rectified2 = image.as_greyscale(rectified2)

	# Crop excess height

	height	= numpy.min([rectified1.shape[0], rectified2.shape[0]])
	width	= numpy.min([rectified1.shape[1], rectified2.shape[1]])
	
	rectified1 = rectified1[:height, :width]
	rectified2 = rectified2[:height, :width]

	# Scale down

	if (MAX_WIDTH < width):
		ratio = float(MAX_WIDTH)/width
		rectified1 = arrays.resample(rectified1, [ int(ratio*x) for x in rectified1.shape ])
		rectified2 = arrays.resample(rectified2, [ int(ratio*x) for x in rectified2.shape ])

	# Load mask1 and mask2, set image1 to the image with the largest corresponding mask

	mask1 = numpy.load("mask1.npy")
	mask2 = numpy.load("mask2.npy")

	reverse = (numpy.sum(mask1) < numpy.sum(mask2))
	
	image1 = rectified1
	image2 = rectified2

	if reverse:
		image1 = rectified2
		image2 = rectified1
	
	# Run GCS

	shape = image1.shape
	gcs = GrowingCorrespondenceSeeds(shape, disparity, window)
	
	if reverse:
		(disparity2, disparity1, similarity2, similarity1) = gcs.match(image1, image2)
	else:
		(disparity1, disparity2, similarity1, similarity2) = gcs.match(image1, image2)

	image.write("gcs1.png", disparity1)
	image.write("gcs2.png", disparity2)
