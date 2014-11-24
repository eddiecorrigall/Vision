import sys
import numpy

from Queue import PriorityQueue

sys.path.append("..")

import image
import costs

MAX_WIDTH = 512

WINDOW = 3
DISPARITY = 16#32#64
DISPARITY_SPAN = 2*DISPARITY+1

ROLL = False
TAU = 0.6
MU = 0.05

##### ##### ##### ##### ##### 
##### Tau (Seed Correspondence)
##### ##### ##### ##### ##### 

def Tau_disparity_index(seed):
	# Map get_disparity(seed) to [0, DISPARITY_SPAN)
	return DISPARITY + get_disparity(seed) - 1

def Tau_initialize(shape): # Space complexity O(H*W*(2*D+1))
	
	# IMPORTANT:
	# Tau should normally be implimented by a 2D array of ArrayLists
	# However, since python list insert is O(n), then binary search is also O(n)
	# A sparse matrix is very costly memory-wise, but very efficient in runtime
	
	# NOTE:
	# Disparity can be anywhere from [-DISPARITY, DISPARITY]
	
	(height, width) = shape
	return numpy.zeros([height, width, DISPARITY_SPAN], dtype = bool)

def Tau_contains(Tau, seed): # Runtime complexity is O(1)
	(yy, xx, xx_prime) = seed
	return Tau[yy][xx][Tau_disparity_index(seed)]

def Tau_insert(Tau, seed): # Runtime complexity is O(1)
	(yy, xx, xx_prime) = seed
	Tau[yy][xx][Tau_disparity_index(seed)] = True

##### ##### ##### ##### ##### 

def get_disparity(seed):
	(yy, xx, xx_prime) = seed
	return (xx_prime - xx)

def get_similarity(auxilary, L, R, seed):

	(yy, xx, xx_prime) = seed

	tt = Tau_disparity_index(seed)

	if (numpy.isfinite(auxilary[yy][xx][tt])):
		return auxilary[yy][xx][tt]

	L_window = image.neighbours(L, yy, xx, WINDOW, roll = ROLL)
	R_window = image.neighbours(R, yy, xx_prime, WINDOW, roll = ROLL)

	cc = costs.ncc(L_window, R_window)

	auxilary[yy][xx][tt] = cc

	return cc

def get_best_neighbours(auxilary, L, R, seed):

	(height, width) = L.shape

	# ...
	
	(yy, xx, xx_prime) = seed

	neighbourhoods = [
		[
			(yy+0, xx-1, xx_prime-1),
			(yy+0, xx-2, xx_prime-1),
			(yy+0, xx-1, xx_prime-2),
		], [
			(yy+0, xx+1, xx_prime+1),
			(yy+0, xx+2, xx_prime+1),
			(yy+0, xx+1, xx_prime+2)
		], [
			(yy-1, xx+0, xx_prime+0),
			(yy-1, xx-1, xx_prime+0),
			(yy-1, xx+1, xx_prime+0),
			(yy-1, xx+0, xx_prime-1),
			(yy-1, xx+0, xx_prime+1)
		], [
			(yy+1, xx+0, xx_prime+0),
			(yy+1, xx-1, xx_prime+0),
			(yy+1, xx+1, xx_prime+0),
			(yy+1, xx+0, xx_prime-1),
			(yy+1, xx+0, xx_prime+1)
		]
	]

	# ...

	best_neighbours = []

	for neighbourhood in neighbourhoods:

		best_seed = None
		best_similarity = -numpy.inf

		for neighbour_seed in neighbourhood:

			(yy, xx, xx_prime) = neighbour_seed

			if DISPARITY < numpy.abs(get_disparity(seed)):
				continue

			if (ROLL):

				if (yy < 0): continue
				if (xx < 0): continue
				if (xx_prime < 0): continue

				if (height <= yy): continue
				if (width <= xx): continue
				if (width <= xx_prime): continue

			else:

				if (yy < WINDOW): continue
				if (xx < WINDOW+DISPARITY): continue
				if (xx_prime < WINDOW+DISPARITY): continue

				if (height-WINDOW <= yy): continue
				if (width-WINDOW-DISPARITY <= xx): continue
				if (width-WINDOW-DISPARITY <= xx_prime): continue

			neighbour_similarity = get_similarity(auxilary, L, R, neighbour_seed)

			if (best_similarity < neighbour_similarity):
				best_similarity = neighbour_similarity
				best_seed = neighbour_seed

		if best_seed is None:
			continue

		if numpy.isfinite(best_similarity) == False:
			continue

		best_neighbours.append(best_seed)

	return frozenset(best_neighbours)

def preemptive_match(auxilary, L, R):

	(height, width) = L.shape

	print("GCS Preemptive Matching...")

	initial_seed_count = int(2.0*numpy.sqrt(height*width))
	yy_range = numpy.random.randint(low = WINDOW, high = height-WINDOW, size = initial_seed_count)
	xx_range = numpy.random.randint(low = WINDOW+DISPARITY, high = width-WINDOW-DISPARITY, size = initial_seed_count)
	dd_range = numpy.array(range(DISPARITY_SPAN))-DISPARITY

	Seeds = PriorityQueue()

	for ii in xrange(initial_seed_count):

		percent_complete = numpy.floor(100*(float(1+ii)/initial_seed_count))
		sys.stdout.write("\r\tProgress: %d%%" % percent_complete)
		sys.stdout.flush()

		yy = yy_range[ii]
		xx = xx_range[ii]

		cc_range = [ get_similarity(auxilary, L, R, (yy, xx, xx+disparity)) for disparity in dd_range ]

		best_index = numpy.argmax(cc_range)
		best_similarity	= cc_range[best_index]
		best_disparity	= dd_range[best_index]

		Seeds.put( (0-best_similarity, (yy, xx, xx+best_disparity)) )

	sys.stdout.write("\r\n")

	return Seeds

def gcs_matching(L, R, Seeds = None, tau = TAU, mu = MU, show = True):

	(height, width) = shape = L.shape

	# ...

	auxilary = numpy.nan*numpy.ones([height, width, DISPARITY_SPAN])

	Tau	= Tau_initialize(shape)
	Tau_count = 0

	# Best costs
	CC			= 0-numpy.inf*numpy.ones(shape)
	CC_prime	= 0-numpy.inf*numpy.ones(shape)

	# Best disparities
	DD			= 0-(1+DISPARITY)*numpy.ones(shape)
	DD_prime	= 0-(1+DISPARITY)*numpy.ones(shape)

	# Handle preemptive matching
	if Seeds is None:
		Seeds = preemptive_match(auxilary, L, R)

	# Setup visuals
	if show:
		import matplotlib.pyplot as pyplot
		figure, axis = pyplot.subplots()
		pyplot.plot()
		pyplot.hold(True)
	
	print("GCS Algorithm...")

	while True:

		sys.stdout.write('\r')
		sys.stdout.write("[S Size]: %d" % Seeds.qsize())
		sys.stdout.write('\t')
		sys.stdout.write("[Tau Size]: %d" % Tau_count)
		sys.stdout.flush()
		
		if Seeds.empty(): break

		(_, ss) = Seeds.get()

		for qq in get_best_neighbours(auxilary, L, R, ss):

			(vv, uu, uu_prime) = qq

			cc = get_similarity(auxilary, L, R, qq)

			if (cc < tau): continue
			if Tau_contains(Tau, qq): continue
			if (cc+mu < numpy.min([ CC[vv][uu], CC_prime[vv][uu_prime] ])): continue

			Tau_insert(Tau, qq)
			Tau_count += 1

			Seeds.put( (0-cc, qq) )
			
			if (CC[vv][uu] < cc):
				CC[vv][uu] = cc
				DD[vv][uu] = get_disparity(qq)

			if (CC_prime[vv][uu_prime] < cc):
				CC_prime[vv][uu_prime] = cc
				DD_prime[vv][uu_prime] = get_disparity(qq)

			if show and (Tau_count % 1000) == 0:
				image.show(DD, block = False)
				figure.canvas.draw()
				axis.cla()
	
	sys.stdout.write('\n')

	return (DD, DD_prime, CC, CC_prime)

##### ##### ##### ##### ##### 
##### RUN PROGRAM
##### ##### ##### ##### ##### 

if __name__ == "__main__":

	import pickle
	from arrays import *

	L_disparity_map_path = "L_dmap"
	R_disparity_map_path = "R_dmap"

	if (sys.argv[1] == "show"):

		L_disparity_map = pickle.load(open(L_disparity_map_path, "rb"))
		R_disparity_map = pickle.load(open(R_disparity_map_path, "rb"))

		(height, width) = shape = L_disparity_map.shape
	
	else:
		
		if len(sys.argv) <= 2:
			print("Requires left and right image path as input!")
			exit()

		L_path = sys.argv[1]
		R_path = sys.argv[2]

		L = image.read(L_path)
		R = image.read(R_path)

		L = image.as_greyscale(L)
		R = image.as_greyscale(R)

		# Crop excess height

		height	= numpy.min([L.shape[0], R.shape[0]])
		width	= numpy.min([L.shape[1], R.shape[1]])

		L = L[:height, :width]
		R = R[:height, :width]

		# Scale by height

		if (MAX_WIDTH < width):
			aspect = float(MAX_WIDTH)/width
			L = resample(L, [ int(aspect*x) for x in L.shape ])
			R = resample(R, [ int(aspect*x) for x in R.shape ])

		print("Left shape: " + str(L.shape))
		print("Right shape: " + str(R.shape))

		(L_disparity_map, R_disparity_map, L_cost_map, R_cost_map) = gcs_matching(L, R)

		pickle.dump(L_disparity_map, open(L_disparity_map_path, "wb"))
		pickle.dump(R_disparity_map, open(R_disparity_map_path, "wb"))

	image.show(L_disparity_map, title = "Left Dispartiy Map")
	image.show(R_disparity_map, title = "Right Dispartiy Map")
