import sys

sys.path.append("..")

import numpy

from Queue import PriorityQueue

import image
import costs

''' # SYNTH
SCALE = 1.0
DISPARITY = 12
WINDOW = 2
TAU = 0.6
MU = 0.05
'''

''' # PM
SCALE = 1.0
DISPARITY = 8
WINDOW = 2
TAU = 0.6
MU = 0.05
'''

#''' # STAR
SCALE = 1.0
DISPARITY = 8
WINDOW = 2
TAU = 0.8
MU = 0.1
#'''

''' # MARTIN
SCALE = 0.5
DISPARITY = 32
WINDOW = 2
TAU = 0.85
MU = 0.05
'''

DISPARITY_SPAN = 2*DISPARITY+1
WINDOW_SPAN = 2*WINDOW+1

def get_disparity(seed):
	(yy, xx, xx_prime) = seed
	return (xx_prime - xx)

def get_similarity(L, R, seed, auxilary):

	(yy, xx, xx_prime) = seed

	dd = DISPARITY + get_disparity(seed) - 1

	if (numpy.isfinite(auxilary[yy][xx][dd])):
		return auxilary[yy][xx][dd]

	L_window = image.neighbours(L, yy, xx, WINDOW)
	R_window = image.neighbours(R, yy, xx_prime, WINDOW)

	cc = costs.ncc(L_window, R_window)

	auxilary[yy][xx][dd] = cc

	return cc

def get_best_neighbours(L, R, seed, auxilary):

	(height, width) = L.shape
	(yy, xx, xx_prime) = seed

	if DISPARITY < numpy.abs(get_disparity(seed)):
		return set()

	H = WINDOW+1
	W = DISPARITY+2

	if (H < yy) == False:		return set()
	if (W < xx) == False:		return set()
	if (W < xx_prime) == False:	return set()

	if (yy			< height-H) == False:	return set()
	if (xx			< width-W) == False: 	return set()
	if (xx_prime	< width-W) == False: 	return set()

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

	best_neighbours = []

	for neighbourhood in neighbourhoods:

		best_seed = neighbourhood[0]
		best_similarity = get_similarity(L, R, neighbourhood[0], auxilary)

		for nn in neighbourhood[1:]:
			cc = get_similarity(L, R, nn, auxilary)
			if (best_similarity < cc):
				best_similarity = cc
				best_seed = nn

		if numpy.isfinite(best_similarity) == False:
			continue

		best_neighbours.append(best_seed)

	return frozenset(best_neighbours)

def Tau_initialize(shape): # Space complexity O(2*H*W^2)
	
	# IMPORTANT:
	# Tau should normally be implimented by a 2D array of ArrayLists
	# However, since python list insert is O(n), then binary search is also O(n)
	# A sparse matrix is very costly memory-wise, but very efficient in runtime
	
	# NOTE:
	# Disparity can be anywhere from (-width, width)
	
	(height, width) = shape
	return numpy.zeros([height, width, DISPARITY_SPAN], dtype = bool)

def Tau_contains(Tau, seed): # Runtime complexity is O(1)

	(height, width, _) = Tau.shape
	(yy, xx, xx_prime) = seed

	dd = DISPARITY + get_disparity(seed) - 1

	return Tau[yy][xx][dd]

def Tau_insert(Tau, seed): # Runtime complexity is O(1)

	(height, width, _) = Tau.shape
	(yy, xx, xx_prime) = seed

	dd = DISPARITY + get_disparity(seed) - 1

	Tau[yy][xx][dd] = True

def gcs_matching(L, R, Seeds, auxilary, tau = TAU, mu = MU, show = True):

	# little tau is first pass screening, ie. "We require a 90% positive match"
	# mu is an exploration threshold, ie. "We will explore if there is significant improvement"

	(height, width) = shape = L.shape

	# ...

	Tau	= Tau_initialize(shape)
	Tau_count = 0

	# Best costs
	CC			= 0-numpy.inf*numpy.ones(shape)
	CC_prime	= 0-numpy.inf*numpy.ones(shape)

	# Best disparities
	DD			= 0-(1+DISPARITY)*numpy.ones(shape)
	DD_prime	= 0-(1+DISPARITY)*numpy.ones(shape)

	if show:
		import matplotlib.pyplot as pyplot
		figure, axis = pyplot.subplots()
		pyplot.plot()
		pyplot.hold(True)

	while not Seeds.empty():

		(_, ss) = Seeds.get()

		for qq in get_best_neighbours(L, R, ss, auxilary):

			(vv, uu, uu_prime) = qq

			cc = get_similarity(L, R, qq, auxilary)

			#T = 0.9 - float(Tau_count) / (height*width)
			
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

			if show and (Tau_count % 500) == 0:
				image.show(DD, block = False)
				figure.canvas.draw()
				axis.cla()

	return (DD, DD_prime, CC, CC_prime)

if (__name__ == "__main__"):

	import os
	import pickle

	#numpy.random.seed(382)

	skip = False

	if (sys.argv[1] == "skip"):
		skip = True

	##### Growing Correspondence Seeds

	if skip == False:

		L_path = sys.argv[1]
		R_path = sys.argv[2]

		L_image = image.greyscale(image.read(L_path, scale = SCALE))
		R_image = image.greyscale(image.read(R_path, scale = SCALE))

		if (L_image.shape[0] != R_image.shape[0]):
			print("Left and Right image shapes differ!")
			exit()

		(height, width) = shape = L_image.shape
		
		# Preemptive Matching... Randomize

		dd_range = numpy.array(range(DISPARITY_SPAN))-DISPARITY

		auxilary = numpy.nan*numpy.ones([height, width, DISPARITY_SPAN])

		Seeds_count = int(2*numpy.sqrt(width*height))
		Seeds = PriorityQueue(height*width)

		yy_range = numpy.random.randint(low = WINDOW, high = height-WINDOW, size = Seeds_count)
		xx_range = numpy.random.randint(low = WINDOW+DISPARITY, high = width-WINDOW-DISPARITY, size = Seeds_count)

		for ii in xrange(Seeds_count):

			#numpy.random.shuffle(dd_range) # Unnecessary
			cc = [ get_similarity(L_image, R_image, (yy_range[ii], xx_range[ii], xx_range[ii]+dd), auxilary) for dd in dd_range ]
			kk = numpy.argmax(cc)
			Seeds.put( (0-cc[kk], (yy_range[ii], xx_range[ii], xx_range[ii]+dd_range[kk]) ) )
			print(Seeds.qsize(), Seeds_count)

		# ...

		(DD, DD_prime, CC, CC_prime) = gcs_matching(L_image, R_image, Seeds, auxilary)

		pickle.dump(DD, open("DD", "wb"))
		pickle.dump(DD_prime, open("DD_prime", "wb"))
		
		pickle.dump(CC, open("CC", "wb"))
		pickle.dump(CC_prime, open("CC_prime", "wb"))

		exit()

	else:

		DD = pickle.load(open("DD", "rb"))
		DD_prime = pickle.load(open("DD_prime", "rb"))

		CC = pickle.load(open("CC", "rb"))
		CC_prime = pickle.load(open("CC_prime", "rb"))

		(height, width) = shape = DD.shape

	image.show(DD)
	image.show(DD_prime)
