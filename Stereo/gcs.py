import sys
import numpy
from Queue import PriorityQueue

sys.path.append("..")

import costs
import image

def compute_similarity(L, R, seed, window = 3):

	(u, u_prime, v) = seed

	L_window = image.neighbours(L, v, u, window)
	if L_window is None: return (0-numpy.inf)

	R_window = image.neighbours(R, v, u_prime, window)
	if R_window is None: return (0-numpy.inf)

	return costs.ncc(L_window, R_window)

def get_similarity(L, R, seed, cache = False):

	# cache is seed => similarity
	# otherwise False

	if isinstance(cache, dict):
		if seed not in cache.keys():
			similarity = cache[seed] = compute_similarity(L, R, seed)
			return similarity
		else:
			return cache[seed]

	return compute_similarity(L, R, seed)

def get_x_inhibition_zone(shape, seed):

	# Given a seed, an element of the matching table Tau
	# Obtained from the unconstrained growth procedure.

	(u, u_prime, v) = seed

	# Then the union of...
	#	(:, u_prime, v) = (a, u_prime, v) where a is not u, AND
	#	(u, :, v) = (u, b, v) where b is not u_prime
	# ... is called the x-inhibition zone of s.

	zone = []
	zone.extend([(u, b, v)			for b in xrange(width) if (b is not u_prime)])
	zone.extend([(a, u_prime, v)	for a in xrange(width) if (a is not u)])

	return frozenset(zone)

def get_best_neighbours(L, R, seed, cache = False):

	(x, x_prime, y) = seed

	neighbourhoods = [
		[
			(x-1, x_prime-1, y+0),
			(x-2, x_prime-1, y+0),
			(x-1, x_prime-2, y+0),
		], [
			(x+1, x_prime+1, y+0),
			(x+2, x_prime+1, y+0),
			(x+1, x_prime+2, y+0)
		], [
			(x+0, x_prime+0, y-1),
			(x-1, x_prime+0, y-1),
			(x+1, x_prime+0, y-1),
			(x+0, x_prime-1, y-1),
			(x+0, x_prime+1, y-1)
		], [
			(x+0, x_prime+0, y+1),
			(x-1, x_prime+0, y+1),
			(x+1, x_prime+0, y+1),
			(x+0, x_prime-1, y+1),
			(x+0, x_prime+1, y+1)
		]
	]

	best_neighbours = []

	for neighbourhood in neighbourhoods:

		best_seed = neighbourhood[0]
		best_similarity = get_similarity(L, R, neighbourhood[0], cache)

		for nn in neighbourhood[1:]:
			cc = get_similarity(L, R, nn, cache)
			if (cc < best_similarity):
				best_similarity = cc
				best_seed = nn

		if numpy.isfinite(best_similarity) == False:
			continue

		best_neighbours.append(best_seed)

	return frozenset(best_neighbours)

def gcs_matching(L, R, Seeds):

	# L, R are images of the same size
	# Seeds is a min priority queue containing pairs (similarity, x, x_prime, y)

	shape = (height, width) = L.shape

	cache = dict() # Auxilary cache

	tau = 0.9
	Tau = set()

	while not Seeds.empty():

		(similarity, x, x_prime, y) = Seeds.get()
		ss = (x, x_prime, y)

		if (numpy.isfinite(similarity) == False): break

		print( len(Tau), len(cache), Seeds.qsize() )

		if (width*height) < len(Tau):
			break

		for qq in get_best_neighbours(L, R, ss, cache):

			cc = get_similarity(L, R, qq, cache)

			if (cc < tau): continue
			if (0 in [0 if (zz in Tau) else 1 for zz in get_x_inhibition_zone(shape, qq)]): continue

			# This is a good match
			Tau.add(qq)

			# Pair seed with with negated similarity, and place it in the min-priority queue
			pair = [cc]
			pair.extend(qq)
			Seeds.put(pair)
	# ...

	return Tau

if __name__ == "__main__":

	L_path = sys.argv[1]
	R_path = sys.argv[2]

	L = image.greyscale(image.read(L_path))
	R = image.greyscale(image.read(R_path))
	(height, width) = shape = L.shape

	Seeds = PriorityQueue(width*width*height)

	# Preemptive matching...

	import matching
	import pickle
	import os

	disparity_path = L_path + ".disparity"

	D = None

	print("Load disparity...\n")
	if os.path.exists(disparity_path):
		D = pickle.load(open(disparity_path, "rb"))
	else:
		D = matching.block_matching(L, R,)
		pickle.dump(D, open(disparity_path, "wb"))

	for yy in xrange(height):
		for xx in xrange(width):
			dd = int(D[yy][xx])
			ss = (xx, xx+dd, yy)
			cc = get_similarity(L, R, ss)
			if (cc < 0.9): continue
			print( (cc, ss) )
			pair = [cc]
			pair.extend(ss)
			Seeds.put(pair)

	# ...
	
	print("Do work...")
	Tau = gcs_matching(L, R, Seeds)

	print("Convert to depth map...")
	D = numpy.zeros(shape)
	for tt in Tau:
		(x, x_prime, y) = tt
		if (x < 0): continue
		if (x_prime < 0): continue
		if (y < 0): continue
		D[y][x] = numpy.abs(x - x_prime)

	image.show(D)
