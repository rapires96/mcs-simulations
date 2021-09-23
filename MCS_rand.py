#MCS random algorithm for random selection process and greedy algorithm
#exec: python MCS_rand.py --samples 1000 --reward 100 --l 50 --time 1
import numpy as np
import pandas as pd
import os
import sys
import math
from matplotlib import pyplot as plt
import seaborn as sns
import random
import argparse

MIN_SAMPLES = 20
REWARD = 100
LAMBDA = 10
STEP = 5
TIME = 1
PATH = None
#
# Data
SStrategy = {
	'A': [4, [0.90252827, 0.91066735, 0.91092855, 0.91666625], 0.623, [1.5986, 1.5049, 1.5421, 1.5049], 0.628, 'summer center 20fps'],
    'B': [8, [0.59143803, 0.63986266, 0.64408618, 0.67376032], 0.581, [9.8368, 6.063, 5.8548, 4.6787], 0.681, 'summer center & right 20fps'],
    'C': [2, [0.7633532, 0.71191908, 0.62435817, 0.58279836], 0.592, [1.8967, 2.4147, 3.9092, 5.1964], 1.477, 'winter center 10fps'],
    'D': [2, [0.55484865, 0.49488992, 0.46383198, 0.47564543], 0.518, [3.4654, 5.1389, 6.6177, 5.9783], 2.736, 'winter right 10fps'],
    'E': [2, [0.51529945, 0.46001078, 0.38968902, 0.38225037], 0.530, [2.3987, 3.2028, 5.0119, 5.2987], None, 'winter left 10fps'],
    'F': [1, [0.28474328, 0.25622082, 0.23781912, 0.25209962], 0.258, [10.582, 16.452, 24.931, 17.831], 3.732, 'winter center 5fps'],
    'G': [4, [0.21906206, 0.21906206, 0.21906206, 0.21906206], 0.513, [163.746, 108.633, 88.898, 49.317], None, 'winter center & right 10fps'],
    'H': [6, [0,0,0,0], 0, [None, None, None, None], None, 'winter 3views 10fps']
}

def get_arguments():
	"""Argument parser"""
	parser = argparse.ArgumentParser(description = 'MCS algorithm')
	parser.add_argument('-n','--samples', type=int, default=MIN_SAMPLES)
	parser.add_argument('-r','--reward', type=int, default=REWARD) 
	parser.add_argument('-l','--l', type=int, default=LAMBDA)
	parser.add_argument('-t','--time', type=float, default=TIME)
	parser.add_argument('-p','--path', type=str, default=PATH)

	return parser.parse_args()

args = get_arguments()

def run_sim(SStrategy, nsamples, R, df_sim):
	i = 0
	total_cost = 0
	#Initialize
	n = random.randint(0, len(SStrategy)-1)
	s, cost, density, tsign, maha, omega, i = draw_sample(SStrategy, n)
	total_cost = cost
	###
	while len(df_sim) < nsamples and total_cost < R:
		#Check values and contributions
		if maha == None or maha > 10 or density < 0.25:
			df_sim.loc[len(df_sim)] = [s, cost, 0 , i]
		else:
			df_sim.loc[len(df_sim)] =  [s, cost, omega, i]
		#Draw new sample
		n = random.randint(0, len(SStrategy)-1)
		s, cost, density, tsign, maha, omega, i = draw_sample(SStrategy, n)
		total_cost += cost
	return df_sim

def draw_sample(SStrategy, n):

	strategy = list(SStrategy.keys())
	s = strategy[n]
	cost, density, tsign = SStrategy[s][0], SStrategy[s][2], SStrategy[s][4]
	i = random.randint(0,3)
	maha, omega =  SStrategy[s][3][i], SStrategy[s][1][i]
	
	return s, cost, density, tsign, maha, omega, i

def utilities(df_sim, l, t, R):
	"""Compute u0, ui"""
	omega = list(df_sim['omega'].values)
	U = [math.log(1 + o * t) for o in omega]
	uo = l * math.log(1 + np.sum(U)) - R
	costs = list(df_sim['cost'].values)
	ui = []
	for ci in costs:
		first_arg = ((ci*t)/(np.sum(costs)*t)) * R
		second_arg = ci*t
		ui.append(first_arg - second_arg)
	df_sim = df_sim.assign(ui = ui)

	return df_sim, uo

def qoi_eval(df_sim):
	"""Get values from sim dataframe"""
	strategy = list(df_sim.strategy.values)
	index = list(df_sim.idx.values)

	ts_dist = [] # distance error
	tsign = [] # false negatives
	maha = [] # mahalanobis distances
	density = [] #densities

	#collect QoI params from samples, in FUTURE might want to use GPS loc for graphical display in presentation
	for i in range(len(strategy)):
		s = strategy[i]
		j = index[i]
		density.append(SStrategy[s][2])
		#print(f'bueno{index}: {SStrategy[s][3]}')
		maha.append(SStrategy[s][3][j])
		if SStrategy[s][4] != None:
			tsign.append(True)
			ts_dist.append(SStrategy[s][4])
		else:
			tsign.append(False)
	#obtain statistics from samples
	positives, negatives, total = tsign.count(True), tsign.count(False), len(tsign)
	density = max(density)
	maha = [val for val in maha if val != None]
	mahalanobis = min(maha)
	ts_dist = np.mean(ts_dist)
	return density, mahalanobis, ts_dist, positives, negatives

def write_sim(file,uo,ui,cost,samples,density,maha,tserr,tspos,tsneg,counts):
	"""write results in file"""
	with open(file, 'a') as f:
		line = f'{uo},{ui},{cost},{samples},{density},{maha},{tserr},{tspos},{tsneg},{counts}\n'
		f.write(line)

def main():
	# Get arguments
	nsamples = args.samples
	R = args.reward
	l = args.l
	time = args.time
	rfile = args.path

	# Dataframe of simulation
	df_sim = pd.DataFrame(columns = ['strategy','cost','omega', 'idx'])
	
	# Recruit randomly users until budget is finished
	df_sim = run_sim(SStrategy, nsamples, R, df_sim)
	sim_cost = np.sum(df_sim['cost'].values)
	remain = R - sim_cost
	
	
	# Compute utilities
	df_sim, uo = utilities(df_sim, l, time, R)
	
	values, counts = np.unique(df_sim.strategy.values, return_counts = True)
	Scount = {values[i]: counts[i] for i in range(len(values))}
	
	#print(f"column cost: \n{np.sum(df_sim['cost'].values)}")
	#print(f'Samples ui:\n{df_sim}')

	# Evaluate the global QoI of samples, Min mahalanobis distance, Best density of model, avg localization err, ts pos/ps
	density, mahalanobis, ts_dist, positives, negatives = qoi_eval(df_sim)
	
	print(f'Budget: {R}; Simulation cost: {sim_cost}; Remaining = {R - sim_cost}')
	print(f'U0 = {uo:.2f}')
	print(f'number samples = {len(df_sim)}, mean ui = {df_sim.ui.mean():.4f}')
	print(f'U0 = {uo:.2f}')
	print(f'Density: {density}')
	print(f'Mahalanobis: {mahalanobis}')
	print(f'TSign: pos: {positives}, neg: {negatives}, distance err (m): {ts_dist:.5f}')
	print(f'Counts {Scount}')
	print(f'Dataframe {df_sim}')

	if args.path != None:
		#file,uo,ui,cost,samples,density,maha,tserr,tspos,tsneg,counts
		write_sim(args.path, uo, df_sim.ui.mean(), sim_cost, len(df_sim), density, mahalanobis, ts_dist, positives, negatives, Scount)


if __name__ == '__main__':
	main()