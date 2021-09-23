#MCS algorithm with Discrete R.V. distribution,  
import numpy as np
import pandas as pd
import os
import sys
import math
from matplotlib import pyplot as plt
import seaborn as sns
import random
from huffmanTree import HuffmanTree
import argparse

#Default values
SAMPLES = 10000
REWARD = 100
LAMBDA = 50
STEP = 5
TOTALR = 100
STAGER = 25
TIME = 1
#Determines the number of rounds

#Data
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
	parser.add_argument('-n','--samples', type=int, default=SAMPLES)
	parser.add_argument('-r','--reward', type=int, default=REWARD)
	parser.add_argument('-l','--l', type=int, default=LAMBDA)
	parser.add_argument('-s', '--stage', type=int, default=STAGER)
	parser.add_argument('-t', '--time', type=int, default=TIME)
	parser.add_argument('-p', '--path', type=str)
	return parser.parse_args()

args = get_arguments()

####SIMULATION METHODS####
def draw_sample(s):
	cost, density, tsign = SStrategy[s][0], SStrategy[s][2], SStrategy[s][4]
	i = random.randint(0,3)
	maha, omega =  SStrategy[s][3][i], SStrategy[s][1][i]
	return cost, density, tsign, maha, omega, i

def add_to_df(df_sim, rm_list, s, cost, density, tsign, maha, omega, i, stage):
	if maha == None or maha > 10 or density < 0.25:
		df_sim.loc[len(df_sim)] = [s, cost, 0 , stage, i]
		rm_list.append(s)
	else:
		df_sim.loc[len(df_sim)] =  [s, cost, omega, stage, i]
	return df_sim, rm_list

#def update_weights(df_sim, stage, strats, weights):
def update_weights(df_sim, stage, strats, sweights): #update the weights 
	#sweights = {s: weights[idx] for idx, s in enumerate(strats)}
	df_stage = df_sim[df_sim["stage"]==stage]
	omegas = []

	strats = list(np.unique(strats))

	for idx, s in enumerate(strats):
		if s in sweights.keys():
			dfs = df_stage.where(df_stage["strategy"] == s)
			mval = dfs["omega"].max()

			# Give more value to those contributions 
			if mval > 0.8: mval = mval * 3
			elif mval > 0.7: mval = mval * 2
			else: mval = mval * 1

			w = sweights[s]
			sweights[s] = mval + w
			#print(f'strat:{s}, max omega:{mval}, previous weight: {w}, total {mval + w}')

	return list(sweights.keys()), list(sweights.values())

def get_stagecost(df_sim, stage):

	if len(df_sim) != 0:
		df_stage = df_sim[df_sim["stage"]==stage]
		stageCost = df_stage["cost"].sum()
	else:
		stageCost = 0
	return stageCost

		#<omegas> for the strat s in <stage>
def cstatement(strats, df_sim):
	"""Control Statement to ensure all strategies are in the simulation"""
	#strats = list(SStrategy.keys())
	strats = list(np.unique(strats))

	for s in strats: 
		if s not in np.array(df_sim.strategy):
			return False
			break
	return True

def run_sim(df_sim, totalR, stageR):
	#SIMULATION PARAMS
	nstages = totalR/stageR
	#Initialize probability distribution EVEN HF TREE
	strats = list(SStrategy.keys())
	weights = [1 for i in range(len(strats))]
	sweights = {s: weights[idx] for idx, s in enumerate(strats)}
	#Create Huffman Tree
	hf = HuffmanTree(strats, weights)
	tree, root = hf.generate_tree()
	#Draw sample from the tree
	s = HuffmanTree.draw_strat(tree, root)
	#sweights = {s: weights[idx] for idx, s in enumerate(strats)}
	stage = 0
	rm_list = [] #store removed strats

	for i in range(int(nstages)):
		#initialize
		stageCost = 0
		stageCost = get_stagecost(df_sim, stage)
		s = HuffmanTree.draw_strat(tree, root)
		cost, density, tsign, maha, omega, i = draw_sample(s)
		stageStrats = []
		
		#print(f'STAGE {stage}')
		while stageCost < stageR:

			stageCost = get_stagecost(df_sim, stage)
			if stage > 0 and cstatement(list(SStrategy.keys()), df_sim): #once all strategies are in the df
				df_sim, rm_list = add_to_df(df_sim, rm_list, s, cost, density, tsign, maha, omega, i, stage)
				stageStrats.append(s)
			else: #during initial stages we wish for diversity of strategies
				if s not in stageStrats or len(stageStrats) == len(SStrategy): #check it hasn't been used bf
					if int(cost + stageCost) >= stageR: break
					stageStrats.append(s)
					df_sim, rm_list = add_to_df(df_sim, rm_list, s, cost, density, tsign, maha, omega, i, stage)
				else: pass #if it has been used used bf

			s = HuffmanTree.draw_strat(tree, root)
			cost, density, tsign, maha, omega, i = draw_sample(s)
			stageCost = get_stagecost(df_sim, stage)
			if stageCost + cost >= stageR: break

		#stageCost = 0
		rm_list = list(np.unique(rm_list))
		for rm in rm_list:
			if rm in sweights.keys(): del sweights[rm]		
		#update weights
		strats, weights = update_weights(df_sim, stage, stageStrats, sweights)
		hf = HuffmanTree(strats, weights)
		tree, root = hf.generate_tree()
		stage += 1 #Next Stage
	#print(f'data frame:\n{df_sim}\nstrategy weights:\n{sweights}')
	return df_sim, sweights

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

def utilities(df_sim, totalR, stageR, l, t):

	nstages = int(totalR/stageR)
	#print(f'NSTAGES: {nstages}')
	R = totalR

	omega = list(df_sim.omega.values)
	U = [math.log(1 + o * t) for o in omega]
	uo = l * math.log(1 + np.sum(U)) - R

	ui = []
	for i in range(nstages):
		df_stage = df_sim[df_sim["stage"]==i]
		costs = list(df_stage.cost)
		stg_cost =  int(np.sum(df_stage.cost.values))
		for ci in costs:
			first_arg = ((ci*t)/stg_cost) * stageR
			second_arg = ci*t
			ui.append(first_arg - second_arg)

	df_sim = df_sim.assign(ui = ui)
	return df_sim, uo

def write_sim(file,uo,ui,cost,samples,density,maha,tserr,tspos,tsneg,counts,weights):
	"""write results in file"""
	with open(file, 'a') as f:
		line = f'{uo},{ui},{cost},{samples},{density},{maha},{tserr},{tspos},{tsneg},{counts},{weights}\n'
		f.write(line)
    
def main():
	""""""
	totalR = args.reward
	stageR = args.stage
	l = args.l
	t = args.time
	file = args.path

	df_sim = pd.DataFrame(columns = ['strategy','cost','omega','stage','idx'])
	df_sim, sweights = run_sim(df_sim, totalR, stageR)

	values, counts = np.unique(df_sim.strategy.values, return_counts = True)
	Scount = {values[i]: counts[i] for i in range(len(values))}
	sim_cost = np.sum(df_sim['cost'].values)

	df_sim, uo =  utilities(df_sim, totalR, stageR, l, t)
	#print(f'number samples = {len(df_sim)}, mean ui = {df_sim.ui.mean():.4f}')
	#print(f'U0 = {uo:.2f}')

	density, mahalanobis, ts_dist, positives, negatives = qoi_eval(df_sim)
	samples = len(df_sim)
	mean_ui = float(np.mean(df_sim.ui.values))

	#print(f'Budget: {totalR}; Simulation cost: {sim_cost}; Remaining = {totalR - sim_cost}')
	#print(f'Density: {density}')
	#print(f'Mahalanobis: {mahalanobis}')
	#print(f'TSign: pos: {positives}, neg: {negatives}, distance err(m): {ts_dist:.5f}')
	#print(f'Counts {Scount}')

	#print(f'data frame:\n{df_sim}') #\nstrategy weights:\n{sweights}')

	#print(f'strategy weights:\n{sweights}')
	print(mean_ui)
	if args.path != None:
		#file,uo,ui,cost,samples,density,maha,tserr,tspos,tsneg,counts
		write_sim(file,uo,mean_ui,sim_cost,samples,density, mahalanobis,ts_dist,positives,negatives,Scount,sweights)
		
if __name__ == '__main__':
	main()

"""
	i = 0
	selected = []
	strategy = list(SStrategy.keys())
	while len(selected) < STEP:
	    s = random.randint(0, len(SStrategy)-1)
	    if s not in selected:
	        selected.append(s)
	selected = [strategy[i] for i in selected]
	#print(selected)

def qoi_eval():
	
	delete = []
	for s in selected:
    cost, density, tsign = SStrategy[s][0], SStrategy[s][2], SStrategy[s][4]
    i = random.randint(0,3)
    maha, omega =  SStrategy[s][3][i], SStrategy[s][1][i]
    #simS[s] = [cost, omega]
    print(f'Strategy {s}: {density}, {maha}, {tsign}, cost:{cost}, omega:{omega}')
    #Threshold
    if maha == None or maha > 10 or density < 0.25:
        delete.append(s)
        if s not in simS: df_sim.loc[len(df_sim)] = [s, cost, 0]
        else: df_sim.loc[len(df_sim)] = [s, cost, 0]
    else:
        if s not in simS: df_sim.loc[len(df_sim)] =  [s, cost, omega]
        else: df_sim.loc[len(df_sim)] = [s, cost, omega]
"""