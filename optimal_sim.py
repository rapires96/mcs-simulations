"""does multiple simulations of NE_yang.py to obtain optimal values with multiple nusers and cost values"""
import numpy as np
import pandas as pd
import os
import sys
import math
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

NUSERS = [10, 20, 40, 60, 80]
COST = [1]
REWARD = [0, 100]
LAMBDA = 10
PATH = os.getenv('PATH')

def get_arguments():
	"""Argument parser"""
	parser = argparse.ArgumentParser(description = 'Yang Nash Equlibrium for Stackelberg Game')
	parser.add_argument('-n','--nusers', nargs = '+', type=int, default=NUSERS)
	parser.add_argument('-c','--cost', nargs = '+', type=int, default=COST)
	parser.add_argument('-r','--rangeR', nargs = '+', type=int, default=REWARD)
	parser.add_argument('-l','--l', type=int, default=LAMBDA)
	parser.add_argument('-p', '--path', type=str, default=PATH)
	return parser.parse_args()
args = get_arguments()

def read_results():
	#NUSERS, COST: R_opt, UO_opt, UI_opt, T_opt
	df = pd.DataFrame(columns=['NUSERS', 'COST', 'R_opt', 'u0_opt', 'ui_opt', 't_opt'])
	with open(args.path, 'r') as f:
		next(f)
		lines = f.readlines()

	for line in lines:
		row = list(line.split(','))
		[nusers, cost, R_opt, u0_opt, ui_opt, t_opt] = row
		df_length = len(df)
		df.loc[df_length] = [int(nusers), int(cost), float(R_opt), float(u0_opt), float(ui_opt), float(t_opt)]
		#df.append(row)

	return df

def plot_nuser(nuser, df):
	sns.set_style('whitegrid')
	ax = sns.lineplot(x="COST", y="u0_opt", dashes=False, ci=None, marker='o', data = df)
	ax.set_title('Impact of cost on Crowdsourcer profit')
	ax.set(xlabel='cost', ylabel='crowdsourcer profit')
	ax.legend(title = f'nuser: {nuser}')
	plt.show()

def plot_cost(cost, df):
	sns.set_style('whitegrid')
	ax = sns.lineplot(x='NUSERS', y='u0_opt', dashes=False, ci=None, marker='o', data = df)
	ax.set_title('Impact of number of users on Crowdsourcer profit')
	ax.set(xlabel='n users', ylabel='crowdsourcer profit')
	ax.legend(title = f'cost: {cost}')
	plt.show()

def sim_plots(nusers, cost):
	""""""
	print(nusers)
	print(cost)
	df = read_results()
	print(df)
	subdf = dict()
	#if the key is cost, nusers change. if the key is nusers, cost changes
	for i in nusers:
		subdf[f'nuser_{i}'] = df[df['NUSERS'] == i]
	for j in cost:
		subdf[f'cost_{j}'] = df[df['COST'] == j]

	#for value in subdf.values(): print(value)
	print(subdf.keys()) 

	# 'NUSERS', 'COST', 'R_opt', 'u0_opt', 'ui_opt', 't_opt'
	for key, value in subdf.items():
		if 'nuser' in key:
			nuser = key.split('_')[1]
			plot_nuser(nuser, value)
		if 'cost' in key:
			cost = key.split('_')[1]
			plot_cost(cost, value)
	
def run_sim(nusers, cost, rangeR, l):

	for i in nusers:
		for j in cost:
			command = f'python NE_yang.py --nusers {i} --cost {j} --rangeR {rangeR[0]} {rangeR[1]} --l {l}'
			os.system(command)

def main():
	""""""
	nusers = args.nusers
	cost = args.cost
	rangeR = args.rangeR
	l = args.l

	lusers = len(nusers)
	lcost = len(cost)

	#run_sim(nusers, cost, rangeR, l)
	sim_plots(nusers, cost)

if __name__ == '__main__':
	main()