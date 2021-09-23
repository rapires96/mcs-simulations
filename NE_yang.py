"""Computes optimal Reward, crowdsourcer utility and sensing time for omega = 1
method wresults.py is used for optimal_sim.py, not as stand-alone code"""
import numpy as np
import pandas as pd
import os
import sys
import math
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

NUSERS = 10
COST = 1
REWARD = [0,100]
LAMBDA = 10
PATH = os.getenv('PATH')

def get_arguments():
	"""Argument parser"""
	parser = argparse.ArgumentParser(description = 'Yang Nash Equlibrium for Stackelberg Game')
	parser.add_argument('-n','--nusers', type=int, default=NUSERS)
	parser.add_argument('-c','--cost', type=float, default=COST)
	parser.add_argument('-r','--rangeR',nargs = '+', type=int, default=REWARD)
	parser.add_argument('-l','--l', type=int, default=LAMBDA)
	parser.add_argument('-p', '--path', type=str, default=PATH)
	return parser.parse_args()

args = get_arguments()

def control_statement(key, users, selected):
	"""control statement value to select
	users in the group of winners"""
	numerator = users[key] + np.sum(list(selected.values()))
	denominator = len(selected)
	formula = numerator/denominator
	return formula

def generate_users(nusers, cost):
	"""generates n users with costs cost"""
	u = {}
	for i in range(nusers):
		key = '{0:0>3d}'.format(i)
		u[key] = cost
	#SORT ACCORDING TO UNIT COSTS OF USERS
	users = {k: v for k, v in sorted(u.items(), key=lambda item: item[1])}
	return users

def calculate_t(R, selected, key):
	
	n1 = (len(selected)-1)*R
	d1 = np.sum(list(selected.values()))
	first_arg = float(n1)/float(d1)

	n2 = (len(selected)-1)*selected[key]
	d2 = np.sum(list(selected.values()))
	second_arg = 1 - float(n2)/float(d2)

	return first_arg * (second_arg)
	#print('here')
def nash_equilibrium(users, R, nusers):
	#print(f'users: {users}')
	selected = {}
	item = iter(users.items())
	i = 0

	#ADD TWO FIRST (CHEAPEST) USERS TO THE SELECTED DICT
	for j in range(1):
		key, value = next(item)
		selected[key] = value
		i += 1

	#CHOOSE SELECTED USERS
	key, value =  next(item)
	#print(f'n users = {nusers}')
	while users[key] < control_statement(key, users, selected) and i < nusers:
		#print(f'{key}: {value} < {control_statement(key, users, selected)}')
		i += 1
		selected[key] = value
		if i != nusers:
			key, value = next(item)
		else:
			break
	i, j = 0, 0
	#print(len(selected))
	#CALCULATE TIMES FOR THE USERS
	times = {}
	for key, value in users.items():
		if key in selected.keys():
			times[key] = [value, calculate_t(R, selected, key)]
		else:
			times[key] = [value, 0]
	return times, selected
	#print(f'times: {times}')

def compute_utilities(times, R, l):
	#CALCULATE UTILITIES
	t = [value[1] for value in times.values()]
	l = 50
	U = [math.log(1+ti) for ti in t]
	uo = l * math.log(1 + np.sum(U)) - R
	
	utilities = {}
	for key, value in times.items():
		first_arg =(value[1]/np.sum(t))*R
		#second_arg = value[0]
		second_arg = value[1]*value[0]
		utilities[key] = [value[0], value[1], (first_arg - second_arg)]
	return uo, utilities

def wresults(R_opt, time_opt, umax, ui_opt, nusers, cost):
	"""Write simulation results in file results.txt"""
	#TITLE = 'NUSERS, COST: R_opt, UO_opt, UI_opt, T_opt'
	with open(args.path, 'a') as f:
		line = f'{nusers},{cost},{R_opt},{umax},{ui_opt},{time_opt}\n'
		f.write(line)

def sim_plots(nusers, rewards, u0s, uis, timeui, cost):

	umax = np.amax(u0s)
	pos = u0s.index(umax)
	
	R_opt = rewards[pos]
	time_opt = timeui[pos]
	ui_opt = uis[pos]
	#"""
	sns.set_style('whitegrid')

	#ax1 = sns.lineplot(rewards, u0s, ax = ax1)
	
	ax1 = sns.lineplot(rewards, u0s)
	ax1.set_title('Platform profit')
	ax1.set(xlabel='R', ylabel='profit')
	#ax1.legend(title = f'n clients: {nusers}\ncost: {cost}\nmax: {umax:.2f}\noptimal R: {R_opt}')
	plt.show()
	#ax2 = sns.lineplot(rewards, uis, ax = ax2)
	ax2 = sns.lineplot(rewards, uis)
	ax2.set_title('Mean user profit')
	ax2.set(xlabel='R', ylabel='profit')
	#ax2.legend(title = f'n clients: {nusers}\ncost: {cost}\noptimal R: {R_opt}\noptimal ui: {ui_opt:.2f}')
	plt.show()
	#ax3 = sns.lineplot(rewards, uis, ax = ax3)
	ax3 = sns.lineplot(rewards, timeui)
	ax3.set_title('Mean user sensing time')
	ax3.set(xlabel='R', ylabel='times')
	ax3.legend(title = f'n clients: {nusers}\ncost: {cost}\noptimal R: {R_opt}\noptimal time: {time_opt:.2f}')
	plt.show()
	#"""
	"""
	#Use this when using optimal_sim.py
	wresults(R_opt, time_opt, umax, ui_opt, nusers, cost)
	"""

def simulation(nusers, cost, R, l):
	#CREATE USERS
	u = generate_users(nusers, cost)
	# NASH EQULIBRIUM TIMES
	times, selected = nash_equilibrium(u, R, nusers)
	#UTILITIES
	uo, utilities = compute_utilities(times, R, l) 
	#utilities user: [cost, time, utility]
	t = np.mean([v[1] for v in utilities.values()]) #mean value for users
	ui = np.mean([v[2] for v in utilities.values()]) #mean value for users

	return uo, ui, t

def main():
	nusers = args.nusers
	cost = args.cost
	rangeR = args.rangeR
	l = args.l

	print(f'number of users: {nusers}')
	print(f'cost: {cost}')
	print(f'lambda: {l}')
	
	rewards = [] #R values
	u0s = [] #platform utilities
	uis = [] #mean user utilities
	timeui = [] #mean user times

	for r in range(rangeR[0], rangeR[1]):
		a, b, c = simulation(nusers, cost, r, l)
		rewards.append(r)
		u0s.append(a)
		uis.append(b)
		timeui.append(c)

	sim_plots(nusers, rewards, u0s, uis, timeui, cost)

if __name__ == '__main__':
	main()
