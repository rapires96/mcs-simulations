"""Utility maximization, based on D Yang Nash Equilibrium Algorithm"""
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
REWARD = 100
LAMBDA = 10

#------------------------------------------------

def get_arguments():
	"""Argument parser"""
	parser = argparse.ArgumentParser(description = 'Yang Nash Equlibrium for Stackelberg Game')
	parser.add_argument('-n','--nusers', type=int, default=NUSERS)
	parser.add_argument('-c','--cost', type=int, default=COST)
	parser.add_argument('-r','--rangeR',nargs = '+', type=int, default=REWARD)
	parser.add_argument('-l','--l', type=int, default=LAMBDA)
	return parser.parse_args()

args = get_arguments()

#------------------------------------------------
#Secondary Methods
def control_statement(key, users, selected):
	"""Control statement value to select
	users in the group of winners"""
	numerator = users[key] + np.sum(list(selected.values()))
	denominator = len(selected)
	formula = numerator/denominator
	return formula

def create_usr(nusers, cost):
	"""Generates n users with costs cost"""
	u = {}
	for i in range(nusers):
		key = '{0:0>3d}'.format(i)
		u[key] = cost
	#SORT ACCORDING TO UNIT COSTS OF USERS
	usr = {k: v for k, v in sorted(u.items(), key=lambda item: item[1])}
	return usr

def nash_eq(users, R, nusers):
    """Generates n users with costs cost"""
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
    while users[key] < control_statement(key, users, selected) and i < nusers:
        i += 1
        selected[key] = value
        if i != nusers:
            key, value = next(item)
        else:
            break

    #CALCULATE TIMES FOR THE USERS
    times = {}
    for key, value in users.items():
        if key in selected.keys():
            times[key] = [value]
        else:
            times[key] = [0]
    users = times
    #print(f'USERS: {users}')
    #print(f'SELECTED: {selected}')
    return users
    #print(f'times: {times}')

def compute_utilities(users, R, l, w):    
    #CALCULATE UTILITIES
    c = [value[0] for value in users.values()]
    U = [math.log(1+w) for ci in c]
    uo = l * math.log(1 + np.sum(U)) - R
    #print(uo)
    utilities = {}
    for key, value in users.items():
        first_arg =(value[0]/np.sum(c))*R
        second_arg = value[0]
        utilities[key] = [value[0], first_arg - second_arg]
    users = utilities
    return uo, users

#---------------------------------------

def sim_plots(nusers, rewards, u0s, uis, cost):
    # plots 
	umax = np.amax(u0s)
	pos = u0s.index(umax)
	R_opt = rewards[pos]

    #two plots in one
	fig, (ax1, ax2) = plt.subplots(ncols= 2, nrows=1)
	fig.subplots_adjust(wspace = 0.3)
	sns.set_style('whitegrid')

	ax1 = sns.lineplot(rewards, u0s, ax = ax1)	
	#ax1 = sns.lineplot(rewards, u0s)	
	ax1.set_title('Platform utility')
	ax1.set(xlabel='R', ylabel='utility')
	ax1.legend(title = f'n clients: {nusers}\ncost: {cost}\nmax: {umax:.2f}\noptimal R: {R_opt}')
	plt.show()
	ax2 = sns.lineplot(rewards, uis, ax = ax2)
	#ax2 = sns.lineplot(rewards, uis)
	ax2.set_title('Mean user utility (R)')
	ax2.set(xlabel='R', ylabel='utility')
	ax2.legend(title = f'n clients: {nusers}\ncost: {cost}')
	plt.show()

#---------------------------------------

def simulation(nusr, cost, R, l, w): # number of users, cost, reward, lambda
    usr = create_usr(nusr, cost) #dictionary of users. {usr_id : cost}
    usr = nash_eq(usr, R, nusr)
    uo, usr = compute_utilities(usr, R, l, w)
    ui = np.mean([v[1] for v in usr.values()]) #mean value for users
    return uo, ui

#---------------------------------------

def main():
	nusers = args.nusers
	cost = args.cost
	rangeR = args.rangeR
	l = args.l

	print(f'number of users: {nusers}')
	print(f'cost: {cost}')
	print(f'lambda: {l}')
	#print(f'Reward (R): {R}')
	
	rewards = [] #R values
	u0s = [] #platform utilities
	uis = [] #mean user utilities
	timeui = [] #mean user times

	print(f'CROWDSOURCER UTILITIES: {}')
	print(f'')

	for r in range(rangeR[0], rangeR[1]):
		a, b, c = simulation(nusers, cost, r, l)
		rewards.append(r)
		u0s.append(a)
		uis.append(b)
		timeui.append(c)

	sim_plots(nusers, rewards, u0s, uis, cost)


if __name__ == '__main__':
	main()