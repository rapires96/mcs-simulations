import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob
import os
import sys
import math
import seaborn as sns
import random
import argparse
from pathlib import Path

PATH = ''

def get_arguments():
	"""Argument parser"""
	parser = argparse.ArgumentParser(description = 'MCS algorithm')
	parser.add_argument('-f', '--folder', type=str)
	return parser.parse_args()

args = get_arguments()

def plot_utilities():
	"""Awesome"""
	sns.set_style('whitegrid')
	ax = sns.lineplot(x = 'R', y = 'uo', data = dfR)


def get_Rmeans(sims):
	"""get the mean values for every R"""
	c = ['R', 'uo', 'ui', 'cost', 'samples', 'density', 'mahalanobis', 'tserr', 'tspos', 'tsneg']
	dfR = pd.DataFrame(columns = c)
	for s in sims:
		file_qoi = Path(s, 'QoI_sheet.csv')
		
		R = int(str(str(s.rsplit('/',1)[1]).split('l')[0]).split('R')[1])
		df = pd.read_csv(file_qoi)

		uo, ui, cost = float(df.uo.mean()), float(df.ui.mean()), float(df.cost.mean())
		samples, maha, den = float(df.samples.mean()), float(df.mahalanobis.mode()), float(df.density.mode())
		tserr, tspos, tsneg = float(df.tserr.mean()), float(df.tspos.mean()), float(df.tsneg.mean())

		dfR.loc[len(dfR)] = [R, uo, ui, cost, samples, maha, den, tserr, tspos, tsneg]

	dfR = dfR.sort_values('R', ascending=True)
	return dfR

def plot_utilities(df_alg):
	"""Awesome"""
	sns.set_style('whitegrid')
	ax = sns.lineplot('R','uo',hue='algorithm',
	 dashes=False, ci=None, marker='o', data = df_alg)
	#ax.set_title('Platform utility')
	ax.set(xlabel='R', ylabel='Crowdsourcer profit')
	plt.show()

	sns.set_style('whitegrid')
	ax = sns.lineplot('R','ui',hue='algorithm',
	 dashes=False, ci=None, marker='o', data = df_alg)
	#ax.set_title('Platform utility')
	ax.set(xlabel='R', ylabel='User profit')
	plt.show()

	sns.set_style('whitegrid')
	ax = sns.lineplot('R','samples',hue='algorithm',
	 dashes=False, ci=None, marker='o', data = df_alg)
	#ax.set_title('Platform utility')
	ax.set(xlabel='R', ylabel='Number of samples')
	plt.show()

def main():
	"""hello"""
	folder = args.folder
	if 'MCS' in folder:
		sims = glob.glob(str(Path(folder, '*')))
		for s in sims:
			if 'copies' in s: del sims[sims.index(s)]
			elif 'QoI.csv' in s: del sims[sims.index(s)]
		
		dfR = get_Rmeans(sims)
		print(f"{folder.rsplit('/',1)}:\n{dfR}")
		fileQOI = Path(folder, 'QoI.csv')
		dfR.to_csv(fileQOI, sep = ',', header = True)

	else:
		sim_folders = glob.glob(str(Path(folder, 'MCS*')))
		#add alg column
		df_alg = pd.DataFrame()
		for f in sim_folders:
			A = f.rsplit('/',1)[1]
			pathdf = str(Path(f, 'QoI.csv'))
			dfR = pd.read_csv(pathdf)
			n = len(dfR)
			dfR['algorithm'] = [A for i in range(n)]
			df_alg = df_alg.append(dfR, ignore_index=True)
			#(dfW, ignore_index=True)
		print(df_alg)
		plot_utilities(df_alg)

	#dfR = dfR.sort_values('R', ascending=True)

if __name__ == '__main__':
	main()