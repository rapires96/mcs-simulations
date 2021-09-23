#statistical analysis
import numpy as np
import pandas as pd
import os
import sys
import math
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import re
from pathlib import Path

PATH = ''
def get_arguments():
	"""Argument parser"""
	parser = argparse.ArgumentParser(description = 'Obtain stats from simulation')
	parser.add_argument('-p', '--path', type=str, default=PATH)
	return parser.parse_args()

args = get_arguments()

def readData(file, df, df_counts, df_weights):

	with open(file, 'r') as f:
		lines = f.readlines()

	for line in lines[1::]:
		
		uo,ui,cost,samples,density,maha,tserr,tspos,tsneg,_ = tuple(str(line.split('{')[0]).split(','))
		second_half = str(line.split('{',1)[1])
		aCounts = second_half.split('}', 1)[0]
		if 'sweight' in lines[0]:
			aWeights = second_half.split('}', 1)[1]
			aWeights = aWeights.split(',',1)[1]
			aWeights = re.sub(r'[{}]','', aWeights)
		
		df.loc[len(df)] = [float(uo),float(ui),float(cost),int(samples),
						   float(density),float(maha),float(tserr),int(tspos),int(tsneg)]
		#insert counts into dataframe
		dCounts = {}
		for item in aCounts.split(','):
			key = item.split(':')[0]
			key = re.sub(r"'", '', key)
			key = key.strip()
			val = item.split(':')[1]
			val = int(val.strip())
			dCounts[key] = [val]
		for c in list(df_counts.columns):
			if c not in dCounts.keys(): dCounts[c] = [0]
		dfC = pd.DataFrame.from_dict(dCounts)
		df_counts = df_counts.append(dfC, ignore_index=True)

		if 'sweight' in lines[0]:
			#insert weights into dataframe
			dWeights = {}
			for item in aWeights.split(','):
				key = item.split(':')[0]
				key = re.sub(r"'", '', key)
				key = key.strip()
				val = item.split(':')[1]
				val = float(val.strip())
				dWeights[key] = [val]
			for c in list(df_weights.columns):
				if c not in dWeights.keys(): dWeights[c] = [0]
			dfW = pd.DataFrame.from_dict(dWeights)
			df_weights = df_weights.append(dfW, ignore_index=True)

	if 'sweight' in lines[0]:
		return df, df_counts, df_weights
	else:
		return df, df_counts

def weights_plot(df_weights):

	print(df_weights.mean(axis=0))
	print(type(df_weights.mean(axis=0)))
	print(df_weights.mean(axis=0)[0])

	dWeights = {}
	for idx, k in enumerate(df_weights.columns):
		dWeights[k] = df_weights.mean(axis=0)[idx]
	
	c = list(dWeights.keys())
	val = list(dWeights.values())
	p = [v/np.sum(val) for v in val]

	sns.set_style('whitegrid')
	axes = sns.barplot(c, p)  # new bars
	axes.set(xlabel='Class', ylabel='Probability')
	plt.ylim(0, 0.75)
	plt.show()

	p = [0.125 for i in range(len(dWeights))]

	axes = sns.barplot(c, p)  # new bars
	axes.set(xlabel='Class', ylabel='Probability')
	plt.ylim(0, 0.75)
	plt.show()
	

def main():

	path = args.path
	nsim = int(str(path.rsplit('results',1)[1]).split('sim')[0])
	df = pd.DataFrame(columns = ['uo','ui','cost','samples','density','mahalanobis','tserr','tspos','tsneg'])
	df_counts = pd.DataFrame(columns = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'])
	df_weights = pd.DataFrame(columns = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'])

	print(f'Number of Simulations: {nsim}')

	if 'MCS_Alg' in path:
		df, df_counts, df_weights = readData(path, df, df_counts, df_weights)
	else:
		df, df_counts = readData(path, df, df_counts, df_weights)

	if 'MCS_Alg' in path:
		weights_plot(df_weights)

	# uo,ui,cost,samples,density,maha,tserr,tspos,tsneg
	maha, counts1 = np.unique(list(df.mahalanobis), return_counts = True)
	den, counts2 = np.unique(list(df.density), return_counts = True)
	
	#Write simulation results into CSV files
	wpath = path.rsplit('/', 1)[0]

	excel = Path(wpath, 'QoI_sheet.csv')
	excelW = Path(wpath, 'Weight_sheet.csv')
	excelC = Path(wpath, 'Counts_sheet.csv')

	df.to_csv(excel, sep = ',', header = True)
	df_counts.to_csv(excelC, sep = ',', header = True)
	if 'MCS_Alg' in path:
		df_weights.to_csv(excelW, sep = ',', header = True)

if __name__ == '__main__':
	main()

#read data
#sim mean values 
#print(f'Counts\n {df_counts.mean(axis=0)}')
#print(f'Samples\n {df.samples.mean()}')
#print(f'True positives: {df.tspos.mean()}')
#print(f'False Negatives: {df.tsneg.mean()}')
#print(f'TS: {df.tserr.mean()}')
#print(f'CW utility: {df.uo.mean()}')
#print(f'User Utility: {df.ui.mean()}')
#print(min(maha))
#print(min(den))
#print(df_counts)