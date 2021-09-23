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

PATH = os.getenv('PATH')

def get_arguments():
	"""Argument parser"""
	parser = argparse.ArgumentParser(description = 'MCS algorithm')
	parser.add_argument('-f', '--folder', type=str, default = PATH)
	return parser.parse_args()
args = get_arguments()

def recallSR(qfile):
	
	df = pd.read_csv(qfile)
	trueP = list(df.tspos.values)
	falseN = list(df.tsneg.values)

	recall = lambda x, y: float(x/(x + y))
   
	sim_recall = [recall(trueP[i], falseN[i]) for i in range(len(df))]

	return np.mean(sim_recall), np.max(sim_recall), np.min(sim_recall), np.std(sim_recall)

def plot_recall(dfM):

	sns.set_style('whitegrid')
	ax = sns.lineplot('reward','mean',hue='algorithm',
	 dashes=False, ci=None, marker='o', data = dfM)
	#ax.set_title('Platform utility')
	ax.set(xlabel='R', ylabel='recall')
	plt.show()

def main():
	""""""
	sortR = lambda s: int(str(str(s.rsplit('/',1)[1]).split('l')[0]).split('R')[1]) #R50l50t1
	#Simulation Folder
	folderS = glob.glob(str(Path(args.folder,'MCS*')))

	#c = [f.rsplit('/', 1)[1] for f in folderS]
	#c = c + ['R']
	dfM_recall = pd.DataFrame(columns = ['mean', 'reward', 'algorithm']) #mean recall per strategy and reward

	for f in folderS:
		alg = f.rsplit('/',1)[1]
		folderR = glob.glob(str(Path(f, 'R*')))
		folderR.sort(key=sortR)
		for q in folderR:
			R = sortR(q)
			qfile = str(Path(q,'QoI_sheet.csv'))
			recall_avg, recall_max, recall_min, recall_std = recallSR(qfile)
			#print(f'Algorithm: {alg}, R: {R}\n mean: {recall_avg}, maximum: {recall_max},\
#minimum: {recall_min}, standard dev: {recall_std}')

			dfM_recall.loc[len(dfM_recall)] = [recall_avg, R, alg]
	print(dfM_recall)
	plot_recall(dfM_recall)

if __name__ == '__main__':
	main()