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


def utilitiesSR(qfile):
	""""""
	df = pd.read_csv(qfile)
	nsamples = list(df.samples.values)
	uis = list(df.ui.values)
	uos = list(df.uo.values)
	SWs = [uos[i] + (nsamples[i] * uis[i]) for i in range(len(df))]

	return np.mean(uos), np.min(uos), np.max(uos), np.std(uos), np.mean(SWs)

def plot_sw(df):
	print('send nudes')
	sns.set_style('whitegrid')
	ax = sns.lineplot('R','mean',hue='algorithm',
	 dashes=False, ci=None, marker='o', data = df)
	#ax.set_title('Platform utility')
	ax.set(xlabel='R', ylabel='Mean Social Welfare')
	plt.show()

def main():
	""""""
	sortR = lambda s: int(str(str(s.rsplit('/',1)[1]).split('l')[0]).split('R')[1])
	folderS = glob.glob(str(Path(args.folder,'MCS*')))

	dfSW = pd.DataFrame(columns = ['mean', 'R', 'algorithm'])

	for f in folderS:
		alg = f.rsplit('/',1)[1]
		folderR = glob.glob(str(Path(f, 'R*')))
		folderR.sort(key=sortR)
		for q in folderR:
			R = sortR(q)
			qfile = str(Path(q,'QoI_sheet.csv'))
			uoMean, uoMin, uoMax, uoStd, swMean = utilitiesSR(qfile)

			dfSW.loc[len(dfSW)] = [swMean, R, alg]

			#print(f'Algorithm u0: {alg}, R: {R}\n mean: {uoMean}, standard dev: {uoStd}, \
#maximum: {uoMax}, minimum: {uoMin}\n Social Welfare: {swMean}')

	plot_sw(dfSW)

if __name__ == '__main__':
	main()
