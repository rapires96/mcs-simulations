import numpy as np
import pandas as pd
import os
import sys
import math
from matplotlib import pyplot as plt
import seaborn as sns
import random
import argparse

class node:
	def __init__(self, symbol, prob, left=None, right=None):
		# probability
		self.prob = prob
		# symbol or character
		self.symbol = symbol
		# node left from current node
		self.left = left
		# node right from current node
		self.right = right
		# tree direction (0/1)
		self.huff = ''
#print(np.random.uniform)
class HuffmanTree:
	def __init__(self, S=[], W=[], P=None): # T=None, R=None
		#strategies
		self.strategies = S
		#omegas values
		self.weights = W
		#
		self.prob = P

	def generate_tree(self):
		#S = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
		#OMEGAS = [1,1,1,1,1,1,1,1]
		total = np.sum(self.weights)
		P = [i/total for i in self.weights]
		nodes = []
		for i in range(len(self.strategies)):
			nodes.append(node(self.strategies[i],P[i]))

		nodes = sorted(nodes, key = lambda x: x.prob)
		tree = {}
		while len(nodes) > 1:
		    nodes = sorted(nodes, key = lambda x: x.prob)
		    left = nodes[0]
		    right = nodes[1]

		    newNode = node(left.symbol+right.symbol, left.prob+right.prob, left.symbol, right.symbol)
		    tree[left.symbol] = left
		    tree[right.symbol] = right
		    nodes.remove(left)
		    nodes.remove(right)
		    nodes.append(newNode)

		root = nodes[0]
		tree[root.symbol] = root

		#WP =  {self.strategies[i]: P[i] for i in range(len(S))}
		self.prob = {self.strategies[i]: P[i] for i in range(len(self.strategies))}
		
		return tree, root

	def draw_strat(tree, root):

		#tree, root = self.tree, self.root?
		U = np.random.uniform()
		#print(U)
		nextN = root.symbol
		l = tree[nextN].symbol
		r = tree[nextN].symbol
		#print(nextN)
		while tree[nextN].left != None:
		    if U < tree[l].prob:
		        nextN = tree[l].symbol
		    else:
		        U = U - tree[l].prob
		        nextN = tree[r].symbol
		    l = tree[nextN].left
		    r = tree[nextN].right

		return nextN

def main():
	S = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
	OMEGAS = [1,1,1,1,1,1,1,1]
	tree, root = generate_tree(S, OMEGAS)
	sample = draw_strat(tree, root)
	print(sample)


if __name__ == '__main__':
	main()



"""
	@property
	def prob(self):
		return self.prob

	@prob.setter
	def prob(self, p):
		self.prob = p
"""