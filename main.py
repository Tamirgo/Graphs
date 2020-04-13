import Graphs as G
import random
import numbers
import numpy as np
import math


#can pick is the graph is directed or not, or is it a clique or not.
new_graph = G.Graphs(['a','b','c','d','e','f','g'])


#a list of tuples, to set more than just one edge, though it is possible to set edgs one by one.
tuple_list = [('a','b'),('a','c'),('c','d'),('e','g'),('e','f'),('b','f'),('a','d')]
new_graph.set_Whole_Neighbours(tuple_list=tuple_list)


#teting the Algorithms.
print (new_graph.getVertecies())
print (new_graph.getEdges())
print (new_graph.BFS('a'))
print (new_graph.DFS())
print (new_graph.setWeights())
print (new_graph.Dijkstra('a'))


if new_graph.check_Connectivity(new_graph):
    print('Graph is connected')
else:
    print('Not Connected.')












