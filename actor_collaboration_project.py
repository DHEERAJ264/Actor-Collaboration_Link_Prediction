import networkx as nx
from igraph import *
from math import *
import sys
import numpy as np
import os
import glob
import random
from random import shuffle
from random import seed
import matplotlib.pyplot as plt
import time
import datetime
import collections
import csv
import pickle


# Constants
k = 5 # Top k recomendations for a target user






#load edges from the dataset

def get_edges(data_path):
	data_file = open(data_path)
	edge_list =list(map(lambda x:tuple(map(int,x.split())),data_file.read().split("\n")[:-1]))
	#Print("I am here in get edge list to debug the function")
	#edge_list =list(map(lambda x:list(map(int,x.split())),data_file.read().split("\n")[:-1]))
	data_file.close()
	return edge_list





#Get the similarity product for a path of the edges by calculating the length of shortest path

def get_similarity_product(sim,shortest_path):
	product = 1
	for i in range(len(shortest_path) - 1):
		product *= sim[shortest_path[i]][shortest_path[i+1]]
		
	return round(product,2)



#Filter out, Sort and Get top-K predictions in the descending order based on the similarity scores from the graph.
def get_top_k_recommendations(graph,sim,i,k):
	return sorted(filter(lambda x: i!=x and graph[i,x] != 1,range(len(sim[i]))),key=lambda x: sim[i][x],reverse=True)[0:k]


###get the vertices set from the edge list
def get_vertices(edge_list):
	result = set()
	for x,y in edge_list:
		result.add(x)
		result.add(y)
	#print(result)
	return result


#Split the dataset into two parts based on the indexes of each edege - Train and Test Part

def datasplit(edge_list):
	random.seed(350)
	indexes = range(len(edge_list))
	test_indexes = set(random.sample(indexes, int(len(indexes)/2)))
	train_indexes = set(indexes).difference(test_indexes)
	test_list = [edge_list[i] for i in test_indexes]
	train_list = [edge_list[i] for i in train_indexes]
	#print("Seperating them of the list compreshension with cross validation")
	print(train_list)
	return train_list,test_list


#Calculates accuracy metrics (Precision & Recall),
# for a given similarity-model against a test-graph

def print_precision_and_recall(sim,traininggraph,testinggraph,test_vertices,train_vertices):
	precision = recall = c = 0
	for i in test_vertices:
		if i in train_vertices:
			actual_actors_of_i = set(testinggraph.neighbors(i))
			if len(actual_actors_of_i) < k:
				k2 = len(actual_actors_of_i)
			else:
				k2 = k

			top_k = set(get_top_k_recommendations(traininggraph,sim,i,k2))


			precision += len(top_k.intersection(actual_actors_of_i))/float(k2)
			recall += len(top_k.intersection(actual_actors_of_i))/float(len(actual_actors_of_i))
			c = c+1
	print ("Precision is : " + str(precision/c))
	print ("Recall is : " + str(recall/c))
	print ("F1-Score is :" + str(2 * ((precision/c * recall/c)/(precision/c + recall/c))))


#http://be.amazd.com/link-prediction/ ####Source To caluculate the formulas of the Prediction Algortihms####
def similarity(graph, i, j, method):
	if method == "common_neighbors":
		return len(set(graph.neighbors(i)).intersection(set(graph.neighbors(j))))
	elif method == "jaccard":
		return len(set(graph.neighbors(i)).intersection(set(graph.neighbors(j))))/float(len(set(graph.neighbors(i)).union(set(graph.neighbors(j)))))
	elif method == "adamic_adar":
		return sum([1.0/math.log(graph.degree(v)) for v in set(graph.neighbors(i)).intersection(set(graph.neighbors(j)))])
	elif method == "resource_allocation":
		return sum([1.0/graph.degree(v) for v in set(graph.neighbors(i)).intersection(set(graph.neighbors(j)))])
	elif method == "preferential_attachment":
		return graph.degree(i) * graph.degree(j)


def local_methods(edge_list,method):
    train_list, test_list = datasplit(edge_list)
    #with open('ur_test.csv','w') as out:
     #   csv_out=csv.writer(out)
      #  csv_out.writerows(test_list)
    #with open('ur_train.csv','w') as out:
    #	csv_out=csv.writer(out)
    #	csv_out.writerows(train_list)
    #print("########################################################################################################################################")
    traininggraph = Graph(train_list)
    testinggraph = Graph(test_list)
    print(testinggraph)
    train_n =  traininggraph.vcount() # This is maximum of the vertex id 
    train_vertices = get_vertices(train_list) # Need this because we have to only consider target users who are present in this set
    test_vertices = get_vertices(test_list) # Set of target value users
    sim = [[0 for i in range(train_n)] for j in range(train_n)]## List comprehension
    for i in range(train_n):
    	for j in range(train_n):
    		if i!=j and i in train_vertices and j in train_vertices:
    			sim[i][j] = similarity(traininggraph,i,j,method)
    			#print(sim)			
    print_precision_and_recall(sim,traininggraph,testinggraph,test_vertices,train_vertices)


########################Main Function #############################
def main():
	if len(sys.argv) < 3 :
		print ("python link_prediction.py <common_neighbors/jaccard/adamic_adar/resource_allocation/preferential_attachment> data_file_path")
		exit(1)

	# Command line argument Parsing during the runtime
	method = sys.argv[1].strip()
	data_path = sys.argv[2].strip()
	edge_list = get_edges(data_path)
	if method == "common_neighbors" or method == "jaccard" or method == "adamic_adar" or method == "resource_allocation" or method == "preferential_attachment":
		local_methods(edge_list,method)
	else:
		print ("python link_prediction.py <common_neighbors/jaccard/adamic_adar/resource_allocation/preferential_attachment> data_file_path")

if __name__ == "__main__":
	main()

