__author__ = 'Rajkumar Pillai'

import numpy as np
import csv
import matplotlib.pyplot as plt
import math
import random
import time
from scipy.spatial import distance

"""
file: HW_Pillai_Rajkumar_kMeans.py
CSCI-720:  Big data analytics
Author: Rajkumar Lenin Pillai

Description: This program implements the kmeans algorithm as mentioned in the homework pdf
"""

def SSE_calc(New_centroid,clusters_formed):

    '''
    This calculates the SSE using euclidean distance or L2 Norm
    :param New_centroid: The new centroid computed using mean values
    :param clusters_formed: Every position in this array stores data points that belong to a cluster where position_no = cluster_no
    :return: result: The Sum of squared error value
    '''


    result=0
    ##Iterating through every cluster
    for i in range(0,len(clusters_formed)):
        single_cluster = clusters_formed[i]
        for j in range(0,len(single_cluster)):
            ## Computing SSE using euclidean distance
            difference=distance.euclidean(single_cluster[j],New_centroid[i])
            difference=math.pow(difference,2)
            result+=difference


    return result

def L2_norm_distance(x,y):

    '''
    This function calcuates the euclidean distance between two points
    :param x: The co-ordinates of data point
    :param y: The co-ordinates of centroid
    :return:distance_list: The list which coantins all euclidean distance of each point
    '''

    distance_list=[]
    for j in y:
       result=distance.euclidean(np.array(x),np.array(j))
       distance_list.append(result)



    return distance_list


def assign_data_points_to_cluster(data, Initial_centroids):

    '''
    This function assigns clusters to data points
    :param data: The set of data points
    :param Initial_centroids: The random generated centroid
    :return: cluster_list: The list where every index corrrespond to cluster number of data point
    '''


    cluster_list = []
    for row in data:
        ## Using Euclidean distance and then finding the min and assigning cluster no to the point
        distances_list = L2_norm_distance(row, Initial_centroids)
        cluster = min(distances_list)
        cluster_no = distances_list.index(cluster) + 1
        cluster_list.append(cluster_no)
    return cluster_list


def seperate_data_points_for_each_cluster(data,k,cluster_list):
    '''
    This function seperates the data points and keeps them together if they belong to same cluster
    :param data: The set of data points
    :param k:  The k value
    :param cluster_list: The list where every index corrrespond to cluster number of data point
    :return: clusters_formed : List with list position correpsonding to cluster no and values in position are data points of same cluster
    '''
    clusters_formed = [None] * (k + 1)
    for i in range(1, k + 1):
        rows_list = []
        for j in range(0, len(cluster_list)):
            if cluster_list[j] == i:
                rows_list.append(data[j])
        clusters_formed[i] = rows_list
    return clusters_formed


def calc_new_centroid(clusters_formed,k):

    '''
    This function computes the mean of every point belonging  to cluster to compute the co-ordinates for new centroid
    :param clusters_formed: List with list position correpsonding to cluster no and values in position are data points of same cluster
    :param k: The k value
    :return: New_centroid : The new computed centroid
    '''
    New_centroid = [None] * (k + 1)
    for i in range(1, len(clusters_formed)):
        w_list = []
        x_list = []
        y_list = []
        z_list = []
        single_cluster = clusters_formed[i]
        # print("Single cluster",single_cluster)
        # print("Cluster no: ",i)
        for points in single_cluster:

            w_list.append(points[0])
            x_list.append(points[1])
            y_list.append(points[2])
            z_list.append(points[3])
        ### Finding mean of all points in same cluster
        w_mean = np.mean(w_list)
        x_mean = np.mean(x_list)
        y_mean = np.mean(y_list)
        z_mean = np.mean(z_list)

        New_centroid[i] = [w_mean, x_mean, y_mean, z_mean]

    return New_centroid


def stopping_criteria(Initial_centroids, New_centroid,SSE_list,k):
    '''
    This function specifies the stopping criteria where the inner loop of kmeans stop when new centroid is same as previous centroid
    :param Initial_centroids: The previous centroid
    :param New_centroid: The new computed centroid
    :param SSE_list: The list having SSE compted in each iteration
    :param k: The k value
    :return: True: if the centroids are same false otherwise
    '''
    SameFlagCounter = 0
    for i in Initial_centroids:
        if i in New_centroid:
            SameFlagCounter += 1
    if SameFlagCounter == len(Initial_centroids):
        return True
    return False


def NO_points_in_cluster(data,k):
    '''
    This is the function used to check if there is any centroid without any data point
    :param data: The set of data points
    :param k: The k value
    :return: data: The set of data points after removing centroid to which no data points were associated
             k: The k value
             Nan_Present: To check If the centorids cluster is empty
             len(index_to_be_deleted): The number of centroids removed which can be used to reduce the k value
    '''


    ## Checking if any centroid without a data point is present
    index_to_be_deleted = []
    Nan_Present=False
    for j in range(len(data)):
        for i in range(0, len(data[j])):
          if math.isnan(data[j][i]):
            index_to_be_deleted.append(j)
            Nan_Present=True
            break

    ##Remving the centroid and reducing the k value
    if Nan_Present:
        data=np.delete(data,(index_to_be_deleted),axis=0)
        k = k - len(index_to_be_deleted)


    return k,data,Nan_Present,len(index_to_be_deleted)




def kmeans_func(data,k,no_of_times=1000):
    '''
    This fucntion calls the helper fucntions and iterates the no of times as specified
    :param data: The set of data points
    :param k: The k value
    :param no_of_times: The No of times the iteration needs to be done for every k
    :return:exectiontime:The time required for execution of inner loop
            SSE_of_k: The SSE value for that k
            New_centroid: The new centorid computed
            clusters_formed_for_calc: The data points in those centroids
    '''

    SSE_list=[] ## Used to store all SSE values
    Clusters_formed_list=[]

    ## The outer loop
    while no_of_times != 0:

        ##Randomly generating initial centroids
        Initial_centroids = np.random.randint(0, max(max(x) for x in data), size=(k, np.size(data, 1)))

        ##Setting the start time for inner loop
        start_time = time.time()


        while True:

            ## Checking if there are centroids not associated with any data point
            k,Initial_centroids,Nan_Present,difference=NO_points_in_cluster(Initial_centroids, k)

            ## Calling the fucntion to get the cluster to which each data point belongs
            cluster_list=assign_data_points_to_cluster(data, Initial_centroids)

            ## Seperating the data points into clsuters
            clusters_formed=seperate_data_points_for_each_cluster(data,k,cluster_list)
            Clusters_formed_list.append(clusters_formed)

            ## Computing New centroid
            New_centroid=calc_new_centroid(clusters_formed,k)
            del New_centroid[0]        ## To remove None which will be present at 0 position
            del clusters_formed[0]     ## To remove None which will be present at 0 position


            New_centroid = np.asarray(New_centroid)

            ##Computing SSE
            SSE=SSE_calc(New_centroid,clusters_formed)
            SSE_list.append(SSE)

            ##Checking if old centroid is equal to new centroid
            if stopping_criteria(Initial_centroids,New_centroid,SSE_list,k):
                exectiontime = time.time() - start_time
                SSE=min(SSE_list)
                break

            ## Updating centroids to  continue next iteration
            Initial_centroids=New_centroid



        ## Decreasing counter for the outer while loop
        no_of_times=no_of_times-1

        ## Updating k vlaue if it was reduced because no points were assigned to the centroid
        if Nan_Present:
            k=k+difference



    SSE_of_k=min(SSE_list)

    ## Clusters with lowest SSE
    clusters_formed_for_calc=Clusters_formed_list[SSE_list.index(min(SSE_list))]


    return exectiontime,SSE_of_k,New_centroid,clusters_formed_for_calc



def Plot_ExecutionTime_vs_K(k_list,Execution_Time_list):
        '''
        This function plots the execution time vs k plot
        :param k_list:  The list of k values
        :param Execution_Time_list: Th elist of execution time values
        :return:
        '''
        plt.scatter(k_list,Execution_Time_list,marker='o')
        plt.plot(k_list, Execution_Time_list,color='red')
        plt.title("Execution Time vs K")
        plt.xlabel("K Values")
        plt.ylabel("Execution Time")
        plt.xticks(k_list)
        plt.yticks(Execution_Time_list)
        plt.savefig('Execution_Time_VS_K.jpg')
        plt.show()


def Plot_SSE_vs_K(k_list, SSE_list):
    '''
    This function plots the  SSE vs k plot
    :param k_list: The list of k values
    :param SSE_list: The SSE list for each k value
    :return:
    '''
    del SSE_list[0]
    plt.scatter(k_list, SSE_list, marker='o')
    plt.plot(k_list, SSE_list, color='red')
    plt.title("SSE vs K")
    plt.xlabel("K Values")
    plt.ylabel("Execution Time")
    plt.xticks(k_list)
    plt.yticks(SSE_list)
    plt.savefig('SSE_VS_K.jpg')
    plt.show()


def main():
    '''
    The main fucntion
    :return:
    '''



    ## Loading dataset
    data=np.loadtxt('HW_K_MEANS__DATA_v2185.csv',delimiter=',')

    ## Removing the first column
    data=np.delete(data,0,1)


    k = 15 ## Declaring value of k
    no_of_times = 10  ## Decalring the no of times to make iterations for outer loop


    Execution_Time_list=[] ## To store execution time for each value of k and inner lop
    SSE_list=[None]*(k+1)  ## To store the SSE for each k
    k_list=[i for i in range(1,k+1)] ## List of k values
    clusters_formed_list=[None]*(k+1) ## The list storing all clusters
    New_centroid_list=[None]*(k+1)   ## To store the final centroids for each k value


    ## Calling k means for each value of k
    for i in range(1,k+1):

        print("Computing kmeans for k=",i)
        exectiontime,SSE,New_centroid,clusters_formed=kmeans_func(data, k=i, no_of_times=no_of_times)
        Execution_Time_list.append(exectiontime)
        SSE_list[i]=SSE
        New_centroid_list[i] = New_centroid

        ## Computing no of points in each cluster
        clusters_list = []
        for j in range(0,len(clusters_formed)):
            clusters=clusters_formed[j]
            count=0
            for item in clusters:
                count+=1
            clusters_list.append(count)
        clusters_formed_list[i] = clusters_list


    ## Printing the results as mentioned in homework.pdf
    print("ClusterID"," A1 \t A2 \tA3 \t A4","\t\tNum.Points in this cluster","\t\t\t\t\t\tSSE For this cluster")

    for i in range(1, k+1):
     Centroids=New_centroid_list[i]
     pos=0
     for j in Centroids:
      centroid_point=j
      result=map(round,centroid_point) ## To round off the values to one significant digit
      print("\t",i,"\t",list(result),"\t\t\t\t",clusters_formed_list[i][pos-1],"\t\t\t\t\t\t\t\t\t",SSE_list[i],"\t")
      pos+=1


    ## Calling plot function to plot the execution time vs k
    Plot_ExecutionTime_vs_K(k_list, Execution_Time_list)

    ## Calling plot function to plot the SSE vs k
    Plot_SSE_vs_K(k_list, SSE_list)

if __name__ == '__main__':
    main()
