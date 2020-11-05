__author__ = 'Rajkumar Pillai'

import numpy as np
import csv
import matplotlib.pyplot as plt
import math
import random
import time
from scipy.spatial import distance

"""
file: HW_MainPillai_Rajkumar_kMeans.py
CSCI-720:  Big data analytics
Author: Rajkumar Lenin Pillai

Description: This program implements the agglomerative clsutering algorithm using single linkage as mentioned in the homework pdf
"""

from scipy.cluster.hierarchy import dendrogram, linkage
import math
import matplotlib.pyplot as plt
from collections import defaultdict
cluster_dictionary=defaultdict()

# def plotmap(clusters):
#     '''
#     This fucntion plots the capitals of cities in 12 clusters with specific colors
#     :param clusters: The dictionary where key corresponding to cluster no and values are the points in the cluster
#     :return:
#     '''
#     from mpl_toolkits.basemap import Basemap
#
#     ## Creating the basemap object
#     map = Basemap(projection='cyl')
#     map.drawcoastlines()
#
#     ## The list of namespace colors
#     colors_list=['green','blue','pink','orange','brown','black','purple','cyan','red','grey','violet','magenta']
#
#     color_index=0  ## Used to give different colors to the cluster
#
#     for items in clusters:
#         list_of_lat_long = cluster_dictionary[items]
#         for item in list_of_lat_long:
#             ## Gettting latitude and longitude of cities
#             lats = item[0]
#             lons = item[1]
#
#             ##Plotting the points on the map
#             map.scatter(lons, lats, marker='o', color=colors_list[color_index], zorder=30)
#         color_index += 1
#
#     plt.title('World Map')
#     plt.savefig('Map.jpg')
#     plt.show()
def haversine_calc_for_dendrogram(lat_long):
    '''
    This function calcualtes the 50x50 matrix used in dendrogarm plotting
    :param lat_long: The list which contains latitude and longitude of every city
    :return: dist_matrix : The 50x50 matrix which has the haversine distance between all points
    '''
    dist_matrix=[]
    from math import radians

    ## For every co-ordinate
    for i in range(len(lat_long)):
        temp = []
        ## Iterating and finding distance to every other co-ordinate
        for j in range (len(lat_long)):
                ## Getting latitude and longitude of co-ordinates
                lat_long1=lat_long[i]
                lat_long2=lat_long[j]
                lat1=lat_long1[0]
                lat2=lat_long2[0]
                long1=lat_long1[1]
                long2=lat_long2[1]


                long1, lat1, long2, lat2 = map(radians, [long1, lat1, long2, lat2])

                diffrenece_long=long2-long1   ##Finding difference of longitude
                diffrenece_lat = lat2 - lat1  ##Finding differnce  of latitude

                ## Computing distance using haversine distance formula
                result=math.pow(math.sin(diffrenece_lat/2),2)+math.cos(lat1)*math.cos(lat2)*math.pow(math.sin(diffrenece_long/2),2)
                ans=2*math.asin(math.sqrt(result))*6371
                if ans==0.0:
                    temp.append(0.0)
                else:
                    temp.append(ans)
        dist_matrix.append(temp)

    return dist_matrix
def haversine_for_points (point1,point2):
    '''
    This function calculates the haversine distance between two points
    :param point1: The list containing latitude and laongitude of point1
    :param point2: The list containing latitude and laongitude of point2
    :return:result: The  haversine distance between two points
    '''

    from math import radians

    ## Getting the latitude and longitude of points
    lat1,long1=point1[0],point1[1]
    lat2, long2 = point2[0], point2[1]

    ##
    long1, lat1, long2, lat2 = map(radians, [long1, lat1, long2, lat2])
    diffrenece_long = long2 - long1 ##Finding difference of longitude
    diffrenece_lat = lat2 - lat1    ##Finding differnce  of latitude

    ## Computing distance using haversine distance formula

    temp = math.pow(math.sin(diffrenece_lat / 2), 2) + math.cos(lat1) * math.cos(lat2) * math.pow(
        math.sin(diffrenece_long / 2), 2)
    result = 2 * math.asin(math.sqrt(temp))*6371

    return result

def plot_dendrogram(haversine_distance_matrix):
    '''
    This function plots the dendrogram of top 50 clusters
    :param haversine_distance_matrix: The 50 x 50 amtrix with haversine distance between them
    :return:
    '''

    ## Using the linkage function of scipy
    single_link = linkage(haversine_distance_matrix, 'single')

    plt.figure(figsize=(30, 20))
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Clustering')
    plt.ylabel('distances')

    ## plotting using scipy denrogram function
    dendrogram(single_link,color_threshold=.6)
    plt.savefig('Dendrogram.jpg')
    plt.show()



def agglomerate(clusters, k):
    '''
    This function implements the agglomerative clustering algorithm with single linkage method
    :param clusters: The list which contains the latitude and longitude of every point
    :param k: The number of clusters
    :return: cluster_dictionary: The dictionary where key corresponding to cluster no and values are the points in the cluster
    '''



    print('First_cluster_no , ', 'second_cluster_no, ', 'distance')

    ##Initialzing every point as single clusters
    for i  in range(len(clusters)):
        cluster_dictionary[i]=[clusters[i]]

    ## To store the list of initial cluster_ids to be used for considering every cluster
    keys_list = list(cluster_dictionary.keys())

    # Clustering           (
    while True:

        ## For every cluster
        for i in range(len(keys_list)):

            ### To avoid considering the current  cluster if already merged
            if i >len(keys_list)-1 or keys_list[i] not in list(cluster_dictionary.keys()) :
                continue
            first_cluster=keys_list[i]

            ###Initializing the inter_cluster distance
            inter_cluster_dist=first_cluster_no=second_cluster_no=cluster_no_delete=math.inf

            ## For every cluster after the current one
            for j in range(i+1,len(keys_list)):

                ### To avoid considering the next cluster if already merged
                if  j > len(keys_list)-1or keys_list[j] not in list(cluster_dictionary.keys()) :
                    continue

                second_cluster=keys_list[j]

                ## For every point the current considered cluster
                for first_cluster_points in cluster_dictionary[first_cluster]:
                    first_point=first_cluster_points

                    ###Initializing the intra_cluster distance
                    intra_cluster_dist=math.inf


                    ## For every point in the next considered cluster
                    for second_cluster_points in cluster_dictionary[second_cluster]:
                        second_point=second_cluster_points
                        if haversine_for_points(first_point,second_point)<intra_cluster_dist:
                            intra_cluster_dist=haversine_for_points(first_point,second_point)

                ## To consider the cluster using single linkage
                if intra_cluster_dist <inter_cluster_dist:
                    inter_cluster_dist=intra_cluster_dist
                    first_cluster_no=first_cluster
                    second_cluster_no=second_cluster


            ## If the clusters are already merged
            if first_cluster_no==math.inf:
                continue
            else:

                print(first_cluster_no, ' , ', second_cluster_no, ' , ', inter_cluster_dist)

                ## To merge the points of second cluster with point of first cluster
                for points in cluster_dictionary[second_cluster_no]:
                  cluster_dictionary[first_cluster_no].append(points)

                ## To remove the second merged cluster from cluster dictionary
                del cluster_dictionary[second_cluster_no]

                ## To remove the second merged cluster from keys list
                keys_list.remove(second_cluster_no)

                ## Stopping criteria if dictionary conatins 12 clusters
                if len(cluster_dictionary.keys())==k:
                    return cluster_dictionary

# def get_long_and_lat(data):
#     '''
#     This funciton is used to get the latitude and longitude using nominatim
#     :param data: The input data which has list of latitude and longitude of all points
#     :return: lat_long: The with  latitude and longitude of all points
#     '''
#
#     from geopy.geocoders import Nominatim
#
#     geolocator = Nominatim()
#
#     lat_long=[]
#     for i in range(len(data)):
#         location=geolocator.geocode(data[i][0])
#         lat_long.append([location.latitude, location.longitude])
#     return lat_long


def main():
    '''
    The main function
    :return:
    '''

    ## Uncomment these lines and the corresponding function if you want to use the api to download latitude and longitude
    # data = np.loadtxt("CS_720_City_Country.csv", dtype='str', delimiter=',')
    # lat_long=get_long_and_lat(data)


    ## The list initialzed with latitude and longitude of all points which was obtained using the get_long_and_lat which is commented
    lat_long=[[51.5156177, -0.0919983], [34.5260131, 69.1776476], [41.3279457, 19.8185323], [28.0000272, 2.9999825], [42.5069391, 1.5212467], [-8.8272699, 13.2439512], [47.561701, -52.715149], [-34.6075616, -58.437076], [40.1776121, 44.5125849], [-35.2975906, 149.1012676], [48.2083537, 16.3725042], [40.3754434, 49.8326748], [40.7412643, -73.5877699], [26.2235041, 50.5822436], [23.7593572, 90.3788136], [13.0977832, -59.6184184], [53.902334, 27.5618791], [50.8465573, 4.351697], [17.250199, -88.770018], [6.4990718, 2.6253361], [27.4727617, 89.629548], [-19.0477251, -65.2594306], [43.8519774, 18.3866868], [-24.655319, 25.908728], [-10.3333333, -53.2], [4.8895453, 114.9417574], [42.6978634, 23.3221789], [12.3676191, -1.5114997], [-3.3638125, 29.3675028], [14.9160169, -23.5096132], [11.568271, 104.9224426], [3.8689867, 11.5213344], [45.421106, -75.690308], [4.3907153, 18.5509126], [12.1191543, 15.0502758], [-33.4377968, -70.6504451], [39.906217, 116.3912757], [4.5980772, -74.0761028], [-11.6931255, 43.2543044], [-4.3217055, 15.3125974], [37.3361905, -121.8905833], [6.869145, -5.2823277], [45.813177, 15.977048], [23.135305, -82.3589631], [35.180282, 33.3736958666529], [50.0874654, 14.4212535], [55.6867243, 12.5700724], [11.85677545, 42.7577845199437], [15.2991923, -61.3872868], [18.4801972, -69.942111], [-0.2201641, -78.5123274], [30.048819, 31.243666], [13.6989939, -89.1914249], [3.752828, 8.780061], [15.3389667, 38.9326763], [59.4372155, 24.7453688], [-26.325745, 31.144663], [9.0107934, 38.7612525], [6.920744, 158.1627143], [-18.1415884, 178.4421662], [60.1713198, 24.9414566], [48.8566101, 2.3514992], [0.390002, 9.454001], [13.45535, -16.575646], [41.6934591, 44.8014495], [52.5170365, 13.3888599], [5.5600141, -0.2057437], [33.9597677, -83.376398], [32.3342432, -64.6970000009087], [14.6222328, -90.5185188], [9.5170602, -13.6998434], [11.861324, -15.583055], [38.2097967, -84.5588311], [18.547327, -72.3395928], [14.0931919, -87.2012631], [47.4983815, 19.0404707], [64.2444268, -21.7681063037804], [28.6141793, 77.2022662], [-6.1753942, 106.827183], [35.7006177, 51.4013785], [33.3024309, 44.3787992], [53.3497645, -6.2602732], [32.0804808, 34.7805274], [41.894802, 12.4853384], [17.9712148, -76.7928128], [35.6828387, 139.7594549], [31.9515694, 35.9239625], [51.128258, 71.43055], [-1.2832533, 36.8172449], [1.3490778, 173.0386512], [42.6638771, 21.1640849], [29.3797091, 47.9735629], [42.8767446, 74.6069949], [17.9640988, 102.6133707], [56.9493977, 24.1051846], [33.8959203, 35.47843], [-29.310054, 27.478222], [6.328034, -10.797788], [32.896672, 13.1777923], [47.1392862, 9.5227962], [54.6870458, 25.2829111], [49.8158683, 6.1296751], [41.9960924, 21.4316495], [-18.9100122, 47.5255809], [-13.973456, 33.7878122], [3.1516636, 101.6943028], [16.3700359, -2.2900239], [12.60503275, -7.98651367343936], [35.8989818, 14.5136759], [7.0909924, 171.3816354], [18.0792379, -15.9780071], [-20.1637281, 57.5045331], [19.4326009, -99.1333416], [47.0244707, 28.8322534], [43.7311424, 7.4197576], [47.949809, 106.966724193881], [42.4415238, 19.2621081], [34.022405, -6.834543], [-25.966213, 32.56745], [19.7540045, 96.1344976], [-22.5744184, 17.0791233], [-0.5470855, 166.922628264555], [27.708796, 85.320244], [52.3745403, 4.89797550561798], [-41.2887467, 174.7772092], [12.1461244, -86.273717], [13.524834, 2.109823], [9.0643305, 7.4892974], [39.0194741, 125.753388], [59.9133301, 10.7389701], [23.5125498, 58.5514017715319], [33.63573935, 72.9230467027632], [7.5006193, 134.6243012], [31.9030821, 35.1951741], [30.1600827, -85.6545729], [-9.4743301, 147.1599504], [-2.4742565, -78.1843168], [-12.0621065, -77.0365256], [14.5906216, 120.9799696], [52.2319237, 21.0067265], [38.7077507, -9.1365919], [25.2856329, 51.5264162], [-4.2694407, 15.2712256], [44.4361414, 26.1027202], [55.7504461, 37.6174943], [-1.8859597, 30.1296751], [17.2960919, -62.722301], [13.95258925, -60.9878235312987], [13.1561864, -61.2279621], [-13.8343691, -171.7692793], [43.9458623, 12.458306], [0.8875498, 6.9648718], [24.6319692, 46.7150648], [14.693425, -17.447938], [44.8178131, 20.4568974], [-36.5986096, 144.6780052], [8.479004, -13.26795], [1.2904753, 103.8520359], [48.1359085, 17.1597440625], [46.0498146, 14.5067824], [-9.4312971, 159.9552773], [2.042778, 45.338564], [-25.7459374, 28.1879444], [37.5666791, 126.9782914], [4.8472017, 31.5951655], [40.4167047, -3.7035825], [6.9031663, 79.9091644], [15.593325, 32.53565], [5.8216198, -55.1771974], [59.3251172, 18.0710935], [46.9482713, 7.4514512], [33.5130695, 36.3095814], [38.5425835, 68.8152131865304], [-6.1791181, 35.7468174], [13.7538929, 100.8160803], [28.6517178, 77.2219388], [6.130419, 1.215829], [-19.9160819, -175.2026424], [10.6572678, -61.5180173], [33.8439408, 9.400138], [39.9215219, 32.8537929], [37.9404379, 58.3822788], [-8.53499465, 179.118649634252], [0.3177137, 32.5813539], [50.4500644, 30.5241037], [23.99764435, 53.6439097569213], [51.5073219, -0.1276474], [38.8950092, -77.0365625], [-34.9059039, -56.1913569], [41.3123363, 69.2787079], [-17.7414972, 168.3150163], [41.9034912, 12.4528349], [10.506098, -66.9146017], [21.0294498, 105.8544441], [15.342101, 44.2005197], [-15.4164488, 28.2821535], [-17.831773, 31.045686]]



    k=12  ## Declaring no of clusters

    ## Calling the fucntion to exceute the agglomeration algorithm
    clusters=agglomerate(lat_long, k)

    ## Uncomment the below line to plot the map
    #plotmap(clusters)

    ## Storing the top 50 clusters in a list
    list_of_points_in_clsuter= cluster_dictionary[0] + cluster_dictionary[1] + cluster_dictionary[3]

    ## To compute the distance between all points in top 50 clusters
    dist_matix=haversine_calc_for_dendrogram(list_of_points_in_clsuter[:50])

    ## To plot the dendrogram
    plot_dendrogram(dist_matix)


if __name__ == '__main__':
    main()