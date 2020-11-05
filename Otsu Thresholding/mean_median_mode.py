__author__ = 'Rajkumar Pillai'

import numpy as np
from scipy import stats
import  matplotlib.pyplot as plt

"""

CSCI-720:  Big data analytics
Author: Rajkumar Lenin Pillai

Description: This program computes the mean, median and mode and also the threshold speed to binarize the given data 
"""

def weight(hist,bins,start,end,total_elements):

    '''
    This computes the weight
    :param hist:              The histogram frequency values
    :param bins:              The speed values in each bin
    :param start:             To specify the start position of data to be considered
    :param end:               To specify the end position of data to be considered
    :param total_elements:    Total no of data points considered
    :return w:              The computed weight value
    '''

    w=0
    if total_elements==0:
        return total_elements

    ## Computing the weight
    sum_of_weights = sum(hist)
    for i in range(start, end):
        w+=hist[i]
    w=w/sum_of_weights
    return w

def mean(hist,bins,start,end,total_elements):

    '''
    This computes the mean
    :param hist:              The histogram frequency values
    :param bins:              The speed values in each bin
    :param start:             To specify the start position of data to be considered
    :param end:               To specify the end position of data to be considered
    :param total_elements:    Total no of data points considered
    :return mean_value:       The computed mean
    '''

    mean_variable=0
    if total_elements==0:
        return mean_variable

    ## Computing the mean
    for i in range(start, end):
        mean_variable+=(bins[i]*hist[i])
    mean_value=mean_variable/total_elements
    return mean_value

def variance(hist,bins,start,end,mean_value,total_elements):
    '''
    This computes the variance
    :param hist:              The histogram frequency values
    :param bins:              The speed values in each bin
    :param start:             To specify the start position of data to be considered
    :param end:               To specify the end position of data to be considered
    :param mean_value:        To mean of data
    :param total_elements:    Total no of data points considered
    :return variance_value:   The computed variance
    '''

    variance_variable = 0
    if total_elements==0:
        return variance_variable

    ## Computing the variance
    for i in range(start, end):
        variance_variable += (((bins[i] - mean_value)**2) * hist[i])
    variance_value=variance_variable/total_elements

    return variance_value


def calc_total_no_of_data_points(hist,start,end):
    '''
    This computes the total no of data points
    :param hist:              The histogram frequency values
    :param start:             To specify the start position of data to be considered
    :param end:               To specify the end position of data to be considered
    :return: total_no_of_data_points: To compute the no of data points
    '''
    total_no_of_data_points = 0
    for i in range(start, end):
        total_no_of_data_points += hist[i]
    return total_no_of_data_points

def mixed_variance(weight_value_b,variance_value_b,weight_value_f,variance_value_f):
    '''
    This computes the mixed variance of given thresdold and returns the computed mixed variance
    :param weight_value_b:          weight-b
    :param variance_value_b:        variance-b
    :param weight_value_f:          weight-f
    :param variance_value_f:        variance-b
    :return: mixed_variance_value :   The mixed variance for the threshold
    '''
    mixed_variance_value=weight_value_b*variance_value_b + weight_value_f*variance_value_f
    return mixed_variance_value

def otsu(hist,bins):
    '''
    This otsu method returns the threshold value to binarize the data
    :param hist:              The histogram frequency values
    :param bins:              The speed values in each bin
    :param data:              The dataset provided
    :return:  min_threshold:            The computed threshold value
    :return:  class_variance_list:      List of mixed variances for all thresholds
    '''

    start=0   ## To specify the start position of data to be considered
    end=0     ## To specify the end position of data to be considered
    class_variance_list=[]          #List of mixed variances for all thresholds

    ## Computing threshold for all possible thresholds
    for i in range(0,len(bins)):
        threshold_value=i

        ## Calcualting the mean , weight the variance to compute the mixed variance for the threshold
        start=0
        end=threshold_value
        total_elements=calc_total_no_of_data_points(hist,start,end)
        weight_value_b=weight(hist,bins,start,end,total_elements)
        mean_value_b=mean(hist,bins,start,end,total_elements)
        variance_value_b=variance(hist,bins,start,end,mean_value_b,total_elements)

        ## Calcualting the mean , weight the variance to compute the mixed variance for the threshold
        start=threshold_value
        end=len(hist)
        total_elements=calc_total_no_of_data_points(hist,start,end)
        weight_value_f=weight(hist,bins,start,end,total_elements)
        mean_value_f=mean(hist,bins,start,end,total_elements)
        variance_value_f=variance(hist,bins,start,end,mean_value_f,total_elements)

        ## Computing the mixed variance as per the formula
        mixed_variance_value=mixed_variance(weight_value_b,variance_value_b,weight_value_f,variance_value_f)
        class_variance_list.append(mixed_variance_value)   ## Storing the mixed variance of each threshold

    min_class_variance=min(class_variance_list)
    min_threshold=class_variance_list.index(min_class_variance)
    print('Minium mixed variance', min_class_variance)
    return min_threshold,class_variance_list

def plotting_func(data):
    '''
    This plots the histogram of given data
    :param data: The data provided
    :return:
    '''
    plt.cla()
    ## To plot histogram of given data
    binwidth = 2
    plt.hist(data, bins=range(int(min(data)), int(max(data) + binwidth), binwidth))
    plt.xticks(range(int(min(data)), int(max(data) + binwidth), binwidth))
    plt.savefig('Histogram_q2.png')
    plt.show()

def calc_mean_median_mode(data):

    '''
    This function computes the mean , median and mode of given data
    :param data: THe data provided
    :return:
    '''

    ## Computing the mean
    mean = np.mean(data)
    print('Mean:', mean)

    ## Computing the median
    median = np.median(data)
    print('Median:', median)

    ## Computing the mode
    mode = stats.mode(data)
    print('Mode', mode[0])

    plotting_func(data) ## Callling the function to plot histogram

def main():
    '''
    The main function
    :return:
    '''

    ## Reading data from file
    data = np.loadtxt('Mystery_Data.csv', delimiter=',', skiprows=1)
    print("--------------Before deleting value 16-------")
    calc_mean_median_mode(data)                   ## Calling the function to compute the mean,median and mode

    ## To get the histogram and bins data for otsu method
    bin_size = 2
    hist, bins, patches = plt.hist(data, bins=range(int(min(data)), int(max(data) + bin_size), bin_size))

    threshold_pos, mixed_variance_list = otsu(hist, bins)  ##Calling the otsu method
    threshold_value = bins[threshold_pos]  ## The threshold speed
    print("Threshold value :", threshold_value, '(mph)')


    print("--------------After deleting value 16--------")
    ## To remove the value 16 from data
    index=np.where(data==16)
    new_data=np.delete(data,index)
    calc_mean_median_mode(new_data)              ## Calling the function to compute the mean,median and mode

    ## To get the histogram and bins data for otsu method after removing the value 16
    bin_size = 2
    hist, bins, patches = plt.hist(new_data, bins=range(int(min(data)), int(max(data) + bin_size), bin_size))

    threshold_pos ,mixed_variance_list= otsu(hist, bins)                 ##Calling the otsu method
    threshold_value = bins[threshold_pos]            ## The threshold speed
    print("Threshold value :",threshold_value, '(mph)')


if __name__ == '__main__':
    main()
