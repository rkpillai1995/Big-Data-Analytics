__author__ = 'Rajkumar Pillai'

import  matplotlib.pyplot as plt
import numpy as np

"""
file: q3.py
CSCI-720:  Big data analytics
Author: Rajkumar Lenin Pillai

Description: This program generates plot of mixed variance vs speed 
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
    print('Minium mixed variance',min_class_variance)
    return min_threshold,class_variance_list



def regularization(normfactor,alpha,data,threshold_value,hist,bins):
    '''
    This computes the regualrization that will be used to compute the cost fnction
    :param normfactor:  The normfactor used in formula
    :param alpha:       The alpha used in formula
    :param data:        The dataset provided
    :param threshold_value:  The threshold value from otsu method
    :param hist:              The histogram frequency values
    :param bins:              The speed values in each bin
    :return: regularization_value : The computed regularization value
    '''

    ## Computing regualrization as per the formula given in homework.pdf
    number_of_points_slow_group=0
    number_of_points_fast_group=0

    ## Calculating the no of points in slow drivers and fast drivers
    for i in range(0,len(data)):
        if data[i]<=threshold_value:
            number_of_points_slow_group+=1
        else:
            number_of_points_fast_group+=1

    ## Regularization formula
    regularization_value=(abs((number_of_points_slow_group-number_of_points_fast_group)) / normfactor ) * alpha
    print("Regualrization Value: ",regularization_value)
    return regularization_value

def main():
    '''
    The main function
    :return:
    '''


    ## Reading data from file
    data = np.loadtxt('DATA_v2185f_FOR_CLUSTERING_using_Otsu.csv', delimiter=",",skiprows=1)


    bin_size=2 # Declaring a binsize for quantizing vehicle speed

    ## To plot histogram of provided data
    hist,bins,patches=plt.hist(data, bins=range(int(min(data)), int(max(data)+bin_size) , bin_size))
    plt.xticks(range(int(min(data)), int(max(data)+bin_size),bin_size))
    plt.savefig('Histogram_q3.png')
    plt.show()


    threshold_pos,mixed_variance_list=otsu(hist,bins) ##calling otsu method
    threshold_value=bins[threshold_pos]                    # threshold_value is the threshold speed to seperate the clusters

    print("Threshold value :",threshold_value, '(mph)')

    normfactor=50   # The norm factor value
    alpha=1         # The alpha value


    regularization_value=regularization(normfactor,alpha,data,threshold_value,hist,bins) ## Calling regularization function

    ## Computing the cost function value
    cost=threshold_value+regularization_value
    print("Cost : ",cost)

    ## Plotting mixed variance vs speed
    plt.plot(mixed_variance_list,bins, marker='o', color='b')
    plt.axhline(cost, color='orange', ls='--')
    plt.yticks(bins)
    plt.yticks(list(plt.yticks()[0]) + [cost])
    plt.xticks(mixed_variance_list)
    ylabel='Speed (Mph)'
    xlabel='Mixed Variance'
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title("Plot of Mixed variance vs Speed")
    legend_list=['Speed (Mph)', 'Cost']
    plt.legend(legend_list)

    plt.savefig('Mixed_variance_vs_Speed.png')   ## Saving the figure in a file

    plt.show()                                   ## Displaying the image
if __name__ == '__main__':
    main()