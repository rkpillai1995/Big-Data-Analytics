__author__ = 'Rajkumar Pillai'

import  matplotlib.pyplot as plt
import numpy as np
import math
from numpy import *
"""
Description: This program implements the otsu method as well as a cost function and generates ROC curve and a Plot of cost_vs_threshold
"""

def weight(hist, bins, start, end, total_elements):
    '''
    This computes the weight
    :param hist:              The histogram frequency values
    :param bins:              The speed values in each bin
    :param start:             To specify the start position of data to be considered
    :param end:               To specify the end position of data to be considered
    :param total_elements:    Total no of data points considered
    :return w:              The computed weight value
    '''

    w = 0
    if total_elements == 0:
        return total_elements

    ## Computing the weight
    sum_of_weights = sum(hist)
    for i in range(start, end):
        w += hist[i]
    w = w / sum_of_weights
    return w


def mean(hist, bins, start, end, total_elements):
    '''
    This computes the mean
    :param hist:              The histogram frequency values
    :param bins:              The speed values in each bin
    :param start:             To specify the start position of data to be considered
    :param end:               To specify the end position of data to be considered
    :param total_elements:    Total no of data points considered
    :return mean_value:       The computed mean
    '''

    mean_variable = 0
    if total_elements == 0:
        return mean_variable

    ## Computing the mean
    for i in range(start, end):
        mean_variable += (bins[i] * hist[i])
    mean_value = mean_variable / total_elements
    return mean_value


def variance(hist, bins, start, end, mean_value, total_elements):
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
    if total_elements == 0:
        return variance_variable

    ## Computing the variance
    for i in range(start, end):
        variance_variable += (((bins[i] - mean_value) ** 2) * hist[i])
    variance_value = variance_variable / total_elements

    return variance_value


def calc_total_no_of_data_points(hist, start, end):
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


def mixed_variance(weight_value_b, variance_value_b, weight_value_f, variance_value_f):
    '''
    This computes the mixed variance of given thresdold and returns the computed mixed variance
    :param weight_value_b:          weight-b
    :param variance_value_b:        variance-b
    :param weight_value_f:          weight-f
    :param variance_value_f:        variance-b
    :return: mixed_variance_value :   The mixed variance for the threshold
    '''
    mixed_variance_value = weight_value_b * variance_value_b + weight_value_f * variance_value_f
    return mixed_variance_value


def otsu(hist, bins):
    '''
    This otsu method returns the threshold value to binarize the data
    :param hist:              The histogram frequency values
    :param bins:              The speed values in each bin
    :return:  min_threshold:            The computed threshold value
    :return:  class_variance_list:      List of mixed variances for all thresholds
    '''

    start = 0  ## To specify the start position of data to be considered
    end = 0  ## To specify the end position of data to be considered
    class_variance_list = []  # List of mixed variances for all thresholds

    ## Computing threshold for all possible thresholds
    for i in range(0, len(bins)):
        threshold_value = i

        ## Calcualting the mean , weight the variance to compute the mixed variance for the threshold
        start = 0
        end = threshold_value
        total_elements = calc_total_no_of_data_points(hist, start, end)
        weight_value_b = weight(hist, bins, start, end, total_elements)
        mean_value_b = mean(hist, bins, start, end, total_elements)
        variance_value_b = variance(hist, bins, start, end, mean_value_b, total_elements)

        ## Calcualting the mean , weight the variance to compute the mixed variance for the threshold
        start = threshold_value
        end = len(hist)
        total_elements = calc_total_no_of_data_points(hist, start, end)
        weight_value_f = weight(hist, bins, start, end, total_elements)
        mean_value_f = mean(hist, bins, start, end, total_elements)
        variance_value_f = variance(hist, bins, start, end, mean_value_f, total_elements)

        ## Computing the mixed variance as per the formula
        mixed_variance_value = mixed_variance(weight_value_b, variance_value_b, weight_value_f, variance_value_f)
        class_variance_list.append(mixed_variance_value)  ## Storing the mixed variance of each threshold

    min_class_variance = min(class_variance_list)
    min_threshold = class_variance_list.index(min_class_variance)
    print('Minium mixed variance',min_class_variance)

    return min_threshold, class_variance_list

def plotting_ROC(All_false_alram_rate, All_true_positive_rate, best_idx):
    '''
    This function plots the ROC curve
    :param All_false_alram_rate: False alarm rate computed for each threshold value
    :param All_true_positive_rate: True positive rate computed for each threshold value
    :param best_idx: The index of best threshold value computed in order to  plot on the ROC curve
    :return:
    '''
    plt.cla()
    plt.plot(All_false_alram_rate, All_true_positive_rate, marker='o', color='b')
    plt.plot(All_false_alram_rate[best_idx], All_true_positive_rate[best_idx], marker='o', markersize=8, color="red")
    random_guessing=[0.0,1.0]
    plt.plot(random_guessing,random_guessing, ls="--", c=".3")
    plt.grid(True)
    plt.tick_params(axis='y', which='both', labelleft='on', labelright='on')
    plt.tick_params(axis='x', which='both', labeltop='on', labelbottom='on')
    plt.xlabel("False Positive Rate (False Alarm Rate)",fontsize='x-large')
    plt.ylabel("True Positive Rate",fontsize='x-large')
    plt.title("ROC Curve by Threhsold",fontsize='x-large')
    legend_list=['ROC curve', 'Point on ROC with Lowest cost function','Random Guessing']
    plt.legend(legend_list,fontsize='x-large')
    plt.savefig('ROC_curve.png')

    plt.show()

def plot_cost_vs_threshold(Cost_of_every_threshold, All_Threshold_values):
    '''
    This function plots cost function as  a function of threshold used
    :param Cost_of_every_threshold: The cost function computed for every threshold
    :param All_Threshold_values: All the possible threshold values
    :return:
    '''
    plt.cla()
    plt.plot(All_Threshold_values, Cost_of_every_threshold, marker='o', color='b')
    plt.title("Plot of Cost function as a function of the threshold used",fontsize='x-large')
    plt.xlabel("Threshold",fontsize='x-large')
    plt.ylabel("Cost Value",fontsize='x-large')
    plt.savefig('cost_vs_threshold.png')
    plt.show()


def cost_func(hist, bins,speed,aggressive):
    '''
    This function computes cost as no of false positives + no of true positives
    :param hist:              The histogram frequency values
    :param bins:              The speed values in each bin
    :param speed:             The speed values of every driver which is rounded to nearest 0.5 mph and sorted in ascending order
    :param aggressive:        The aggressiveness of drivers which are provided
    :return:
    '''


    best_cost_value=math.inf                     ##Initializing the cost function value
    idx=0                                        ## Initializing the index variable for the cost computed


    ### Computing total no of aggressive and non-aggressive drivers
    total_no_of_aggressive = 0
    total_no_of_non_aggressive = 0
    for k in range(0, len(aggressive)):
        if aggressive[k] == 1.0:
            total_no_of_aggressive += 1
        else:
            total_no_of_non_aggressive += 1

    print("Total no of aggressive drivers: ", total_no_of_aggressive)
    print("Total no of non-aggressive drivers: ", total_no_of_non_aggressive)


    cost_of_all_thresholds=[]
    All_false_alarm_rate=[]
    All_true_positive_rate=[]

    ## Computing cost for all possible thresholds
    for i in range(0, len(bins)):
        threshold_value = bins[i]
        TN = FP = FN = TP = 0

        ## Computing True poitive, True negative, false positive and false negative
        for j in range(0,len(speed)) :
            if speed[j]<=threshold_value:
                if aggressive[j]==1.0:
                    FN+=1
                else:
                    TN+=1
            else:
                if aggressive[j] == 1.0:
                    TP+=1
                else:
                    FP+=1

        cost_value=(FN)+(FP)                   ## Computing the cost of the threshold
        cost_of_all_thresholds.append(cost_value)
        if cost_value<=best_cost_value:
            best_cost_value=cost_value
            best_threshold=threshold_value
            best_idx=idx
        false_alarm_rate_value=FP/total_no_of_non_aggressive
        true_pos_rate_value=TP / total_no_of_aggressive

        All_false_alarm_rate.append([false_alarm_rate_value])
        All_true_positive_rate.append([true_pos_rate_value])

        idx+=1


    plotting_ROC(All_false_alarm_rate, All_true_positive_rate,best_idx)        ## CAlling the function to plot ROC curve


    print("Best threshold",best_threshold)

    ## Computing no of drivers that would be let through and non reckless drivers that would be pulled
    no_of_aggressive_drivers_let_through = 0
    no_of_non_reckless_drivers = 0
    for k in range(0, len(aggressive)):
      if speed[k]>best_threshold:
        if aggressive[k] == 1.0:
            no_of_aggressive_drivers_let_through += 1
      if speed[k]>best_threshold:
         if aggressive[k] == 0.0:
            no_of_non_reckless_drivers+=1


    print("No of aggressive let through:  ",no_of_aggressive_drivers_let_through)
    print("No of non-reckless pulled over:  ", no_of_non_reckless_drivers)

    plot_cost_vs_threshold(cost_of_all_thresholds,bins)                         ## calling the function to plot cost vs threshold






def main():
    '''
    The main function
    :return:
    '''

    ## Reading data from file
    data = np.loadtxt('DATA_v525_FOR_CLASSIFICATION_using_Threshold_REVISED.csv', delimiter=",",skiprows=1)
    speed=[]
    aggressive=[]
    for i in range (len(data)):
        speed.append(data[i][0])
        aggressive.append(data[i][1])
    xlabel = 'Speed(mph)'
    ylabel = 'Frequency'

    rounded_speed = []  ## List to store the speeds after rounding peration
    for i in range(0, len(speed)):
        rounded_speed.append(round(speed[i] / 0.5) * 0.5)


    ## To plot histogram of provided data
    bin_size = 0.5  # Declaring a binsize for quantizing vehicle speed
    hist, bins, patches = plt.hist(rounded_speed, bins=np.arange(int(min(rounded_speed)), int(max(rounded_speed) + bin_size), bin_size))
    plt.xticks(np.arange(int(min(rounded_speed)), int(max(rounded_speed) + bin_size), bin_size))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title('Histogram of quantized vehicle speeds')
    plt.savefig('Histogram_q1.png')
    plt.show()


    threshold_pos, mixed_variance_list = otsu(hist, bins)  ## Calling the otsu method
    threshold_value = bins[threshold_pos]  ## The threshold speed
    no_of_aggressive_drivers_let_through_otsu=0
    no_of_non_reckless_drivers_otsu=0

    ## Computing no of drivers that would be let through and non reckless drivers that would be pulled suing otsu method

    for k in range(0, len(aggressive)):
      if speed[k]>threshold_value:
        if aggressive[k] == 1.0:
            no_of_aggressive_drivers_let_through_otsu += 1
      if speed[k]>threshold_value:
         if aggressive[k] == 0.0:
             no_of_non_reckless_drivers_otsu+=1


    print("No of aggressive drivers let through using otsu:  ",no_of_aggressive_drivers_let_through_otsu)
    print("No of non-reckless drivers let through using otsu:  ", no_of_non_reckless_drivers_otsu)
    print('The speed threshold using otsu: ', threshold_value, ' (mph)')


    ## Sorting the rounded speed values as well as the corresponding aggresiveness attribute
    for i in range(0,len(rounded_speed)):
        for j in range(0,len(rounded_speed)-i-1):
            if rounded_speed[j] > rounded_speed[j + 1]:
                rounded_speed[j], rounded_speed[j + 1] = rounded_speed[j + 1], rounded_speed[j]
                aggressive[j],aggressive[j+1]=aggressive[j+1],aggressive[j]


    print("+++++++Using the cost function +++++")
    cost_func(hist, bins,rounded_speed,aggressive)  ## Calling the cost_func method


if __name__ == '__main__':
    main()