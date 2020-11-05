__author__ = 'Rajkumar Pillai'

import  matplotlib.pyplot as plt
import numpy as np


"""

CSCI-720:  Big data analytics
Author: Rajkumar Lenin Pillai

Description: This program generates plot of time (in minutes) as a function of speed (in miles per hour).
"""

def time_calc(distance,velocity):
    '''
    This is used to compute the time for each velocity ranging from 1 to 80 mph
    :param distance: 20 miles
    :param velocity: The array used to store speed of car
    :return: time: The time required to travel 20 miles at each velocity value
    '''
    time=np.zeros(len(velocity))
    for i in range(len(velocity)):
        time[i]=(distance/(velocity[i]))*60
    return time

def main():

    '''
    The main function
    :return:
    '''
    xlabel='Velocity '+'( miles per hour )'
    ylabel = 'Time' + ' (minutes)'
    distance= 20
    velocity=np.arange(0,80+1,1)


    time=time_calc(distance,velocity)  # Calling  the time_calc function to compute the time


    ## Plotting time v/s velocity
    plt.figure(figsize=(60, 60))
    plt.title('Plot of time (in minutes) as a function of speed (in miles per hour)', fontweight="bold", fontsize='20')
    plt.plot(velocity,time,marker='o',color='b')
    plt.xlabel(xlabel, fontsize=18)
    plt.ylabel(ylabel, fontsize=18)
    plt.show()


if __name__ == '__main__':
    main()
