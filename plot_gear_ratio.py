__author__ = 'Rajkumar Pillai'

import  matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches


"""
Description: This program generates plot of given gear ratios.
"""

def GearRatio_calc(No_of_Teeths_Front_Gear, No_of_Teeths_Back_Gear):
    '''
    This computes all possible combination of the gear ratios
    :param No_of_Teeths_Front_Gear:
    :param No_of_Teeths_Back_Gear:
    :return:Gear_Ratio: 2-d list of The No_of_Teeths_Front_Gear / The No_of_Teeths_Back_Gear
    :return:ListofAllGearRatioValues: A list used to store gear ratios to be used as y-axis labels
    '''

    ListofAllGearRatioValues=[]
    Gear_Ratio=np.zeros((len(No_of_Teeths_Back_Gear),len(No_of_Teeths_Front_Gear)))
    for i in range(len(Gear_Ratio)):
        for j in range (len(No_of_Teeths_Front_Gear)):
            Gear_Ratio[i][j]=(No_of_Teeths_Front_Gear[j] / No_of_Teeths_Back_Gear[i])
            ListofAllGearRatioValues.append(Gear_Ratio[i][j])
    return Gear_Ratio,ListofAllGearRatioValues

def plot_Gear_Ratios(Gear_Ratio, Backgear, xlabel, ylabel, ListofAllGearRatioValues):
        '''
        This function plots all possible combinations of the gear ratios
        :param Gear_Ratio:  2-d list of The No_of_Teeths_Front_Gear / The No_of_Teeths_Back_Gear
        :param Backgear: List of No_of_Teeths_Back_Gear
        :param xlabel: The values for x-axis
        :param ylabel: The values for y-axis
        :param ListofAllGearRatioValues: A list used to store gear ratios to be used as y-axis labels
        :return:
        '''

        ## Used to plot the horizontal lines for each gear ratio
        fig=plt.figure(figsize=(100, 100))
        plt.title('Plot of all possible gear ratios', fontweight="bold", fontsize='20')
        hax = fig.add_subplot(111)
        Backgear_Names=['a','b','c','d','e','f','g']

        #Plotting of graph with all possible combination of gear ratios
        for row in range (len(Gear_Ratio)):
            for j in range(len(Gear_Ratio[row])):
                if j==0:

                    # Drawing horizontal line for the gear ratio
                    hax.hlines(Gear_Ratio[row][j], [0], Backgear[row], lw=1)
                    plt.text(Backgear[row],Gear_Ratio[row][j],'A:'+Backgear_Names[row],fontsize=12,color='blue',fontweight="bold")

                if j==1:
                    # Drawing horizontal line for the gear ratio
                    hax.hlines(Gear_Ratio[row][j], [0], Backgear[row], lw=1)
                    plt.text(Backgear[row], Gear_Ratio[row][j], 'B:'+Backgear_Names[row],fontsize=12,color='red',fontweight="bold")

                if j==2:
                    # Drawing horizontal line for the gear ratio
                    hax.hlines(Gear_Ratio[row][j], [0], Backgear[row], lw=1)
                    plt.text(Backgear[row], Gear_Ratio[row][j], 'C:'+Backgear_Names[row],fontsize=12,color='green',fontweight="bold")


        plt.xlabel(xlabel,fontsize=18)
        plt.ylabel(ylabel,fontsize=18)
        plt.yticks(ListofAllGearRatioValues)
        plt.xticks(Backgear)

        ## Plotting the legend to understand color code for each front gear
        blue_color = mpatches.Patch(color='blue', label='Front Gear A / backgear')
        red_color = mpatches.Patch(color='red', label='Front Gear B / backgear')
        green_color=mpatches.Patch(color='green', label='Front Gear C / backgear')
        plt.legend(handles=[blue_color,red_color,green_color],fontsize = 'xx-large')
        plt.show()

def main():

    '''
    The main function
    :return:
    '''
    xlabel='No of teeth '+'on the backcog'
    ylabel='Gear ratios'+'(mph)'
    No_of_Teeths_Front_Gear=[73,51,31]
    No_of_Teeths_Back_Gear=[19,23,33,44,53,63,71]
    Frontgear= np.asarray(No_of_Teeths_Front_Gear)
    Backgear=np.asarray(No_of_Teeths_Back_Gear)


    # To compute the gear ratios
    Gear_Ratio,ListofAllGearRatioValues=GearRatio_calc(Frontgear,Backgear)

    # To plot the gear ratios
    plot_Gear_Ratios(Gear_Ratio,Backgear,xlabel,ylabel,ListofAllGearRatioValues)



if __name__ == '__main__':
    main()