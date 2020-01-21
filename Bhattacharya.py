import csv
import numpy as np
import math
from collections import defaultdict

"""

Description: This program finds the attribute with minimum mixed entropy to  create the decision strump using Bhattacharyac co-efficient
              and generates the classifier program  to classify the test labels
"""
## USed to store decision tree
tree=defaultdict(list)

def addEdge(tree,u,v):
    tree[u].append(v)
def split_calc_P1234(attribute_value_one_C0,attribute_value_one_C1,attribute_value_zero_C0,attribute_value_zero_C1):
    '''
    This  function calculates the  entropy for each value of all  attributes
    :param attribute_value_one_C0: Variable to store the count when attribute value=1 and CookieIsChocKrinkle == 0
    :param attribute_value_one_C1: Variable to store the count when attribute value=1 and CookieIsChocKrinkle == 1
    :param attribute_value_zero_C0: Variable to store the count when attribute value=0 and CookieIsChocKrinkle == 0
    :param attribute_value_zero_C1: Variable to store the count when attribute value=0 and CookieIsChocKrinkle == 1
    :return:P1_attribute_value_zero : Probablility when  attribute value is zero and CookieIsChocKrinkle = 0
           :P2_attribute_value_zero:  Probablility when  attribute value is zero and CookieIsChocKrinkle = 1
           :P3_attribute_value_one:   Probablility when  attribute value is one and CookieIsChocKrinkle = 0
           :P4_attribute_value_one:   Probablility when  attribute value is one and CookieIsChocKrinkle = 1
    '''

    ## To calculate total no of records where attribute value is zero
    attribute_value_zero_total = attribute_value_zero_C0 + attribute_value_zero_C1


    ## Calculation of  probabilities when value of attribute for data record is zero
    if attribute_value_zero_total == 0:
        P1_attribute_value_zero = 0
    else:
        P1_attribute_value_zero = attribute_value_zero_C0 / (attribute_value_zero_total)
    if attribute_value_zero_total == 0:
        P2_attribute_value_zero = 0
    else:
        P2_attribute_value_zero = attribute_value_zero_C1 / (attribute_value_zero_total)



    ## To calculate total no of records where attribute value is zero
    attribute_value_one_total = attribute_value_one_C0 + attribute_value_one_C1

    ## Calculation of  probabilities when value of attribute for data record is one
    if attribute_value_one_total == 0:
        P3_attribute_value_one = 0
    else:
        P3_attribute_value_one = attribute_value_one_C0 / (attribute_value_one_total)
    if attribute_value_one_total == 0:
        P4_attribute_value_one = 0
    else:
        P4_attribute_value_one = attribute_value_one_C1 / (attribute_value_one_total)

    return P1_attribute_value_zero,P2_attribute_value_zero,P3_attribute_value_one,P4_attribute_value_one,attribute_value_zero_total,attribute_value_one_total

def Bhattacharya_calc(p1,p2,p3,p4):
    '''
    This function calcualtes the Bhattacharya co-efficient of Attribute
    :param p1: Probablility when  attribute value is zero and CookieIsChocKrinkle = 0
    :param p2: Probablility when  attribute value is zero and CookieIsChocKrinkle = 1
    :param p3: Probablility when  attribute value is one and CookieIsChocKrinkle = 0
    :param p4: Probablility when  attribute value is one and CookieIsChocKrinkle = 1
    :return: Bhattacharya_coeffiecent: Bhattacharya co-efficient of Attribute
    '''

    x=math.sqrt((p1*p3))        ## Calcualating square root of P(A) * P(B) for  k=1
    y = math.sqrt((p2 * p4))    ## Calcualating square root of P(A) * P(B) for  k=2
    Bhattacharya_coeffiecent=x+y ## Addting the above two variables to get Bhattacharya co-efficient
    return Bhattacharya_coeffiecent

def decision_tree(Node,data,attribute_names_list,Target_Attribute,depth):

    '''
    The function generates decison tree
    :param Node: to store index of rows
    :param data: the dataser
    :param attribute_names_list: anmes if attribute
    :param Target_Attribute:
    :param depth:
    :return:
    '''


    # Attribute_index_list = []  ## List to store the index of every attribute
    Bhattacharya_coeffiecent_list = []  ## To store the Bhattacharya coefficient of every attribute
    #print(sorted(visited_attribute))
    #print(len(visited_attribute))

    best_attribute_to_split_on=None
    best_attribute_threshold=None
    Best_attribute_ID_to_split=None
    best_measure_of_goodness=math.inf
    target_attribute_list=[]

    #print("NON-Exit cond: ", visited_attribute)
    visited_childre_list=[]
    for x in tree.keys():
        for i in tree[x]:
            visited_childre_list.append(i)


    for j in range(0,len(attribute_names_list)):

        temp_list=data[:,j]
        target_attribute_list=data[:,attribute_names_list.index(Target_Attribute)]

        """MEasure of Badness"""
        tempname1=str(attribute_names_list[j]+":0")
        tempname2 = str(attribute_names_list[j] + ":1")
        #print(tempname1)
        #print(tempname2)

        #exit()
        if j != attribute_names_list.index(Target_Attribute) and str(attribute_names_list[j]) not in tree.keys() and tempname1 not in visited_childre_list and tempname2 not in visited_childre_list:
            #print("Theh attribute name: ",attribute_names_list[j])
            #print(tree.keys())
            attribute_value_zero_C0 = 0  ## Variable to store the count when attribute value=0 and CookieIsChocKrinkle == 0
            attribute_value_zero_C1 = 0  ## Variable to store the count when attribute value=0 and CookieIsChocKrinkle == 1

            attribute_value_one_C0 = 0  ## Variable to store the count when attribute value=1 and CookieIsChocKrinkle == 0
            attribute_value_one_C1 = 0  ## Variable to store the count when attribute value=1 and CookieIsChocKrinkle == 1

            '''
            Iterating through every value of attribute that is considered in order to calcuate 
            the count of records which are 1 or 0 when CookieIsChocKrinkle = 1 or 0
            '''
            for i in Node:
                #target_attribute_list[i]= data[i][attribute_names_list.index(Target_Attribute)]

                if temp_list[i] == 0:
                    if  target_attribute_list[i] == 0:
                        attribute_value_zero_C0 += 1
                    else:
                        attribute_value_zero_C1 += 1


                elif temp_list[i] == 1:
                    if  target_attribute_list[i] == 0:
                        attribute_value_one_C0 += 1
                    else:
                        attribute_value_one_C1 += 1


            p1, p2, p3, p4, Attribute_value_zero_total, Attribute_value_one_total = split_calc_P1234(attribute_value_one_C0,
                                                                                                 attribute_value_one_C1,
                                                                                                   attribute_value_zero_C0,
                                                                                                 attribute_value_zero_C1)

            ## To calcualte the Bhattacharya coefficient of eery attribute
            Bhattacharya_coeffiecent = Bhattacharya_calc(p1, p2, p3, p4)

            ## To store index of attribute considered
            #Attribute_index_list.append(j)

            ## To store the Bhattacharya coefficient coefficient of eery attribute
            Bhattacharya_coeffiecent_list.append(Bhattacharya_coeffiecent)

            if (Bhattacharya_coeffiecent<best_measure_of_goodness ):

                best_attribute_to_split_on=attribute_names_list[j]

                best_measure_of_goodness=Bhattacharya_coeffiecent
                Best_attribute_ID_to_split=j+1


    if depth==0 :
       return best_attribute_to_split_on

    best_attribute_to_split_on_list=data[:,attribute_names_list.index(best_attribute_to_split_on)]
    NodeA=[]
    NodeB=[]
    for i in range(0, len(best_attribute_to_split_on_list)):
  
        if best_attribute_to_split_on_list[i] == 0:
            NodeA.append(i)

        elif best_attribute_to_split_on_list[i] == 1:
            NodeB.append(i)


    if len(NodeA) == len(best_attribute_to_split_on_list):
        exit_cond = True

        addEdge(tree, best_attribute_to_split_on, "AllZeros:0")

    if len(NodeB) == len(best_attribute_to_split_on_list):
        exit_cond = True

        addEdge(tree, best_attribute_to_split_on, "ALLOnes:1")


    if len(NodeA)!=len(best_attribute_to_split_on_list):
        exit_cond=False
        # print("*********LEFT CALL*****************")

        best_measure_of_goodness=math.inf
        if str(best_attribute_to_split_on) not in tree.keys():
            tree[best_attribute_to_split_on] = []
        ##Recursive call
        next_best=decision_tree(NodeA, data, attribute_names_list, Target_Attribute, depth=depth-1)
        addEdge(tree, best_attribute_to_split_on , next_best+ ":0")

    if len(NodeB)!=len(best_attribute_to_split_on_list) :
        exit_cond=False
        # print("*********RIGHT CALL*****************")

        best_measure_of_goodness=math.inf
        if str(best_attribute_to_split_on) not in tree.keys():
             tree[best_attribute_to_split_on] = []
        ##Recursive call
        next_best=decision_tree(NodeB, data, attribute_names_list, Target_Attribute, depth=depth-1)
        addEdge(tree, best_attribute_to_split_on , next_best+ ":1")


    return best_attribute_to_split_on


def file_writing_recursion_left(f,parent,spacing,attribute_names_list,depth):
    '''
    USed to write the left subtree
    :param f:
    :param parent:
    :param spacing:
    :param attribute_names_list:
    :param depth:
    :return:
    '''
    if depth ==0:
        return f,spacing
    spacing = spacing+" "

    Index_of_parent_attribute = attribute_names_list.index(parent)

    child = tree[parent]
    x = child[0]
    attribute_name_x = x.split(":")[0]
    threshold_x = x.split(":")[1]
    Index_of_child_attribute_x = attribute_names_list.index(attribute_name_x)
    f.write('\n')
    f.write(spacing + " " + " if data_record[i][" + str(Index_of_child_attribute_x) + "] ==" + str(threshold_x) + ":")
    f.write('\n')
    f.write('\n')
    f.write(spacing + "  " + " f.write(str( " + str(threshold_x) + "))")
    f.write('\n')
    f.write('\n')
    f,spacing = file_writing_recursion_left(f, attribute_name_x,  spacing, attribute_names_list,depth=depth-1)
    return f,spacing
	
def file_writing_recursion_right(f,parent,spacing,attribute_names_list,depth):
    '''
    Used to write the right subtree
    :param f:
    :param parent:
    :param spacing:
    :param attribute_names_list:
    :param depth:
    :return:
    '''
    if depth ==0:
        return f,spacing
    spacing = spacing+" "

    Index_of_parent_attribute = attribute_names_list.index(parent)
  
    child = tree[parent]
    x = child[1]
    attribute_name_x = x.split(":")[0]
    threshold_x = x.split(":")[1]
    threshold_y=0
    Index_of_child_attribute_x = attribute_names_list.index(attribute_name_x)
    f,spacing = file_writing_recursion_right(f, attribute_name_x,  spacing, attribute_names_list,depth=depth-1)
    f.write('\n')
    f.write('\n')
    f.write(spacing +"else:")

    f.write('\n')
    f.write(spacing + " " + "if data_record[i][" + str(Index_of_child_attribute_x) + "] ==" + str(threshold_x) + ":")
    f.write('\n')
    f.write(spacing + " " + " f.write(str( " + str(threshold_x) + "))")

    f.write('\n')

    f.write(spacing + " " + "else"+ ":")
    f.write('\n')
    f.write(spacing + " " + " f.write(str( " + str(threshold_y) + "))")

    f.write('\n')

    f.write('\n')


    spacing=spacing[0:len(spacing)-2]
    return f,spacing

def emit_decision_tree(f,tree,attribute_names_list):
    '''
    This function generates the decision stump for classification in the classifier program and prints the decision stump in training program
    :param f: The file pointer
    :param best_attribute: The best attribute for splitting
    :param threshold:  The threshold as mentioned in homework pdf
    :return: f
    '''

    ## Writing the classsifier program ina seperate file
    f.write('\n')
    f.write('def my_classifier_function(data_record): ')
    f.write('\n')
    f.write('  f = open("HW_06_Pillai_Rajkumar_MyClassifications.csv", "w+")')
    f.write('\n')
    f.write('  for i in range(0, len(data_record)):')
    f.write('\n')
    f.write('\n')
    f.write("   f.write(" + repr('\n') + ")")
    f.write('\n')

    spacing=" "
    parent=next(iter(tree))  # outputs 'foo'


    Index_of_parent_attribute = attribute_names_list.index(parent)
    child=tree[parent]
    x = child[0]
    y = child[1]
    attribute_name_x = x.split(":")[0]
    attribute_name_y = y.split(":")[0]
    threshold_x = x.split(":")[1]
    threshold_y = y.split(":")[1]
    Index_of_child_attribute_x = attribute_names_list.index(attribute_name_x)
    Index_of_child_attribute_y = attribute_names_list.index(attribute_name_y)
    spacing = spacing + "  "
    f.write('\n')
    f.write(spacing + "if data_record[i][" + str(Index_of_parent_attribute) + "] ==" + str(threshold_x) + ":")
    f.write('\n')
    f.write(spacing + " " + " f.write(str( " + str(threshold_x) + "))")
    f.write('\n')

    f.write(spacing + " " + " if data_record[i][" + str(Index_of_child_attribute_x) + "] ==" + str(threshold_x) + ":")
    f.write('\n')
    f.write('\n')

    f.write(spacing + "  " + " f.write(str( " + str(threshold_x) + "))")

    f.write('\n')

    f.write('\n')


    depth=2
    f,spacing=file_writing_recursion_left(f,attribute_name_x,spacing,attribute_names_list,depth)


    f,spacing = file_writing_recursion_right(f, attribute_name_y, spacing, attribute_names_list, depth)
    f.write('\n')
    f.write(spacing + "else:")

    f.write('\n')
    f.write(spacing + " " + "if data_record[i][" + str(Index_of_child_attribute_y) + "] ==" + str(threshold_y) + ":")
    f.write('\n')

    f.write(spacing + " " + " f.write(str( " + str(threshold_y) + "))")
    f.write('\n')

    f.write('\n')
    spacing=spacing[0:len(spacing)-1]
    f.write(spacing + "  " + "else"+ ":")
    f.write('\n')

    f.write(spacing + "  " + " f.write(str( " + str(threshold_x) + "))")

    return f


def emit_trailer(f):
    '''
    This function writes the end of main function adn calls the main function
    :param f: The file pointer
    :return:
    '''
    f.write('\n')
    f.write("if __name__ == '__main__':")
    f.write('\n')
    f.write('  main()')
    f.write('\n')
    f.close()

def emit_classifier_call(f,target_attribute):
    '''
    This function  writes the main function of classifier program which calls the decision stump function which is named as my_classifier_function
    :param f: The file pointer
    :param target_attribute: The best attribute for splitting
    :return: f
    '''

    ## Writing the main function on classifier program
    f.write('\n')
    f.write('\n')
    f.write('def main():')
    f.write('\n')
    f.write("  data_record = np.loadtxt('Data_Population_Survey_as_Binary_VALIDATION_DATA_v800.csv', delimiter=',', skiprows=1)")
    f.write('\n')

    f.write("  data_record = np.delete(data_record, [125, 132, 133, 14, 74, 126, 140, 141,77,78,79,80,81,82,83,84,85,86], 1)")
    f.write('\n')
    f.write('  my_classifier_function(data_record)')
    f.write('\n')
    f.write('  correct = 0')
    f.write('\n')
    f.write('  incorrect = 0')
    f.write('\n')
    f.write("  classify_data = np.loadtxt('HW_06_Pillai_Rajkumar_MyClassifications.csv', delimiter=',') ")
    #f.write('\n')
    #f.write("  classify_data = np.delete(classify_data, [125, 132, 133, 14, 74, 126, 140, 141], 1)")
    f.write('\n')


    f.write('\n')
    f.write('\n')

    ## Writing the function for calcualtion of accuracy
    f.write('  for i in range(0, len(data_record)):')
    f.write('\n')
    f.write('      if data_record[i]['+str(target_attribute)+'] == 0:')
    f.write('\n')
    f.write('            if classify_data[i] == 0:')
    f.write('\n')
    f.write('                  correct += 1')
    f.write('\n')
    f.write('            else:')
    f.write('\n')
    f.write('                  incorrect += 1')
    f.write('\n')
    f.write('      elif data_record[i]['+str(target_attribute)+'] == 1:')
    f.write('\n')
    f.write('            if classify_data[i] == 1:')
    f.write('\n')
    f.write('                   correct += 1')
    f.write('\n')
    f.write('            else:')
    f.write('\n')
    f.write('                   incorrect += 1')
    f.write('\n')
    f.write('  print("Correct: ", correct)')
    f.write('\n')
    f.write('  print("Incorrect: ", incorrect)')
    f.write('\n')
    f.write('  print("Accuracy: ", correct / (correct + incorrect) * 100, "%")')
    f.write('\n')
    return f

def emit_header():
    '''
    This function creates the file pointer to create a new python file for classifier program and also writes the import statements
    :return: f :  The file pointer
    '''
    f = open("Bhattacharya_Classifier.py", "w+")
    f.write('import csv')
    f.write('\n')
    f.write('import numpy as np')
    f.write('\n')
    return f


def main():

    '''
    THe main function which calls the emit functions to create the classifier program
    :return:
    '''

    ## To read the taining data and skpping th efirst row which a=contains the attribute names
    data = np.loadtxt('Data_Population_Survey_as_Binary_v800.csv', delimiter=',', skiprows=1)

    Target_Attribute='CookieIsCrumpets'           ## DEclaring the target attribute

    ### To get the names of attributes  in the training data and storing them in a list
    attribute_names_list=[]
    with open('Data_Population_Survey_as_Binary_v800_modified.csv', 'r') as csvFile:
        reader = csv.reader(csvFile)
        for row in reader:
            attribute_names_list=row
            break

    Target_attribute_index = attribute_names_list.index(Target_Attribute)

    ## Calling decision tree
    Node = []
    Node=[i for i in range(0,len(data[:,len(attribute_names_list)-1]))]
	data=np.delete(data, [125, 132, 133,14,74,126,140,141,77,78,79,80,81,82,83,84,85,86], 1)
    depth=3
    remove_list=(125, 132, 133,14,74,126,140,141,77,78,79,80,81,82,83,84,85,86)
    remove_list=sorted(remove_list)
    count=0
    for i  in remove_list:
        del attribute_names_list[i-count]
        count=count+1
    
    decision_tree(Node,data, attribute_names_list, Target_Attribute,depth)
    print("Final decision tree")
    print(tree)
    f = emit_header()
    f = emit_decision_tree(f,tree,attribute_names_list)
    f = emit_classifier_call(f, Target_attribute_index)
    emit_trailer(f)


if __name__ == '__main__':
    main()