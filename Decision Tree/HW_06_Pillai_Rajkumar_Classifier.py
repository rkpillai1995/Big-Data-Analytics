import csv
import numpy as np

def my_classifier_function(data_record): 
  f = open("HW_06_Pillai_Rajkumar_MyClassifications.csv", "w+")
  for i in range(0, len(data_record)):

   f.write('\n')

   if data_record[i][69] ==0:
     f.write(str( 0))
     if data_record[i][75] ==0:

      f.write(str( 0))


      if data_record[i][62] ==0:

       f.write(str( 0))


       if data_record[i][113] ==0:

        f.write(str( 0))



       else:
        if data_record[i][65] ==1:
         f.write(str( 1))
        else:
         f.write(str( 0))



     else:
      if data_record[i][64] ==1:
       f.write(str( 1))
      else:
       f.write(str( 0))


   else:
    if data_record[i][63] ==1:
     f.write(str( 1))

    else:
     f.write(str( 0))

def main():
  data_record = np.loadtxt('Data_Population_Survey_as_Binary_VALIDATION_DATA_v800.csv', delimiter=',', skiprows=1)
  data_record = np.delete(data_record, [125, 132, 133, 14, 74, 126, 140, 141,77,78,79,80,81,82,83,84,85,86], 1)
  my_classifier_function(data_record)
  correct = 0
  incorrect = 0
  classify_data = np.loadtxt('HW_06_Pillai_Rajkumar_MyClassifications.csv', delimiter=',') 


  for i in range(0, len(data_record)):
      if data_record[i][75] == 0:
            if classify_data[i] == 0:
                  correct += 1
            else:
                  incorrect += 1
      elif data_record[i][75] == 1:
            if classify_data[i] == 1:
                   correct += 1
            else:
                   incorrect += 1
  print("Correct: ", correct)
  print("Incorrect: ", incorrect)
  print("Accuracy: ", correct / (correct + incorrect) * 100, "%")

if __name__ == '__main__':
  main()
