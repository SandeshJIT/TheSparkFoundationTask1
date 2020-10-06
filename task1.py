#--- Program to predict the Students score based on the numbers of hours the student studies --#
#-- Spark Foundation Data Science and Business Analyst Internship Task 1 --#

#importing necessary packages for Linear Regression
import csv
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

#Creating an empty list to take read CSV file input
x = list()


#Reading and storing the values in CSV FILE
with open('task1.csv','r') as fp:
    read = csv.reader(fp)
    for r in read:
        x.append(r)
        
        
#Printing the list
print(x)


#Taking Hours in "a" and Score in "b" 
a =  np.array(x[1][0],dtype="float64")
b = np.array(x[1][1],dtype="float64")
for i in x[1:]:
    a =np.insert(a , 0 ,float(i[0]))
    b =np.insert(b , 0 ,float(i[1]))
    

#Making it 2 Dimenssional array
a=a.reshape(-1,1)
b=b.reshape(-1,1)


#Creating a linearRegression model
model = LinearRegression().fit(a,b)


#Printing the Coef of the model 
r_sq = model.score(a, b)
print(r_sq)


#Ploting the data in a graph and seeing the coef visualy
plt.scatter(a,b)
y_pred = model.predict(a)
print(y_pred)
plt.plot(a,y_pred,color="red")
plt.show()


#Preficting the Score when student studies for 9.25hrs every day
y_pred = model.predict([[9.25]])
print("\nIf the student studies for 9.25hrs/day this model predicts that he will score ",y_pred[0,0])
