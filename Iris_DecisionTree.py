# Grant Gilchrist
# 05/29/2022 
# CS379-2203A-01 - Machine Learning 
# This is a supervised machine learning algorithm that utilizes data from a popular dataset 
# known as the iris dataset. The iris dataset is used to predict the species of iris flower 
# based on sepal length, sepal width, petal length, and pedal width. I will be using a decision 
# tree alogrithm to predict the species of iris based on the data provided, then comparing it to 
# the actual species recorded in the dataset.

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split  
from sklearn.tree import DecisionTreeClassifier  
import seaborn as sns

#---------------------------------------------------------------
# Load CSV File
#---------------------------------------------------------------
# This function loads a csv file from an absolute path

def loadCSV():
    try:
        iris_df = pd.read_csv("D:/Class/Machine Learning/Unit 2/IRIS.csv")
        if not iris_df.empty: 
            return iris_df
        else:
            print('Error: The dataset was empty')
    except:
        print('Error: Please input the correct absolute file path.')

#---------------------------------------------------------------
# Graph
#---------------------------------------------------------------
# This function is used to better understand our data

def graph(iris_df):
    
    iris_df.hist() # creates a histogram of the data
    plt.show() # used to show all figures
    
    # plots a bar graph of the amount flowers for each species of iris
    sns.countplot(x = iris_df['species']) 
  
#---------------------------------------------------------------
# Split the Data
#---------------------------------------------------------------    

def split(iris_df):
    
    # We will need to seperate the data from the label
    # By creating two variables we can seperate the data from the label
    x = iris_df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']] # data
    y = iris_df['species'] # label
    
    # splitting the data into training (20%) and test(80%) data
    x_train,x_test, y_train,y_test = train_test_split(x,y,test_size = 0.2) 
    
    classifier = train(x_train, y_train) # calls the train function sending it the data it split within the function arguments
    predict(classifier, x_test, y_test) # calls the predict function sending it the data it split within the function arguments
      
#---------------------------------------------------------------
# Train the Decision Tree
#---------------------------------------------------------------      

def train(x_train, y_train):
    # we are creating a DecisionTreeClassifier object from the sklearn.tree library
    # criterion: Is the function to measure the quality of a split. “entropy” is used for the information gain.
    # random_state: Is the seed used by the random number generator
    classifier = DecisionTreeClassifier(criterion='entropy', random_state=0) # we are creating a DecisionTreeClassifier object from the sklearn.tree library
    
    # This fits the training data to the DecisionTreeClassifier 
    classifier.fit(x_train, y_train) 
    return classifier
    
#---------------------------------------------------------------
# Train the Decision Tree
#---------------------------------------------------------------    

def predict(classifier, x_test, y_test):
    
    # This predicts the species (y_test) based on the test data provided
    y_pred = classifier.predict(x_test) 
    y_pred[0:100]
    
    # This prints the accuracy of our prediction compared to the actual label (y_test) data
    print(f'Accuracy: {classifier.score(x_test,y_test)*100:.2f}%')
    
#---------------------------------------------------------------
# Main
#---------------------------------------------------------------

def main():
    iris_df = loadCSV() # Loads the CSV as a dataframe into the iris_df variable
    graph(iris_df) # calls a function that graphs the data from the csv file
    split(iris_df) # calls a function that splits the data, trains, then predicts the species

main()
