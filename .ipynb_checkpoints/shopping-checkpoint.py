from collections import Counter
import csv
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly as py

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4

def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE, shuffle=True,stratify=labels
    )

    print("Shape of x_train :", X_train.shape)
    print("Shape of y_train :", y_train.shape)
    print("Shape of x_test :", X_test.shape)
    print("Shape of y_test :", y_test.shape)
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    
    print(y_train[0])
    print(y_train)
    
    print(X_test.shape)
    # Train model and make predictions
    model = train_model(X_train, y_train)
    print("model: ",model)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):

    dataset = pd.read_csv(filename)
  
    #The names of month are no standard abbreviation so instead of calender library we used custom function to standard it.
    months = {'Jan': 0, 'Feb': 1, 'Mar': 2, 'Apr': 3, 'May': 4, 'June': 5, 'Jul': 6, 'Aug': 7,'Sep': 8, 'Oct': 9, 'Nov': 10, 'Dec': 11}
    dataset['Month'] = dataset['Month'].map(months).astype(int)
    dataset['VisitorType'] = dataset['VisitorType'].apply(lambda x: 1 if x=='Returning_Visitor' else 0).astype(int)
    dataset['Weekend'] = dataset['Weekend'].apply(lambda x: 1 if x == True else 0).astype(int)
    dataset['Revenue'] = dataset['Revenue'].apply(lambda x: 1 if x == True else 0).astype(int)
    
    #get_statistics(dataset)
    
    # getting dependent and independent variables
    x = dataset
    # removing the target column revenue from x
    x = x.drop(['Revenue'], axis = 1)

    #LogScale:
    x=np.log1p(x)

    y = dataset['Revenue']

    # checking the shapes
    print("Shape of x:", x.shape)
    print("Shape of y:", y.shape)
    
    return x,y
    raise NotImplementedError


def train_model(evidence, labels):
    evidence = np.array(evidence)
    labels = np.array(labels)
    
    #clf = KNeighborsClassifier() 
    #clf.fit(evidence,labels)
    clf = KNN(k=5)
    clf.fit(evidence, labels)
    return clf

    
    raise NotImplementedError


def evaluate(labels, predictions):
    acc = np.sum(predictions == labels) / len(labels)
    print('Accuracy: %.3f' %acc)
    
    raise NotImplementedError


def get_statistics(dataset):
    
    print(dataset.head())
    print(dataset.describe())
    print(dataset.info())
    print(dataset.isna().sum())

    pairplot = sns.pairplot(dataset, vars = ['Administrative', 'Administrative_Duration', 'Informational',
                                             'Informational_Duration', 'ProductRelated', 'ProductRelated_Duration',
                                             'BounceRates', 'ExitRates', 'PageValues', 'SpecialDay'],hue = 'Revenue')
    fig1 = pairplot.get_figure()
    fig1.savefig("snspairplot.png") 
    
    heatmap = sns.heatmap(dataset.iloc[:,0:-1].corr('pearson'), ax=ax, annot=True)
    fig1 = heatmap.get_figure()
    fig1.savefig("heatmap.png")



def euclidean_distance(x1, x2):
    distance = np.linalg.norm(x1 - x2)#np.sqrt(np.sum((x1-x2)**2))
    return distance

class KNN:
    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        print("selfytrain-----",self.y_train.shape)

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return predictions

    def _predict(self, x):
        # compute the distance
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
    
        # get the closest k
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i].item() for i in k_indices]

        # majority voye
        most_common = Counter(k_nearest_labels).most_common()
        return most_common[0][0]    
    
    

if __name__ == "__main__":
    main()
