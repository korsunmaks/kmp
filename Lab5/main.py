from math import sqrt
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
import numpy as np


def task1(data_set):
    print("Statistical information:")
    print(data_set.describe(), "\n")
    df = pd.DataFrame(data_set)

    # Display information about features
    features = df.drop('class', axis=1)
    print("Number of features:", len(features.columns))
    print("Features:", list(features.columns), "\n")
    
    #Check missimg values
    if df.isnull().any().any():
        print("Missing values found!\n")
    else:
        print("No missing values found.\n")

    # Check for duplicate records
    if df.duplicated().any():
        print("Duplicate records found!\n")
    else:
        print("No duplicate records found.\n")

    # Display class distribution
    print("Class distribution by ", data_set.groupby('class').size(), "\n")


def task2(data_set):
    color_wheel = {
        "1": "red",
        "2": "blue",
        "3": "green",
    }
    colors = data_set["class"].map(lambda x: color_wheel.get(x))
    scatter_matrix(data_set, c=colors)
    plt.show()


def get_train_and_test_data(data_set):
    seed_size = data_set.iloc[:, :-1].values
    seed_class = data_set.iloc[:, 7].values
    train_size, test_size, train_class, test_class = train_test_split(seed_size,seed_class, test_size=0.2)
    return train_size, test_size, train_class, test_class

def scale_size(train_size, test_size):
    scaler = StandardScaler()
    scaler.fit(train_size)
    train_size = scaler.transform(train_size)
    test_size = scaler.transform(test_size)
    return train_size, test_size

def task3_to_5(data_set):
    train_size, test_size, train_class, test_class = get_train_and_test_data(data_set)
    classifier = LinearDiscriminantAnalysis()
    classifier.fit(train_size, train_class)
    test_predict = classifier.predict(test_size)
    score = classifier.score(test_size, test_class)
    # Task 3
    print("Correct predictions proportion = ", score, "\n") 
    # Task 4
    print(classification_report(test_class, test_predict)) 
    # Task 5    
    cm = confusion_matrix(test_class, test_predict) 
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classifier.classes_)
    disp.plot()
    plt.show()
    
def task6(data_set):
    train_size, test_size, train_class, test_class = get_train_and_test_data(data_set) #Get data
    train_size, test_size = scale_size(train_size, test_size)  #Scaling
    classifier = LinearDiscriminantAnalysis()
    classifier.fit(train_size, train_class)
    test_predict = classifier.predict(test_size)
    score = classifier.score(test_size, test_class)
    print("Correct predictions proportion = ", score, "\n")
    print(classification_report(test_class, test_predict))
    cm = confusion_matrix(test_class, test_predict)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classifier.classes_)
    disp.plot()
    plt.show()


def task7(data_set):
    train_size, test_size, train_class, test_class = get_train_and_test_data(data_set)
    train_size, test_size = scale_size(train_size, test_size)
    classifier =LinearDiscriminantAnalysis(   
        solver='lsqr', 
        shrinkage='auto', 
        priors=None, 
        n_components=None,  
        store_covariance=True, 
        tol=0.0001 
    )
    classifier.fit(train_size, train_class)
    score_forest = classifier.score(test_size, test_class)
    print("Correct predictions proportion = ", score_forest, "\n")


def get_type_indexes(test_class, class_name):
    indexes = np.where(test_class == class_name)
    indexes = indexes[0]
    return indexes[:3]

def get_objects(test_size, test_class):
    unique_classes = np.unique(test_class)
    indexes = []
    for i in range(len(unique_classes)):
        indexes.append(get_type_indexes(test_class, unique_classes[i]))
    elements = []
    for i in range(len(indexes)):
        elements.append(test_size[indexes[i]])
    return elements

def get_euclide_distance(obj1,obj2):
    return np.sqrt(np.sum((obj1-obj2)**2))

def get_nearest_neighbor(element, train_size):
    distances = [get_euclide_distance(element, set) for set in train_size]
    min_index = np.argmin(distances)
    return train_size[min_index]

def is_all_true(arr):
    flag = True
    for i in range(len(arr)):
        if arr[i] == False:
            flag = False
    return flag

def get_type(size, types, element):
    for i in range(len(size)):
        is_element = element == size[i]
        if is_all_true(is_element):
            return types[i]


def task9(data_set):
    train_size, test_size, train_class, test_class = get_train_and_test_data(data_set)
    elements = get_objects(test_size, test_class)

    for element_group in elements:
        for element in element_group:
            closest_element = get_nearest_neighbor(element, train_size)
            print(f"The nearest neighbor to an element {element} is {closest_element}")

            type1 = get_type(test_size, test_class, element)
            type2 = get_type(train_size, train_class, closest_element)
            print(f"Types: {type1} <--> {type2}")



def main():
    data_set = pd.read_csv("Seeds.csv")
    data_set["class"] = [str(i) for i in data_set["class"]]
    # task1(data_set)
    # task2(data_set)
    # task3_to_5(data_set)
    # task6(data_set)
    # task7(data_set)
    task9(data_set)

main()


