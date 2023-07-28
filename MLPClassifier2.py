import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import *
from sklearn.decomposition import PCA
import numpy as np
import joblib

def load_data():
    mc_data = pd.read_csv('./mulit_classification_data.csv')
    mc_data.drop([391, 393], inplace=True)
    combined_data = combine_labels(mc_data)
    combined_data['Width'] = combined_data['X_Maximum'] - combined_data['X_Minimum']
    combined_data['Length'] = combined_data['Y_Maximum'] - combined_data['Y_Minimum']
    combined_data['Total_Perimeter'] = combined_data['X_Perimeter'] * combined_data['Y_Perimeter']

    return combined_data
    
def combine_labels(data):
    labels = ['Z_Scratch', 'K_Scatch', 'Stains', 'Dirtiness', 'Other_Faults'] #합쳐야 하는 레벨들
    one_hot_indices = [28, 29, 30, 31, 33] #각 레벨의 열 위치

    new_data = []

    for index, row in data.iterrows():
        new_row = row.iloc[:27].to_list() #기존데이터의 독립변수들을 새로운 행에 담음

        label_value = ''
        for i, label in enumerate(labels):
            if row[one_hot_indices[i]] == 1: #각 레벨의 열 위치가 1있는지 확인하고,
                label_value += label + ', ' #있으면, 레벨이름을 새로운 행에 넣음

        new_row.append(label_value[:-2])

        new_data.append(new_row) #새로운 행에 새로운 데이터의 담음

    new_columns = list(data.columns[:27]) + ['Faults'] #기존 열 이름을 유지시키고, 새로운 y이 열 이름를 Faults로 바꿈.
    return pd.DataFrame(new_data, columns=new_columns) #pandas에 쓸수 있게 새로운 데이터를 데이터 프레임화 시킴.

def split_standardize_pca(combined_data):
    X = combined_data.drop(['Faults'], axis = 1)
    y = combined_data['Faults']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3, stratify = y, random_state = 42)

    mlp_ss = StandardScaler()
    mlp_pca = PCA(n_components=19)
    
    X_train_ss = mlp_ss.fit_transform(X_train)
    X_train_pca = mlp_pca.fit_transform(X_train_ss)

    X_test_ss = mlp_ss.transform(X_test)
    X_test_pca = mlp_pca.transform(X_test_ss)
    
    return X_train_pca, X_test_pca, y_train, y_test, mlp_ss, mlp_pca

def pca(data):
    
    return data

def upsampling(X_train_ss, y_train):
    X_train_sampled, y_train_sampled = SMOTE(random_state=42).fit_resample(
    X_train_ss, y_train)

    return X_train_sampled, y_train_sampled

clf = MLPClassifier(
    hidden_layer_sizes=(100, 50),
    batch_size = 1,
    max_iter=5000,
    alpha=0.001,
    solver='adam',
    activation='relu',
    learning_rate_init = 0.001,
    momentum = 0.8,
    tol = 0.000001,
    early_stopping = True,
    validation_fraction = 0.1,
    random_state = 42,
    verbose=True
    )

def load_data():
    mc_data = pd.read_csv('./mulit_classification_data.csv')
    mc_data.drop([391, 393], inplace=True)
    combined_data = combine_labels(mc_data)
    combined_data['Width'] = combined_data['X_Maximum'] - combined_data['X_Minimum']
    combined_data['Length'] = combined_data['Y_Maximum'] - combined_data['Y_Minimum']
    combined_data['Total_Perimeter'] = combined_data['X_Perimeter'] * combined_data['Y_Perimeter']

    combined_data = combined_data[combined_data['Faults'].isin(['Bumps', 'Pastry'])]  # Filter only Bumps and Pastry

    return combined_data
    
def combine_labels(data):
    labels = ['Pastry', 'Bumps']  # the labels you are interested in
    one_hot_indices = [27, 32]  # the indices for Pastry and Bumps

    new_data = []

    for index, row in data.iterrows():
        new_row = row.iloc[:27].to_list()  # get independent variables

        label_value = None
        for i, label in enumerate(labels):
            if row[one_hot_indices[i]] == 1:  # check if this label is 1
                label_value = label  # assign the label to the row
                break

        if label_value:  # if the label was one of the labels of interest, add it to new_data
            new_row.append(label_value)
            new_data.append(new_row)

    new_columns = list(data.columns[:27]) + ['Faults']  # keep original column names and change y column name to Faults
    return pd.DataFrame(new_data, columns=new_columns)  # return a new dataframe

def split_standardize_pca(combined_data):
    global X_test_pca, y_test

    X = combined_data.drop(['Faults'], axis = 1)
    y = combined_data['Faults']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3, stratify = y, random_state = 42)

    mlp_ss = StandardScaler()
    mlp_pca = PCA(n_components=19)
    
    X_train_ss = mlp_ss.fit_transform(X_train)
    X_train_pca = mlp_pca.fit_transform(X_train_ss)

    X_test_ss = mlp_ss.transform(X_test)
    X_test_pca = mlp_pca.transform(X_test_ss)
    
    return X_train_pca, X_test_pca, y_train, y_test, mlp_ss, mlp_pca

def upsampling(X_train_ss, y_train):
    X_train_sampled, y_train_sampled = SMOTE(random_state=42).fit_resample(
    X_train_ss, y_train)

    return X_train_sampled, y_train_sampled

clf = MLPClassifier(
    hidden_layer_sizes=(100, 50),
    batch_size = 1,
    max_iter=5000,
    alpha=0.001,
    solver='adam',
    activation='relu',
    learning_rate_init = 0.001,
    momentum = 0.8,
    tol = 0.000001,
    early_stopping = True,
    validation_fraction = 0.1,
    random_state = 42,
    verbose=True
    )

def train_data():
    combined_data = load_data()
    X_train, X_test, y_train, y_test, mlp_ss, mlp_pca = split_standardize_pca(combined_data)
    X_train_sampled, y_train_sampled = upsampling(X_train, y_train)
    
    clf = MLPClassifier(
        hidden_layer_sizes=(100, 50),
        batch_size = 1,
        max_iter=5000,
        alpha=0.001,
        solver='adam',
        activation='relu',
        learning_rate_init = 0.001,
        momentum = 0.8,
        tol = 0.000001,
        early_stopping = True,
        validation_fraction = 0.1,
        random_state = 42,
        verbose=True
    )
    clf.fit(X_train_sampled, y_train_sampled)
    
    y_test_pred = clf.predict(X_test)
    accuracy = clf.score(X_test, y_test)
    print("Test Accuracy:", accuracy)
    print(classification_report(y_test, y_test_pred))
    return clf, mlp_ss, mlp_pca, X_test, y_test

def train_data_bumps_pastry():
    combined_data = load_data()
    combined_data = combined_data[combined_data['Faults'].isin(['Bumps', 'Pastry'])]
    X_train, X_test, y_train, y_test, mlp_ss, mlp_pca = split_standardize_pca(combined_data)
    X_train_sampled, y_train_sampled = upsampling(X_train, y_train)

    clf = MLPClassifier(
        hidden_layer_sizes=(100, 50),
        batch_size = 1,
        max_iter=5000,
        alpha=0.001,
        solver='adam',
        activation='relu',
        learning_rate_init = 0.001,
        momentum = 0.8,
        tol = 0.000001,
        early_stopping = True,
        validation_fraction = 0.1,
        random_state = 42,
        verbose=True
    )
    clf.fit(X_train_sampled, y_train_sampled)

    y_test_pred = clf.predict(X_test)
    accuracy = clf.score(X_test, y_test)
    print("Test Accuracy:", accuracy)
    print(classification_report(y_test, y_test_pred))
    return clf, mlp_ss, mlp_pca, X_test, y_test

def load_test_data():
    combined_data = load_data()
    X = combined_data.drop(['Faults'], axis = 1)
    y = combined_data['Faults']
    return X, y

def main():
    clf_all, mlp_ss_all, mlp_pca_all = train_data()
    clf_bp, mlp_ss_bp, mlp_pca_bp = train_data_bumps_pastry()

    test_data, y_test = load_test_data()  
    test_data_ss_all = mlp_ss_all.transform(test_data)  
    test_data_pca_all = mlp_pca_all.transform(test_data_ss_all)  
    test_data_ss_bp = mlp_ss_bp.transform(test_data)  
    test_data_pca_bp = mlp_pca_bp.transform(test_data_ss_bp)  

    prediction_all = clf_all.predict(test_data_pca_all)
    prediction_bp = clf_bp.predict(test_data_pca_bp)

    final_prediction = np.where((prediction_bp == 'Bumps') | (prediction_bp == 'Pastry'), prediction_bp, prediction_all)

    print(classification_report(y_test, final_prediction))  

if __name__ == "__main__":
    main()
        
        

