import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import *

def load_data():
    mc_data = pd.read_csv('./mulit_classification_data.csv')
    faluts = (mc_data['Pastry'] + mc_data['Z_Scratch'] + mc_data['K_Scatch'] + mc_data['Stains'] 
              + mc_data['Dirtiness'] + mc_data['Bumps'] + mc_data['Other_Faults'])
    combined_data = combine_labels(mc_data)
    return combined_data
    
def combine_labels(data):
    labels = ['Pastry', 'Z_Scratch', 'K_Scatch', 'Stains', 'Dirtiness', 'Bumps', 'Other_Faults'] #합쳐야 하는 레벨들
    one_hot_indices = [27, 28, 29, 30, 31, 32, 33] #각 레벨의 열 위치

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

def split_standardize(combined_data):
    X = combined_data.drop('Faults', axis = 1)
    y = combined_data['Faults']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2, stratify = y, random_state = 42)

    ss = StandardScaler()

    X_train_ss = ss.fit_transform(X_train)
    X_test_ss = ss.transform(X_test)

    return X_train_ss, X_test_ss, y_train, y_test

def upsampling(X_train_ss, y_train):
    X_train_sampled, y_train_sampled = SMOTE(random_state=42).fit_resample(
    X_train_ss, y_train)

    return X_train_sampled, y_train_sampled

clf = MLPClassifier(
    hidden_layer_sizes=(50, 50),
    batch_size = 1,
    max_iter=50,
    alpha=0.0001,
    solver='adam',
    activation='relu',
    learning_rate_init = 0.001,
    momentum = 0.9,
    tol = 0.0001,
    early_stopping = True,
    validation_fraction = 0.1,
    random_state = 42,
    verbose=True
    )

def train_data():
    combined_data = load_data()
    X_train, X_test, y_train, y_test = split_standardize(combined_data)
    X_train_sampled, y_train_sampled = upsampling(X_train, y_train)
    clf.fit(X_train_sampled, y_train_sampled)
    y_test_pred = clf.predict(X_test)
    accuracy = clf.score(X_test, y_test)
    print("Test Accuracy:", accuracy)
    print(classification_report(y_test, y_test_pred))

if __name__ == "__main__":
    train_data()


