import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from joblib import dump, load
import json
torch.manual_seed(42)

class MDNNModel(nn.Module):
    def __init__(self, input_size, num_classes, num_domains):
        super(MDNNModel, self).__init__()
        self.num_domains = num_domains
        self.shared_fc1 = nn.Linear(input_size, 256)
        self.shared_fc2 = nn.Linear(256, 128)
        self.shared_fc3 = nn.Linear(128, num_classes)
        self.domain_fc1 = nn.ModuleList([nn.Linear(input_size, 256) for _ in range(num_domains)])
        self.domain_fc2 = nn.ModuleList([nn.Linear(256, 128) for _ in range(num_domains)])
        self.domain_fc3 = nn.ModuleList([nn.Linear(128, num_classes) for _ in range(num_domains)])

    def forward(self, x, domain_idx):
        if domain_idx is None:
            x = F.relu(self.shared_fc1(x))
            x = F.relu(self.shared_fc2(x))
            x = self.shared_fc3(x)
        else:
            x = F.relu(self.domain_fc1[domain_idx](x))
            x = F.relu(self.domain_fc2[domain_idx](x))
            x = self.domain_fc3[domain_idx](x)

        return F.log_softmax(x, dim=1)

class MultiDomainDataset(Dataset):
    def __init__(self, path):
        data = pd.read_csv(path)
        self.label_to_index = {}
        self.index_to_label = {}
        self.dataset, self.num_domains = self.combine_labels(data)

        self.smote = SMOTE(random_state=42) #SMOTE를 클래스 attribute에 지정함
        self.dataset, self.y_res = self.smote.fit_resample(self.dataset.iloc[:, :27], self.dataset.iloc[:, 27])
        self.dataset = pd.concat([self.dataset, pd.DataFrame(self.y_res)], axis=1)
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x = self.dataset.iloc[idx, :27]
        y = self.dataset.iloc[idx, 27]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(self.label_to_index[y], dtype=torch.long)
    
    def combine_labels(self, data):
        labels = ['Pastry', 'Z_Scratch', 'K_Scatch', 'Stains', 'Dirtiness', 'Bumps', 'Other_Faults']
        one_hot_indices = [27, 28, 29, 30, 31, 32, 33]

        new_data = []

        for index, row in data.iterrows():
            new_row = row.iloc[:27].tolist()

            label_value = ''
            for i, label in enumerate(labels):
                if row[one_hot_indices[i]] == 1:
                    label_value += label + ', '

            label_value = label_value[:-2] # strip trailing comma and space

            # add label to dictionary if not already present
            if label_value not in self.label_to_index:
                new_index = len(self.label_to_index)
                self.label_to_index[label_value] = new_index
                self.index_to_label[new_index] = label_value

            new_row.append(label_value)

            new_data.append(new_row)

        new_columns = list(data.columns[:27]) + ['Faults']
        num_domains = len(self.label_to_index)
        return pd.DataFrame(new_data, columns=new_columns), num_domains

class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_loss = None
        self.counter = 0
        self.early_stop = False
        self.best_accuracy = 0.0

    def __call__(self, loss, accuracy):
        if self.best_loss is None:
            self.best_loss = loss
            self.best_accuracy = accuracy
            return False
        elif loss >= self.best_loss + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return self.early_stop
        else:
            if accuracy > self.best_accuracy:
                self.best_loss = loss
                self.best_accuracy = accuracy
            self.counter = 0
            return False

def train_and_test_mdnn(model, criterion, optimizer, train_loader, test_loader, num_epochs, num_domains):
    early_stopping = EarlyStopping(patience=5, delta=0.0001)

    for epoch in range(num_epochs):
        start_time = time.time()

        # Training
        model.train()
        correct_train = 0
        total_train = 0
        total_loss_train = 0
        for i, (inputs, labels) in enumerate(train_loader):
            domain_idx = None if num_domains == 1 else int(labels.max())  # Extract domain index from label
            outputs = model(inputs, domain_idx)
            loss = criterion(outputs, labels)

            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss_train += loss.item()

        train_acc = correct_train / total_train
        avg_loss_train = total_loss_train / len(train_loader)

        # Testing
        model.eval()
        with torch.no_grad():
            correct_test = 0
            total_test = 0
            total_loss_test = 0
            all_predicted = []
            all_labels = []
            for inputs, labels in test_loader:
                domain_idx = None if num_domains == 1 else int(labels.max())
                outputs = model(inputs, domain_idx)
                _, predicted = torch.max(outputs.data, 1)
                all_predicted.extend(predicted.numpy())
                all_labels.extend(labels.numpy())
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()

                loss = criterion(outputs, labels)
                total_loss_test += loss.item()

        test_acc = correct_test / total_test
        avg_loss_test = total_loss_test / len(test_loader)

        end_time = time.time()
        epoch_time = end_time - start_time

        print(f"Epoch {epoch + 1:2d} ({epoch_time:.2f} sec): Train - Loss = {avg_loss_train:.3f}, Accuracy = {train_acc:.3f} / Test - Loss = {avg_loss_test:.3f}, Accuracy = {test_acc:.3f}, Time Duration = {epoch_time:.3f}sec")

        # Early Stopping 적용
        if early_stopping(avg_loss_test, test_acc):
            print("[[Early Stopping!]]")
            print(f"최종 Test Accuracy : {early_stopping.best_accuracy:.3f}")
            print(classification_report(all_labels, all_predicted))
            break

    return model


def multi_domain_main(epoch_count=100, batch_size=10, learning_rate=0.001, train_ratio=0.8):
    full_dataset = MultiDomainDataset('./mulit_classification_data.csv')
    train_size = int(train_ratio * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

    # Apply StandardScaler to the data in tensor format
    scaler = StandardScaler()

    train_features = torch.stack([item[0] for item in train_dataset])
    train_features = torch.tensor(scaler.fit_transform(train_features), dtype=torch.float32)

    test_features = torch.stack([item[0] for item in test_dataset])
    test_features = torch.tensor(scaler.transform(test_features), dtype=torch.float32)

    train_labels = torch.tensor([item[1] for item in train_dataset])
    test_labels = torch.tensor([item[1] for item in test_dataset])

    sm = SMOTE(random_state=42)
    X_train, y_train = sm.fit_resample(train_features.numpy(), train_labels.numpy()) 
    X_train, y_train = torch.tensor(X_train), torch.tensor(y_train)

    X_test = test_features
    y_test = test_labels

    train_dataset = [(X_train[i], y_train[i]) for i in range(len(X_train))]
    test_dataset = [(X_test[i], y_test[i]) for i in range(len(X_test))]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    num_classes = len(set(full_dataset.dataset['Faults']))
    num_domains = len(set(full_dataset.dataset['Faults']))

    print('num_classes:', num_classes)
    print('num_domains:', num_domains)

    model = MDNNModel(input_size=27, num_classes=num_classes, num_domains=num_domains) 
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    trained_model = train_and_test_mdnn(model, criterion, optimizer, train_loader, test_loader, epoch_count, num_domains)

    with open('index_to_label.json', 'w') as f: 
        json.dump(full_dataset.index_to_label, f)

    return trained_model, scaler, full_dataset.index_to_label

if __name__ == "__main__":
    trained_model, scaler, label_mapping = multi_domain_main()

"""
    torch.save(trained_model.state_dict(), 'MDNN_Mulit_Classification.pt')
    dump(scaler, "MDNN_Mulit_Scaler.joblib")
    with open('MDNN_Mulit_label.json', 'w') as f:
        json.dump(label_mapping, f)
"""