import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.preprocessing import StandardScaler

class Model(nn.Module):
    def __init__(self, input_size, num_classes):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class MultiClassDataset(Dataset):
    def __init__(self, path):
        self.dataset = pd.read_csv(path).values
        self.x_data = self.dataset[:, :27]  # 독립변수
        self.y_data = self.dataset[:, 27:]  # 타겟데이터

        # 데이터 스케일링 (평균 0, 표준편차 1)
        scaler = StandardScaler()
        self.x_data = scaler.fit_transform(self.x_data)

    def __len__(self):
        return len(self.dataset)  # 총데이터수 반환
    
    def __getitem__(self, idx):
        x = self.x_data[idx]  # 전처리된 독립변수 사용
        y = self.y_data[idx]  # 타겟데이터
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

def train_and_test(model, criterion, optimizer, train_loader, test_loader, num_epochs):
    for epoch in range(num_epochs):
        start_time = time.time()
        
        correct_train = 0
        total_train = 0
        for i, (inputs, labels) in enumerate(train_loader):
            outputs = model(inputs)
            loss = criterion(outputs, torch.max(labels, 1)[1])
            
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == torch.max(labels, 1)[1]).sum().item()
            
            optimizer.zero_grad()
            loss.backward() 
            optimizer.step()
        
        train_acc = correct_train / total_train

        with torch.no_grad():
            correct_test = 0
            total_test = 0
            for inputs, labels in test_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total_test += labels.size(0)
                correct_test += (predicted == torch.max(labels, 1)[1]).sum().item()
            
        test_acc = correct_test / total_test

        end_time = time.time()
        epoch_time = end_time - start_time
        
        print(f"Epoch {epoch+1} ({epoch_time:.2f} sec): Train - Loss = {loss.item():.4f}, Accuracy = {train_acc:.3f} / Test - Accuracy = {test_acc:.3f}, Time Duration = {epoch_time:.3f}sec")

def multiple_main(epoch_count=20, batch_size=10, learning_rate=0.01, train_ratio=0.8):
    full_dataset = MultiClassDataset('./mulit_classification_data.csv')
    train_size = int(train_ratio * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    model = Model(input_size=27, num_classes=7)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_and_test(model, criterion, optimizer, train_loader, test_loader, epoch_count)

if __name__ == "__main__":
    multiple_main()