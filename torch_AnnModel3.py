import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import time

class Model(nn.Module):
    def __init__(self, input_cnt, hidden_config, output_cnt):
        super(Model, self).__init__()
        self.hidden_layers = nn.ModuleList([])
        for i in range(len(hidden_config)):
            if i == 0:
                self.hidden_layers.append(nn.Linear(input_cnt, hidden_config[i])) #입력층과 은닉층 fully-connected
            else:
                self.hidden_layers.append(nn.Linear(hidden_config[i-1], hidden_config[i])) # 은닉층간의 fully-connected
        self.output_layer = nn.Linear(hidden_config[-1], output_cnt) # 은닉층과 출력층 fully-connected
        self.relu = nn.ReLU()
    
    def forward(self, x):
        for hidden_layer in self.hidden_layers:
            x = self.relu(hidden_layer(x))
        x = self.output_layer(x)
        return x

# 파이토치의 Dataset 클래스 상속받으면 간편하게 사용가능
class MultiClassDataset(Dataset):
    def __init__(self, path):
        self.dataset = pd.read_csv(path).values
        
    def __len__(self):
        return len(self.dataset) # 총데이터수 반환
    
    def __getitem__(self, idx):
        x = self.dataset[idx,:27] #독립변수
        y = self.dataset[idx,27:] #타겟데이터
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
        
        print(f"Epoch {epoch+1} ({epoch_time:.2f} sec): Train - Loss = {loss.item()}, Accuracy = {train_acc:.3f} / Test - Accuracy = {test_acc:.3f}, Time Duration = {epoch_time:.3f}sec")
 
def multiple_main(epoch_count = 10, batch_size = 10, learning_rate = 0.001, train_ratio = 0.6):

    full_dataset = MultiClassDataset('./mulit_classification_data.csv')
    train_size = int(train_ratio * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


    model = Model(27, [2,5], 7) #마찬가지로 은닉측을 2, 5보다 더 늘려서 테스트 해보기
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


    train_and_test(model, criterion, optimizer, train_loader, test_loader, epoch_count)

if __name__ == "__main__":
    multiple_main()
