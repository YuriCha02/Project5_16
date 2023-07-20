import csv 
import torch
import numpy as np
import time
from torch import nn, optim
#기본적인 코드는 torch_AnnModel1.py와 동일
RND_MEAN, RND_STD = 0.0, 1.0
LEARNING_RATE = 0.001
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Model(nn.Module):
    def __init__(self, input_cnt, set_hidden, output_cnt):
        super(Model, self).__init__()
        self.layers = nn.ModuleList()
        
        last_cnt = input_cnt
        for hidden_cnt in set_hidden:
            self.layers.append(nn.Linear(last_cnt, hidden_cnt))
            self.layers.append(nn.ReLU())
            last_cnt = hidden_cnt
            
        self.layers.append(nn.Linear(last_cnt, output_cnt)) # 이진분류모델이므로 출력함수에 Sigmoid 추가
        self.layers.append(nn.Sigmoid())
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

#Baseline의 binarymain함수 구현
def binary_main(epoch_count:int=10, mb_size:int=10, report:int=1, train_ratio:float=0.8, set_hidden = (1, 2)):

    data, input_cnt, output_cnt = binary_load_dataset()
    model = Model(input_cnt, set_hidden, output_cnt).to(device)
    criterion = nn.BCELoss() # 이진분류에서 자주 사용하는 Binary Cross Entropy 적용
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

    train_and_test(model, criterion, optimizer, data, epoch_count, mb_size, report, train_ratio, output_cnt)


def binary_load_dataset(): 
    with open('./binary_classification_data.csv') as csvfile: 
        csvreader = csv.reader(csvfile) 
        next(csvreader) 
        rows = [] 
        for row in csvreader:
            rows.append(row) 

    input_cnt, output_cnt = 8, 1 #마찬가지로 피쳐수에따라 입력층 자동조절해주는 함수 생각해보기
    data = np.asarray(rows, dtype='float32') # AnnModel1.py와는 다른방식이 적용됨
    
    return data, input_cnt, output_cnt


def train_and_test(model, criterion, optimizer, data, epoch_count, mb_size, report, train_ratio, output_cnt):
    device = next(model.parameters()).device
    
    shuffle_map = np.arange(data.shape[0])
    np.random.shuffle(shuffle_map)

    mini_batch_step_count = int(data.shape[0] * train_ratio) // mb_size
    test_begin_index = mini_batch_step_count * mb_size
    
    test_data = data[shuffle_map[test_begin_index:]]

    for epoch in range(epoch_count):
        losses = []
        accs = []
        start_time = time.time()
        for nth in range(mini_batch_step_count):
            if nth == 0:
                np.random.shuffle(shuffle_map[:test_begin_index])
                
            train_data = data[shuffle_map[mb_size * nth : mb_size * (nth+1)]]
            train_x = torch.Tensor(train_data[:,:-output_cnt]).to(device)
            train_y = torch.Tensor(train_data[:,-output_cnt:]).to(device)

            model.train()
            optimizer.zero_grad()

            pred = model(train_x)
            loss = criterion(pred, train_y)

            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            accs.append((pred >= 0.5).eq(train_y).sum().item() / mb_size) 

        if report > 0 and (epoch+1) % report == 0:
            model.eval()
            with torch.no_grad():
                test_x = torch.Tensor(test_data[:,:-output_cnt]).to(device)
                test_y = torch.Tensor(test_data[:,-output_cnt:]).to(device)

                pred = model(test_x)
                loss = criterion(pred, test_y)
                accuracy = (pred >= 0.5).eq(test_y).sum().item() / test_y.shape[0]
                end_time = time.time() - start_time
                print(f"Epoch {epoch+1} : Train - Loss = {np.mean(losses):.3f}, Accuracy = {np.mean(accs):.3f} / Test - Loss = {loss.item():.3f}, Accuracy = {accuracy:.3f}, Time Duration = {end_time:.3f}sec")



if __name__ == "__main__":
    binary_main()
