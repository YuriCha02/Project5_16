import csv 
import torch
from torch import nn, optim
import time
import numpy as np
import csv 
import torch
from torch import nn, optim
import time
import numpy as np

# 정규분포를 갖는 난수값을 생성하기 위한 매개변수 정의
RND_MEAN, RND_STD = 0.0, 1.0
LEARNING_RATE = 0.001

# GPU사용가능시 cuda, 그외에 cpu 적용
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# PyTorch의 nn.Module을 상속받아 모델 생성
class Model(nn.Module):
    def __init__(self, input_cnt, set_hidden, output_cnt): # input_cnt = 입력층, set_hidden = 은닉층, output_cnt = 출력층
        super(Model, self).__init__()
        self.layers = nn.ModuleList() # 파이토치의 ModuleList 상속받아 파라미터 인식
        
        # 입력층 - 은닉층과의 연결, 은닉층끼리의 연결을 nn.Linear(fully-connected)를 통해 설정한 set_hidden 뉴런수를 따라 연결
        last_cnt = input_cnt
        for hidden_cnt in set_hidden:
            self.layers.append(nn.Linear(last_cnt, hidden_cnt))
            self.layers.append(nn.ReLU())
            last_cnt = hidden_cnt
            
        # 은닉층과 출력층과의 연결
        self.layers.append(nn.Linear(last_cnt, output_cnt))
    
    # 모델에 입력된 input값이 각 층을 통과하며 최종적으로 출력되는 함수 지정
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def main(epoch_count:int=10, mb_size:int=10, report:int=1, train_ratio:float=0.8, set_hidden = (1, 2)): 
    #Baseline에서는 set_hidden(1,2)을 ai_program에서 설정했지만 이를 하이퍼파라미터(arg)방식으로 사용자가 쉽게 설정하는 방법도 생각해볼 수 있음
    data, input_cnt, output_cnt = load_dataset() # 데이터와 입력층과 출력층을 지정함수 load_dataset에서 호출
    model = Model(input_cnt, set_hidden, output_cnt).to(device)
    criterion = nn.MSELoss()  # 손실 함수로 MSE 사용 -> 다른손실함수를 적용할 필요가 있을지?
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)  # SGD 경사하강법 적용

    train_and_test(model, criterion, optimizer, data, epoch_count, mb_size, report, train_ratio, output_cnt)

def load_dataset(): 
    with open('./Regression_data.csv') as csvfile: 
        csvreader = csv.reader(csvfile) 
        next(csvreader) 
        rows = [] 
        for row in csvreader:
            rows.append(row) 

    input_cnt, output_cnt = 10, 1 # 여기서 입력층의경우 데이터셋의 피쳐수에따라 결정되는데 사용자가 따로 지정하지 않고도 데이터셋에 맞게 동작하는 방법 생각해보기
    data = np.zeros([len(rows), input_cnt + output_cnt])
    
    # 전복의 성별 원핫인코딩 적용
    for n, row in enumerate(rows):
        if row[0] == 'M': 
            data[n, 0] = 1
        if row[0] == 'F':
            data[n, 1] = 1
        if row[0] == 'I': 
            data[n, 2] = 1
        data[n, 3:] = row[1:]
    
    return data, input_cnt, output_cnt


def train_and_test(model, criterion, optimizer, data, epoch_count, mb_size, report, train_ratio, output_cnt):
    shuffle_map = np.arange(data.shape[0])
    np.random.shuffle(shuffle_map)
    mini_batch_step_count = int(data.shape[0] * train_ratio) // mb_size
    
    #데이터 셔플
    test_x = torch.Tensor(data[shuffle_map[mini_batch_step_count:], :-output_cnt]).to(device)
    test_y = torch.Tensor(data[shuffle_map[mini_batch_step_count:], -output_cnt:]).to(device)

    for epoch in range(epoch_count):
        losses = []
        accs = []
        start_time = time.time()
        np.random.shuffle(shuffle_map[:mini_batch_step_count])
        for nth in range(mini_batch_step_count):
            mini_batch = np.take(data, shuffle_map[nth*mb_size:(nth+1)*mb_size], axis=0)
            x = torch.Tensor(mini_batch[:, :-output_cnt]).to(device)
            y = torch.Tensor(mini_batch[:, -output_cnt:]).to(device)

            # 경사 초기화, 순전파, 손실 계산, 역전파, 가중치 업데이트를 파이토치 내장함수로 간단히 구현
            optimizer.zero_grad() # 그래디언트 초기화
            y_hat = model(x)
            loss = criterion(y_hat, y)  # 손실 값 계산
            loss.backward() # 그래디언트 계산
            optimizer.step() # 모델의 파라미터 업데이트

            accuracy = eval_accuracy(y_hat, y)
            losses.append(loss.item())
            accs.append(accuracy.item())
        
        if report > 0 and (epoch+1) % report == 0:
            duration = time.time() - start_time
            test_y_hat = model(test_x)
            test_loss = criterion(test_y_hat, test_y)
            test_acc = eval_accuracy(test_y_hat, test_y)
            print("Epoch: {}, Avg Train Loss: {:.3f}, Train Accuracy: {:.3f}, Test Loss: {:.3f}, Test Accuracy: {:.3f}, Time Duration: {:.3f}sec".\
                format(epoch+1, np.mean(losses), np.mean(accs), test_loss.item(), test_acc.item(), duration))

def eval_accuracy(y_hat, y):
    with torch.no_grad():
        diff = torch.abs(y_hat - y) / y
        return 1 - diff.mean()

if __name__ == "__main__":
    main()
