import csv
import torch
import numpy as np
from torch import nn, optim
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import time

RND_MEAN, RND_STD = 0.0, 1.0
LEARNING_RATE = 0.001
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#model2와 동일
class Model(nn.Module):
    def __init__(self, input_cnt, set_hidden, output_cnt):
        super(Model, self).__init__()
        self.layers = nn.ModuleList()

        last_cnt = input_cnt
        for hidden_cnt in set_hidden:
            self.layers.append(nn.Linear(last_cnt, hidden_cnt))
            self.layers.append(nn.ReLU())
            last_cnt = hidden_cnt

        self.layers.append(nn.Linear(last_cnt, output_cnt))
        self.layers.append(nn.Sigmoid())

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# 데이터 불균형할경우 adjust_ratio를 사용할수있도록 매개변수 추가
def binary_main(epoch_count = 10, mb_size = 10, report=1, train_ratio = 0.6, adjust_ratio = True, set_hidden = [1, 2]):
    data, input_cnt, output_cnt = binary_load_dataset(adjust_ratio)

    model = Model(input_cnt, set_hidden, output_cnt).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

    train_and_test(model, criterion, optimizer, data, epoch_count, mb_size, report, train_ratio)

# adjust_ratio가 True일 경우 타겟의 이진분류값을 1:1로 조절
# ***False로 설정시 모델 작동할때 모든 클래스를 한쪽값으로만 예측하여 경고창 발생
def binary_load_dataset(adjust_ratio):
    pulsars, stars = [], []

    with open('./binary_classification_data.csv') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)
        for row in csvreader:
            if row[8] == '1':
                pulsars.append(row)
            else:
                stars.append(row)

    input_cnt, output_cnt = 8, 1

    star_cnt, pulsar_cnt = len(stars), len(pulsars)

    if adjust_ratio:
        data = np.zeros([2 * star_cnt, 9])
        data[:star_cnt, :] = np.asarray(stars, dtype="float32")
        for n in range(star_cnt):
            data[star_cnt + n] = np.asarray(pulsars[n % pulsar_cnt], dtype='float32')
    else:
        data = np.zeros([star_cnt + pulsar_cnt, 9])
        data[:star_cnt, :] = np.asarray(stars, dtype='float32')
        data[star_cnt:, :] = np.asarray(pulsars, dtype='float32')

    return data, input_cnt, output_cnt

# pricision, recall, F1 스코어를 sklearn 라이브러리를 통해 추가
# Baseline 코드의 def safe_div(p, q)는 pricision 등을 계산할때 필요한 함수였으므로 삭제
def train_and_test(model, criterion, optimizer, data, epoch_count, mb_size, report, train_ratio):
    shuffle_map = np.arange(data.shape[0])
    np.random.shuffle(shuffle_map)
    mini_batch_step_count = int(data.shape[0] * train_ratio) // mb_size
    test_begin_index = mini_batch_step_count * mb_size
    test_data = data[shuffle_map[test_begin_index:]]

    for epoch in range(epoch_count):
        losses = []
        start_time = time.time()
        for nth in range(mini_batch_step_count):
            if nth == 0:
                np.random.shuffle(shuffle_map[:test_begin_index])
            train_data = data[shuffle_map[mb_size * nth:mb_size * (nth + 1)]]
            train_x = torch.Tensor(train_data[:, :-1]).to(device)
            train_y = torch.Tensor(train_data[:, -1:]).to(device)

            model.train()
            optimizer.zero_grad()
            pred = model(train_x)
            loss = criterion(pred, train_y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        if report > 0 and (epoch + 1) % report == 0:
            model.eval()
            with torch.no_grad():
                test_x = torch.Tensor(test_data[:, :-1]).to(device)
                test_y = torch.Tensor(test_data[:, -1:]).to(device)
                pred = model(test_x)
                loss = criterion(pred, test_y)
                y_hat = pred > 0.5
                y_real = test_y > 0.5
                eval_res = eval_accuracy(y_hat, y_real)
                print(f"[Epoch {epoch + 1}] Train Loss = {np.mean(losses)} | Test Loss = {loss.item()}")
                print(f"Test Accuracy = {eval_res[0]}, Precision = {eval_res[1]}, Recall = {eval_res[2]}, F1 = {eval_res[3]}")
                end_time = time.time()
                print(f"Time Duration = {end_time - start_time:.3f}sec")

def eval_accuracy(y_hat, y_real):
    y_hat = y_hat.cpu().numpy()
    y_real = y_real.cpu().numpy()
    accuracy = accuracy_score(y_real, y_hat)
    precision = precision_score(y_real, y_hat)
    recall = recall_score(y_real, y_hat)
    f1 = f1_score(y_real, y_hat)
    return [accuracy, precision, recall, f1]

if __name__ == "__main__":
    binary_main()
