import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.preprocessing import StandardScaler
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

def load_model():
    model = MDNNModel(input_size=27, num_classes=9, num_domains=9)  # Make sure these parameters match your model architecture
    model.load_state_dict(torch.load('MDNN_Mulit_Classification.pt'))
    model.eval()
    scaler = load('MDNN_Mulit_Scaler.joblib')
    with open('MDNN_Mulit_label.json', 'r') as f:
        label_mapping = json.load(f)
    return model, scaler, label_mapping

def predict(data, model, scaler, label_mapping):

    data = pd.DataFrame([data], index=[0])
    scaled_data = scaler.transform(data)
    scaled_data = torch.tensor(scaled_data).float()

    with torch.no_grad():
        outputs = model(scaled_data, domain_idx=6)
        _, predicted = torch.max(outputs.data, 1)

    # Convert index to label
    predicted_label = label_mapping[str(int(predicted.item()))]

    return predicted_label

def predict_csv(csv_file, model, scaler, label_mapping):
    data = pd.read_csv(csv_file)

    scaled_data = scaler.transform(data)
    scaled_data = torch.tensor(scaled_data).float()

    with torch.no_grad():
        outputs = model(scaled_data, domain_idx=6)
        _, predicted = torch.max(outputs.data, 1)

    predicted_labels = [label_mapping[str(int(idx))] for idx in predicted]

    return predicted_labels


#예제
# 모델들 불러오기
model, scaler, label_mapping = load_model()

# 예제 데이터
data = [42, 50, 270900, 270944, 267, 17, 44, 24220, 76, 108, 1687, 1, 0, 80, 0.0498, 0.2415, 0.1818, 0.0047, 0.4706, 1, 1, 2.4265, 0.9031, 1.6435, 0.8182, -0.2913, 0.5822]  # example data, should be replaced

predicted_label = predict(data, model, scaler, label_mapping)
predicted_label = predict_csv('mulit_classification_data_label_removed.csv', model, scaler, label_mapping)