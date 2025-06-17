# 训练LSTM模型
# 使用torch来做
# 数据格式：
    # {"X": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.14285714285714285]], "y": [0.0, 0.0, 0.125]}
    # 其中X为前五年的金银铜牌率，y为第六年的金银铜牌率
# 目标：
    # 预测金银铜牌率

import torch
import torch.nn as nn
import json
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import os

# 定义数据集类
class OlympicDataset(Dataset):
    def __init__(self, X, y, metadata):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.metadata = metadata  # 存储元数据
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.metadata[idx]

# 定义LSTM模型
class OlympicLSTM(nn.Module):
    def __init__(self, input_size=3, hidden_size=64, num_layers=2):
        super(OlympicLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 3)  # 输出3个值：金银铜牌率
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # 只使用最后一个时间步的输出
        return out

# 加载数据
def load_data(file_path):
    X = []
    y = []
    metadata = []  # 存储所有元数据
    
    with open(file_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            X.append(data['X'])
            y.append(data['y'])
            metadata.append({
                'noc': data['noc'],
                'sex': data['sex'],
                'sport': data['sport'],
                'y_year': data['y_year']
            })
    
    return np.array(X), np.array(y), metadata

# 主函数
def main():
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 加载数据
    X, y, metadata = load_data('data/cleaned/lstm_training_data_3.jsonl')
    
    # 划分训练集和测试集
    train_mask = [m['y_year'] < 2017 for m in metadata]
    test_mask = [m['y_year'] >= 2017 for m in metadata]
    
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    metadata_train = [m for i, m in enumerate(metadata) if train_mask[i]]
    metadata_test = [m for i, m in enumerate(metadata) if test_mask[i]]
    
    # 创建数据加载器
    train_dataset = OlympicDataset(X_train, y_train, metadata_train)
    test_dataset = OlympicDataset(X_test, y_test, metadata_test)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    
    # # 全量数据训练代码
    # # 加载包含所有特征的训练数据
    # data_path = f'data/cleaned/lstm_training_data_3_strategic.jsonl'
    # X_full, y_full, metadata_full = load_data(data_path)
    
    # # --- : 不再划分训练/测试集，而是使用全部数据 ---
    # print(f"使用全部 {len(X_full)} 条数据进行最终训练...")
    # full_dataset = OlympicDataset(X_full, y_full, metadata_full)
    # full_loader = DataLoader(full_dataset, batch_size=32, shuffle=True) # shuffle=True在训练时是好习惯
    
    # 初始化模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = OlympicLSTM(input_size=8).to(device)
    
    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    # criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 训练模型
    num_epochs = 80
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch_X, batch_y, _ in train_loader:  # 忽略元数据
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')
    
    # 评估模型

    # 评估模型
    model.eval()
    test_loss = 0
    test_results = []  # 存储所有测试结果

    with torch.no_grad():
        for batch_X, batch_y, batch_metadata in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            test_loss += criterion(outputs, batch_y).item()

            # 收集每个样本的预测结果和元数据
            predictions = outputs.cpu().numpy()
            true_values = batch_y.cpu().numpy()

            # 遍历批次中的每个样本
            for i in range(len(predictions)):
                # 确保元数据中的每个字段都是原生Python类型
                noc_val = batch_metadata['noc'][i]
                sex_val = batch_metadata['sex'][i]
                sport_val = batch_metadata['sport'][i]
                y_year_val = batch_metadata['y_year'][i]

                # 进一步检查并转换，以防 DataLoader 转换了它们
                if isinstance(noc_val, torch.Tensor):
                    noc_val = noc_val.item() if noc_val.numel() == 1 else noc_val.tolist()
                if isinstance(sex_val, torch.Tensor):
                    sex_val = sex_val.item() if sex_val.numel() == 1 else sex_val.tolist()
                if isinstance(sport_val, torch.Tensor):
                    sport_val = sport_val.item() if sport_val.numel() == 1 else sport_val.tolist()
                if isinstance(y_year_val, torch.Tensor):
                    y_year_val = y_year_val.item() # y_year 应该是单个数值

                result = {
                    'noc': noc_val,
                    'sex': sex_val,
                    'sport': sport_val,
                    'y_year': y_year_val,
                    'prediction': predictions[i].tolist(),
                    'true_value': true_values[i].tolist()
                }
                test_results.append(result)

    avg_test_loss = test_loss / len(test_loader)
    print(f'Test Loss: {avg_test_loss:.4f}')

    # 保存预测结果
    results = {
        'test_results': test_results,
        'test_loss': float(avg_test_loss)
    }

    # 确保输出目录存在
    os.makedirs('output', exist_ok=True)

    # 保存结果到JSON文件
    with open('output/test_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    # 保存模型
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), 'models/olympic_lstm.pth')
    



if __name__ == '__main__':
    main()