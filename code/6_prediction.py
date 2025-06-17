import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import os


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


def main(window_size=3, input_size=8):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model_path = 'models/olympic_lstm.pth' 
    if not os.path.exists(model_path):
        print(f"错误：找不到模型文件 {model_path}。")
        return

    model = OlympicLSTM(input_size=input_size, hidden_size=64, num_layers=2)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    df_full_history = pd.read_csv("data/cleaned/sports_medals_padded.csv")
    
    # --- 由于文件中无2028年数据，我们使用2024年的项目数据作为最佳估计 ---
    df_programs = pd.read_csv("archive/summerOly_programs.csv")
    df_programs_long = df_programs.melt(
        id_vars=['Sport', 'Discipline'], 
        var_name='Year', 
        value_name='event_count'
    )
    df_programs_long['Year'] = df_programs_long['Year'].astype(str).str.replace('*', '', regex=False)
    df_programs_long = df_programs_long[pd.to_numeric(df_programs_long['Year'], errors='coerce').notna()]
    df_programs_long['Year'] = df_programs_long['Year'].astype(int)
    df_programs_long['event_count'] = pd.to_numeric(df_programs_long['event_count'], errors='coerce').fillna(0).astype(int)

    all_sport_importance = df_programs_long.groupby(['Year', 'Sport'])['event_count'].sum().reset_index()
    all_sport_importance.rename(columns={'event_count': 'sport_importance'}, inplace=True)

    sport_importance_2028 = all_sport_importance[all_sport_importance['Year'] == 2024][['Sport', 'sport_importance']].copy()
    
    if sport_importance_2028.empty:
        print("错误：在 summerOly_programs.csv 中未能找到2024年的项目数据，无法进行估计。")
        return

    # 准备用于缩放的Scaler 
    scaler_participates = MinMaxScaler()
    scaler_sport_importance = MinMaxScaler()
    scaler_hhi = MinMaxScaler()
    scaler_conversion_rate = MinMaxScaler()

    df_full_history['participates_scaled'] = scaler_participates.fit_transform(df_full_history[['participates']])
    df_full_history['sport_importance_scaled'] = scaler_sport_importance.fit_transform(df_full_history[['sport_importance']])
    df_full_history['hhi_scaled'] = scaler_hhi.fit_transform(df_full_history[['hhi']])
    df_full_history['conversion_rate_scaled'] = scaler_conversion_rate.fit_transform(df_full_history[['medal_conversion_rate']])
    
    # --- 准备预测输入序列 ---
    years_for_window = [2016, 2020, 2024]
    df_window_data = df_full_history[df_full_history['Year'].isin(years_for_window)].copy()

    group_sizes = df_window_data.groupby(['NOC', 'Sport', 'Sex'])['Year'].transform('size')
    df_valid_sequences = df_window_data[group_sizes == window_size].copy()
    df_valid_sequences = df_valid_sequences.sort_values(by=['NOC', 'Sport', 'Sex', 'Year'])

    metadata_df = df_valid_sequences.drop_duplicates(subset=['NOC', 'Sport', 'Sex'])
    prediction_metadata = metadata_df[['NOC', 'Sport', 'Sex']].to_dict('records')

    # 提取特征序列，并直接重塑为 (批次数, 时间步长, 特征数) 的形状
    features_to_use = [
        'Gold_rate', 'Silver_rate', 'Bronze_rate', 'is_host',
        'participates_scaled', 'sport_importance_scaled', 'hhi_scaled', 'conversion_rate_scaled'
    ]
    feature_values = df_valid_sequences[features_to_use].values
    prediction_inputs = feature_values.reshape(-1, window_size, input_size)

    input_tensor = torch.FloatTensor(np.array(prediction_inputs)).to(device)
    
    with torch.no_grad():
        predicted_rates_tensor = model(input_tensor)
    
    predicted_rates = predicted_rates_tensor.cpu().numpy()

    # --- 后处理与保存结果 ---
    df_results_2028 = pd.DataFrame(prediction_metadata)
    df_results_2028[['pred_gold_rate', 'pred_silver_rate', 'pred_bronze_rate']] = predicted_rates

    df_results_2028 = pd.merge(df_results_2028, sport_importance_2028, on='Sport', how='left')
    df_results_2028['sport_importance'] = df_results_2028['sport_importance'].fillna(0)

    df_results_2028['pred_gold'] = df_results_2028['pred_gold_rate'] * df_results_2028['sport_importance']
    df_results_2028['pred_silver'] = df_results_2028['pred_silver_rate'] * df_results_2028['sport_importance']
    df_results_2028['pred_bronze'] = df_results_2028['pred_bronze_rate'] * df_results_2028['sport_importance']
    # 单个项目奖牌预测情况
    df_results_2028.to_csv('output/prediction_2028_sport_level.csv', index=False)

    final_predictions = df_results_2028.groupby('NOC')[['pred_gold', 'pred_silver', 'pred_bronze']].sum()
    final_predictions[final_predictions < 0] = 0
    final_predictions = final_predictions.sort_values(by=['pred_gold', 'pred_silver', 'pred_bronze'], ascending=False).round(2)
    # 总奖牌榜
    output_path = 'output/prediction_2028.csv'
    final_predictions.to_csv(output_path)

    print(f"\n2028年洛杉矶奥运会奖牌榜预测完成！结果已保存到 {output_path}")
    print("预测奖牌榜前10名：")
    print(final_predictions.head(10))


if __name__ == '__main__':
    main()