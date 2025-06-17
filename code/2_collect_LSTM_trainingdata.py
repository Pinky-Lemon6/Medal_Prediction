# 对每个国家，计算出其所有有record的年份，并作补全，并collect数据

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import json
from sklearn.preprocessing import MinMaxScaler


def add_strategic_features(df_sport_level, df_athletes_raw):
    """
    计算并添加两个国家层面的战略特征：
    1. 奖牌转化率 (medal_conversion_rate)
    2. 优势项目集中度指数 (hhi)
    """

    # --- 1. 计算奖牌转化率 ---
    total_medals_per_country = df_sport_level.groupby(['Year', 'NOC'])[['Gold', 'Silver', 'Bronze']].sum().sum(axis=1).reset_index(name='TotalMedals')
    total_athletes_per_country = df_athletes_raw.drop_duplicates(subset=['Year', 'NOC', 'Name']).groupby(['Year', 'NOC']).size().reset_index(name='TotalAthletes')
    df_strategy = pd.merge(total_medals_per_country, total_athletes_per_country, on=['Year', 'NOC'], how='left')
    df_strategy['medal_conversion_rate'] = (df_strategy['TotalMedals'] / df_strategy['TotalAthletes']).fillna(0)

    # --- 2. 计算优势项目集中度指数 (HHI) ---
    df_sport_level['TotalMedals_per_sport'] = df_sport_level[['Gold', 'Silver', 'Bronze']].sum(axis=1)
    df_sport_level['TotalMedals_per_country'] = df_sport_level.groupby(['Year', 'NOC'])['TotalMedals_per_sport'].transform('sum')
    df_sport_level['medal_proportion'] = (df_sport_level['TotalMedals_per_sport'] / df_sport_level['TotalMedals_per_country']).fillna(0)
    df_sport_level['medal_proportion_sq'] = df_sport_level['medal_proportion'] ** 2
    hhi = df_sport_level.groupby(['Year', 'NOC'])['medal_proportion_sq'].sum().reset_index(name='hhi')
    df_strategy = pd.merge(df_strategy, hhi, on=['Year', 'NOC'], how='left')
    
    # 清理不再需要的临时列
    df_sport_level.drop(columns=['TotalMedals_per_sport', 'TotalMedals_per_country', 'medal_proportion', 'medal_proportion_sq'], inplace=True)

    return df_strategy[['Year', 'NOC', 'medal_conversion_rate', 'hhi']]


def padding(df):
    all_years = sorted(df["Year"].unique())
    all_combinations = []
    for noc in df["NOC"].unique():
        for sex in df["Sex"].unique():
            for sport in df["Sport"].unique():
                for year in all_years:
                    all_combinations.append({
                        "NOC": noc,
                        "Sex": sex,
                        "Sport": sport,
                        "Year": year
                    })
    complete_df = pd.DataFrame(all_combinations)
    result = pd.merge(
        complete_df,
        df,
        on=["NOC", "Sex", "Sport", "Year"],
        how="left"
    )
    result[["Gold", "Silver", "Bronze", "participates"]] = result[["Gold", "Silver", "Bronze", "participates"]].fillna(0)
    return result.sort_values(by=["NOC", "Sex", "Sport", "Year"])


def calculate_rate(df):
    # 计算每个国家的金牌率、银牌率、铜牌率
    # 金牌率是指，在当年该项目上，该国获得金牌数/该年该项目的总金牌数
    # 计算每个年份、性别、运动项目下的总奖牌数
    yearly_totals = df.groupby(['Year', 'Sex', 'Sport'])[['Gold', 'Silver', 'Bronze']].sum().reset_index()
    yearly_totals.rename(columns={'Gold': 'Gold_total', 'Silver': 'Silver_total', 'Bronze': 'Bronze_total'}, inplace=True)
    
    df_with_totals = pd.merge(df, yearly_totals, on=['Year', 'Sex', 'Sport'], how='left')

    epsilon = 1e-9 # 防止分母为0
    df_with_totals['Gold_rate'] = df_with_totals['Gold'] / (df_with_totals['Gold_total'] + epsilon)
    df_with_totals['Silver_rate'] = df_with_totals['Silver'] / (df_with_totals['Silver_total'] + epsilon)
    df_with_totals['Bronze_rate'] = df_with_totals['Bronze'] / (df_with_totals['Bronze_total'] + epsilon)
    df_with_totals[['Gold_rate', 'Silver_rate', 'Bronze_rate']] = df_with_totals[['Gold_rate', 'Silver_rate', 'Bronze_rate']].fillna(0)
    
    return df_with_totals


def collect_training_data(df, features, window_size=5):
    training_data = []
    
    feature_columns = features
    
    for (noc, sex, sport), group in tqdm(df.groupby(['NOC', 'Sex', 'Sport'])):
        group = group.sort_values('Year')
        
        if len(group) <= window_size:
            continue
            
        for i in range(window_size, len(group)):
            X_window = group.iloc[i-window_size:i][feature_columns].values
            y_window = group.iloc[i][['Gold_rate', 'Silver_rate', 'Bronze_rate']].values
            y_year = group.iloc[i]['Year']
            
            data_point = {
                'noc': noc,
                'sex': sex,
                'sport': sport,
                'X': X_window.tolist(),
                'y': y_window.tolist(),
                'y_year': int(y_year)
            }
            if np.sum(X_window) > 0:
                training_data.append(data_point)
    
    return training_data


def main(window_size=3):
    # --- 0. 加载原始运动员数据，我们现在需要它来计算战略特征 ---
    df_athletes_raw = pd.read_csv("archive/summerOly_athletes.csv")

    # 加载按项目划分的主数据
    df = pd.read_csv("data/cleaned/sports_medals.csv")

    # --- 1. 计算并合并战略特征 ---
    df_strategic_features = add_strategic_features(df.copy(), df_athletes_raw)
    df_with_features = pd.merge(df, df_strategic_features, on=['Year', 'NOC'], how='left')

    # --- 2. 加载并合并其他特征 (host, sport_importance) ---
    df_hosts = pd.read_csv("archive/summerOly_hosts.csv")
    df_hosts.rename(columns={'NOC': 'Host_NOC'}, inplace=True)
    df_programs = pd.read_csv("archive/summerOly_programs.csv")
    df_programs_long = df_programs.melt(id_vars=['Sport', 'Discipline'], var_name='Year', value_name='event_count')
    df_programs_long = df_programs_long[pd.to_numeric(df_programs_long['event_count'], errors='coerce').notna()]
    df_programs_long['Year'] = df_programs_long['Year'].str.replace('*', '', regex=False).astype(int)
    sport_importance = df_programs_long.groupby(['Year', 'Sport'])['event_count'].sum().reset_index(name='sport_importance')
    df_with_features = pd.merge(df_with_features, df_hosts[['Year', 'Host_NOC']], on='Year', how='left')
    df_with_features['is_host'] = (df_with_features['NOC'] == df_with_features['Host_NOC']).astype(int)
    df_with_features.drop(columns=['Host_NOC'], inplace=True)
    df_with_features = pd.merge(df_with_features, sport_importance, on=['Year', 'Sport'], how='left')
    df_with_features['sport_importance'] = df_with_features['sport_importance'].fillna(0)
    
    # --- 3. 数据补全与奖牌率计算 ---
    df_padded = padding(df_with_features)
    df_processed = calculate_rate(df_padded)

    # --- 4. 特征缩放  ---
    scaler = MinMaxScaler()
    df_processed[['participates_scaled']] = scaler.fit_transform(df_processed[['participates']])
    df_processed[['sport_importance_scaled']] = scaler.fit_transform(df_processed[['sport_importance']])
    df_processed[['hhi_scaled']] = scaler.fit_transform(df_processed[['hhi']])
    df_processed[['conversion_rate_scaled']] = scaler.fit_transform(df_processed[['medal_conversion_rate']])
    
    # 保存包含所有特征的中间文件
    df_processed.to_csv("data/cleaned/sports_medals_padded.csv", index=False)

    # --- 5. 准备最终训练数据 ---
    features_to_use = [
        'Gold_rate', 'Silver_rate', 'Bronze_rate', 
        'is_host', 
        'participates_scaled', 
        'sport_importance_scaled',
        'hhi_scaled',               
        'conversion_rate_scaled'  
    ]
    
    training_data = collect_training_data(df_processed, features=features_to_use, window_size=window_size)

    # 保存训练文件
    with open(f'data/cleaned/lstm_training_data_{window_size}.jsonl', 'w') as f:
        for data in training_data:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')
    
    print("数据处理完成！")
    return


if __name__ == "__main__":
    main()