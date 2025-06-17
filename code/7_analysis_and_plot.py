import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from scipy.stats import norm

def analysis():
    """
    加载点预测结果和历史残差分布，执行全面的MCM问题分析。
    - 使用基于残差的自助法来计算预测区间。
    """

    prediction_file = 'output/prediction_2028.csv'
    sport_level_prediction_file = 'output/prediction_2028_sport_level.csv'
    residuals_file = 'output/prediction_residuals.json' # 加载残差文件
    official_file = 'archive/summerOly_medal_counts.csv'
    
    # 创建国家全名到NOC代码的映射
    df_athletes = pd.read_csv('archive/summerOly_athletes.csv')
    name_to_code_map = df_athletes[['Team', 'NOC']].drop_duplicates().set_index('Team')['NOC'].to_dict()

    # 手动添加/修正一些可能的特殊映射
    name_to_code_map.update({'United States': 'USA', 'Great Britain': 'GBR', 'Soviet Union': 'URS', 'United Kingdom': 'GBR',
        'West Germany': 'FRG', 'East Germany': 'GDR', 'Russia': 'RUS', 'South Korea': 'KOR'})
    

    if not all(os.path.exists(f) for f in [prediction_file, sport_level_prediction_file, residuals_file, official_file]):
        print("错误: 找不到核心数据文件。请确保已运行所有前置脚本。")
        return

    df_2028 = pd.read_csv(prediction_file)
    df_sport_preds = pd.read_csv(sport_level_prediction_file)
    df_official = pd.read_csv(official_file)
    with open(residuals_file, 'r') as f:
        residuals = json.load(f)

    # --- 2. 计算残差的量化分位点以构建预测区间 ---
    q_gold = np.quantile(residuals['gold'], [0.025, 0.975])
    q_silver = np.quantile(residuals['silver'], [0.025, 0.975])
    q_bronze = np.quantile(residuals['bronze'], [0.025, 0.975])

    # --- 3. 计算每个项目的预测区间 ---
    df_sport_preds['gold_err_width'] = (q_gold[1] - q_gold[0]) * df_sport_preds['sport_importance']
    df_sport_preds['silver_err_width'] = (q_silver[1] - q_silver[0]) * df_sport_preds['sport_importance']
    df_sport_preds['bronze_err_width'] = (q_bronze[1] - q_bronze[0]) * df_sport_preds['sport_importance']
    
    df_sport_preds['total_err_width'] = np.sqrt(
        df_sport_preds['gold_err_width']**2 +
        df_sport_preds['silver_err_width']**2 +
        df_sport_preds['bronze_err_width']**2
    )

    # --- 4. 聚合到国家级合并误差 ---
    df_error_agg = df_sport_preds.groupby('NOC').agg(
        Gold_error=('gold_err_width', lambda x: np.sqrt(np.sum(x**2))),
        Total_error=('total_err_width', lambda x: np.sqrt(np.sum(x**2)))
    ).reset_index()

    # --- 5. 最终对比数据 ---
    df_2028.rename(columns={'pred_gold': 'mu_gold', 'pred_silver': 'mu_silver', 'pred_bronze': 'mu_bronze'}, inplace=True)
    df_2028['Total_mu'] = df_2028[['mu_gold', 'mu_silver', 'mu_bronze']].sum(axis=1)
    
    df_2028_with_errors = pd.merge(df_2028, df_error_agg, on='NOC', how='left').fillna(0)
    df_2028_with_errors['Gold_error'] = df_2028_with_errors['Gold_error'] / 2
    df_2028_with_errors['Total_error'] = df_2028_with_errors['Total_error'] / 2

    df_2024_raw = df_official[df_official['Year'] == 2024].copy()
    df_2024_raw['NOC_code'] = df_2024_raw['NOC'].map(name_to_code_map)
    df_2024_raw.dropna(subset=['NOC_code'], inplace=True) # 删除无法映射的行
    df_2024_raw['Total_2024'] = df_2024_raw[['Gold', 'Silver', 'Bronze']].sum(axis=1)

    df_2024_renamed = df_2024_raw.rename(columns={'Gold': 'Gold_2024'})
    df_comparison = pd.merge(
        df_2028_with_errors,
        df_2024_renamed[['NOC_code', 'Gold_2024', 'Total_2024']],
        left_on='NOC',
        right_on='NOC_code',
        how='left'
    ).fillna(0)
    
    # D计算增量
    df_comparison['gold_increment'] = df_comparison['mu_gold'] - df_comparison['Gold_2024']
    df_comparison['total_increment'] = df_comparison['Total_mu'] - df_comparison['Total_2024']

    if 'NOC_code' in df_comparison.columns:
        df_comparison.drop(columns=['NOC_code'], inplace=True)

    # --- 6. 执行所有分析模块 ---
    output_dir = 'plots'
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    sns.set_theme(style="whitegrid")

    analyze_prediction_intervals(df_comparison.copy(), output_dir)
    analyze_movers(df_comparison.copy(), output_dir)
    analyze_first_medal_winners(df_official.copy(), df_comparison.copy(), output_dir)
    analyze_key_sports(df_sport_preds.copy(), output_dir) 

    print(f"\n--- 全部分析完成！图表及报告已保存至 '{output_dir}/' 文件夹。 ---")

def analyze_prediction_intervals(df_plot, output_dir):
        print("分析模块 A: 生成带预测区间的奖牌榜图表...")
        df_gold = df_plot.sort_values(by='mu_gold', ascending=False).head(15)
        plt.figure(figsize=(12, 8)); plt.bar(df_gold['NOC'], df_gold['mu_gold'], yerr=df_gold['Gold_error'], capsize=5, color='gold', alpha=0.7)
        plt.title('Predicted 2028 LA Olympics Gold Medals (Top 15 with 95% Prediction Interval)', fontsize=16); plt.ylabel('Predicted Gold Medal Count')
        plt.xticks(rotation=45, ha='right'); plt.tight_layout(); plt.savefig(f'{output_dir}/plot_A1_gold_table_with_intervals.png', dpi=300); plt.close()

        df_total = df_plot.sort_values(by='Total_mu', ascending=False).head(15)
        plt.figure(figsize=(12, 8)); plt.bar(df_total['NOC'], df_total['Total_mu'], yerr=df_total['Total_error'], capsize=5, color='skyblue', alpha=0.7)
        plt.title('Predicted 2028 LA Olympics Total Medals (Top 15 with 95% Prediction Interval)', fontsize=16); plt.ylabel('Predicted Total Medal Count')
        plt.xticks(rotation=45, ha='right'); plt.tight_layout(); plt.savefig(f'{output_dir}/plot_A2_total_table_with_intervals.png', dpi=300); plt.close()

def analyze_movers(df_plot, output_dir):
    print("分析模块 B: 分析奖牌数变化最大的国家...")
    df_gold_improvers = df_plot.sort_values(by='gold_increment', ascending=False).head(10)
    plt.figure(figsize=(12, 8)); sns.barplot(x='gold_increment', y='NOC', data=df_gold_improvers, palette='viridis')
    plt.title('Top 10 Most Improved Countries by Predicted Gold Medals (2028 vs 2024)', fontsize=16); plt.xlabel('Net Increase in Gold Medals'); plt.tight_layout()
    plt.savefig(f'{output_dir}/plot_B1_gold_improvers.png', dpi=300); plt.close()

def analyze_first_medal_winners(df_hist, df_preds, output_dir):
    print("分析模块 C: 预测首次获奖国家...")
    all_medal_winners = df_hist[df_hist['Total'] > 0]['NOC'].unique()
    zero_medal_countries = set(df_preds['NOC'].unique()) - set(all_medal_winners)
    first_winners_candidates = df_preds[df_preds['NOC'].isin(zero_medal_countries)]
    predicted_first_winners = first_winners_candidates[first_winners_candidates['Total_mu'] - first_winners_candidates['Total_error'] > 0]
    predicted_first_winners = predicted_first_winners.sort_values('Total_mu', ascending=False)
    report = f"--- Analysis of Potential First-Time Medal Winners in 2028 ---\n"
    report += f"Our model predicts that {len(predicted_first_winners)} countries have a high chance (lower bound of 95% PI > 0) to win their first medal.\n"
    if not predicted_first_winners.empty: report += "Candidates: " + ", ".join(predicted_first_winners['NOC'].tolist())
    with open(f'{output_dir}/report_C1_first_medal_winners.txt', 'w') as f: f.write(report)

def analyze_key_sports(df_sport_preds, output_dir):
    print("分析模块 D: 分析各国的关键优势项目...")
    df_sport_preds['Total_pred'] = df_sport_preds['pred_gold'] + df_sport_preds['pred_silver'] + df_sport_preds['pred_bronze']
    for noc in ['USA', 'CHN', 'JPN', 'FRA', 'AUS', 'GBR']:
        df_country_sports = df_sport_preds[df_sport_preds['NOC'] == noc]
        if df_country_sports.empty: continue
        top_sports = df_country_sports.groupby('Sport')['Total_pred'].sum().nlargest(5)
        plt.figure(figsize=(10, 6)); top_sports.sort_values().plot(kind='barh', color='teal')
        plt.title(f"Predicted Top 5 Medal-Contributing Sports for {noc} in 2028", fontsize=16)
        plt.tight_layout(); plt.savefig(f'{output_dir}/plot_D_{noc}_top_sports.png', dpi=300); plt.close()

if __name__ == '__main__':
    analysis()