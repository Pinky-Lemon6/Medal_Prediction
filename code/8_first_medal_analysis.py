import pandas as pd
import numpy as np
from scipy.stats import entropy
from lifelines import CoxPHFitter
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os

def generate_survival_data(input_filepath='archive/summerOly_athletes.csv'):
    """
    从原始奥运数据生成用于生存分析的特征数据。
    """
    try:
        df_athletes = pd.read_csv(input_filepath)
    except FileNotFoundError:
        print(f"错误: 找不到输入文件 '{input_filepath}'。")
        return None

    # 计算每届奥运会每个国家的唯一运动员人数
    participation_df = df_athletes.drop_duplicates(subset=['Year', 'NOC', 'Name'])
    participation_counts = participation_df.groupby(['Year', 'NOC']).size().reset_index(name='AthleteCount')
    participation_counts = participation_counts.sort_values(by=['NOC', 'Year'])

    # 计算生存数据 (duration 和 event_observed)
    medal_winners_noc = set(df_athletes[df_athletes['Medal'] != 'No medal']['NOC'].unique())
    first_medal_year_df = df_athletes[df_athletes['Medal'] != 'No medal'].groupby('NOC')['Year'].min().reset_index(name='FirstMedalYear')
    countries = participation_counts['NOC'].unique()
    survival_data = []

    for noc in countries:
        country_participations = participation_counts[participation_counts['NOC'] == noc]
        event_observed = 1 if noc in medal_winners_noc else 0
        duration = 0
        if event_observed == 1:
            first_medal_year = first_medal_year_df[first_medal_year_df['NOC'] == noc]['FirstMedalYear'].iloc[0]
            duration = country_participations[country_participations['Year'] <= first_medal_year].shape[0]
        else:
            duration = country_participations.shape[0]
        survival_data.append({'NOC': noc, 'duration': duration, 'event_observed': event_observed})
    survival_df = pd.DataFrame(survival_data)

    # 计算特征一：持续投入度
    olympic_years_window = sorted([y for y in participation_counts['Year'].unique() if 1988 <= y <= 2024])
    commitments = []
    for noc in countries:
        country_participations_years = set(participation_counts[participation_counts['NOC'] == noc]['Year'])
        max_streak, current_streak = 0, 0
        for year in olympic_years_window:
            if year in country_participations_years:
                current_streak += 1
            else:
                max_streak = max(max_streak, current_streak)
                current_streak = 0
        max_streak = max(max_streak, current_streak)
        commitments.append({'NOC': noc, 'ContinuousCommitment': max_streak})
    commitment_df = pd.DataFrame(commitments)

    # 计算特征二：运动员增长势头
    participation_counts['GrowthRate'] = participation_counts.groupby('NOC')['AthleteCount'].pct_change()
    def get_last_5_avg_growth(group):
        return group['GrowthRate'].dropna().tail(5).mean()
    momentum_df = participation_counts.groupby('NOC').apply(get_last_5_avg_growth).reset_index(name='AthleteMomentum')
    momentum_df['AthleteMomentum'] = momentum_df['AthleteMomentum'].fillna(0)

    # 计算特征三：项目专注度
    sport_concentration = []
    for noc in countries:
        country_sports = df_athletes[df_athletes['NOC'] == noc]
        if country_sports.empty:
            sport_concentration.append({'NOC': noc, 'SportConcentration': 0})
            continue
        sport_counts = country_sports['Sport'].value_counts()
        concentration_entropy = entropy(sport_counts, base=2)
        sport_concentration.append({'NOC': noc, 'SportConcentration': concentration_entropy})
    concentration_df = pd.DataFrame(sport_concentration)
    
    # 合并所有数据
    final_df = pd.merge(survival_df, commitment_df, on='NOC')
    final_df = pd.merge(final_df, momentum_df, on='NOC')
    final_df = pd.merge(final_df, concentration_df, on='NOC')

    print("数据生成与特征工程完成。")
    return final_df


def train_and_predict(df: pd.DataFrame):
    """
    使用准备好的生存数据训练Cox模型，并预测首次夺牌概率。
    """
    
    # 定义特征列
    feature_cols = ['ContinuousCommitment', 'AthleteMomentum', 'SportConcentration']
    
    # 特征标准化
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    
    # 训练Cox模型
    cph = CoxPHFitter()
    model_df = df[['duration', 'event_observed'] + feature_cols]
    cph.fit(model_df, duration_col='duration', event_col='event_observed')
    
    print("模型训练完成。摘要如下：")
    cph.print_summary()

    # 进行个体化预测
    medal_less_countries = df[df['event_observed'] == 0].copy()
    predictions = []
    for _, country_row in medal_less_countries.iterrows():
        current_duration = country_row['duration']
        feature_vector = country_row[feature_cols]
        survival_prob_at_t_plus_1 = cph.predict_survival_function(
            feature_vector.to_frame().T, 
            times=[current_duration + 1]
        ).iloc[0, 0]
        first_medal_prob = 1 - survival_prob_at_t_plus_1
        predictions.append({
            'NOC': country_row['NOC'],
            'predicted_prob_for_2028': first_medal_prob
        })

    predictions_df = pd.DataFrame(predictions)
    predictions_df = predictions_df.sort_values(by='predicted_prob_for_2028', ascending=False)
    
    print("预测完成。")
    return predictions_df


def plot_results(df_preds: pd.DataFrame, output_dir='plots'):
    """
    将预测结果可视化并保存。
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    sns.set_theme(style="whitegrid")
    
    # 图 (a): 首次夺牌概率散点图
    df_plot_a = df_preds.sort_values(by='predicted_prob_for_2028', ascending=False).head(50)
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    scatter = ax1.scatter(
        x=df_plot_a['predicted_prob_for_2028'], 
        y=df_plot_a['NOC'],
        c=df_plot_a['predicted_prob_for_2028'],
        cmap='viridis_r', s=50
    )
    ax1.invert_xaxis()
    ax1.invert_yaxis()
    ax1.set_xlabel('Probability')
    ax1.set_ylabel('Country (NOC)')
    ax1.set_title('Top 50 Countries: First Medal Probability (2028)')
    plt.colorbar(scatter, ax=ax1, label='Probability', shrink=0.7)
    plt.tight_layout()
    plot_a_filename = os.path.join(output_dir, 'plot_a_first_medal_probability.png')
    plt.savefig(plot_a_filename, dpi=300)
    print(f"图 (a) 已保存为: {plot_a_filename}")
    plt.close(fig1)

    # 图 (b): 蒙特卡洛模拟直方图
    probabilities = df_preds['predicted_prob_for_2028'].values
    n_simulations = 10000
    winning_counts = np.sum(np.random.rand(n_simulations, len(probabilities)) < probabilities, axis=1)
    expected_winners = np.mean(winning_counts)
    print(f"蒙特卡洛模拟完成，预期将有 {expected_winners:.2f} 个国家首次获奖。")

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.hist(winning_counts, bins=range(min(winning_counts), max(winning_counts) + 1), edgecolor='black')
    ax2.set_xlabel('Number of Winning Countries')
    ax2.set_ylabel('Frequency')
    ax2.set_title(f'Monte Carlo Simulation ({n_simulations} runs)')
    plt.tight_layout()
    plot_b_filename = os.path.join(output_dir, 'plot_b_monte_carlo_simulation.png')
    plt.savefig(plot_b_filename, dpi=300)
    print(f"图 (b) 已保存为: {plot_b_filename}")
    plt.close(fig2)


def main():
    """
    主函数，按顺序执行所有步骤。
    """
    # 步骤 1: 生成数据
    survival_feature_df = generate_survival_data(input_filepath='archive/summerOly_athletes.csv')
    
    if survival_feature_df is not None:
        # 步骤 2: 训练模型并预测
        predictions_df = train_and_predict(survival_feature_df)
        
        # 步骤 3: 可视化结果
        plot_results(predictions_df, output_dir='plots')
        
        print("\n所有分析已成功完成！")

if __name__ == '__main__':
    main()