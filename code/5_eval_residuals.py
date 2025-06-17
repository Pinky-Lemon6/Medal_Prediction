import pandas as pd
import json
import numpy as np

def main():
    """
    此脚本的目的:
    1. 高效加载数据，将预测列表转换为DataFrame。
    2. 使用一次性的`merge`操作来匹配预测值与真实值。
    3. 向量化计算所有残差。
    4. 保存结果。
    """
    print("开始计算模型的历史预测残差...")
    
    # --- 1. 加载数据 ---
    df_full_data = pd.read_csv("data/cleaned/sports_medals_padded.csv")
    with open("output/test_results.json", "r") as f:
        predictions_list = json.load(f)['test_results']

    # --- 2. 向量化处理 ---
    print("将预测列表转换为DataFrame...")
    df_preds = pd.DataFrame(predictions_list)
    pred_rates_df = pd.DataFrame(df_preds['prediction'].tolist(), columns=['pred_gold_rate', 'pred_silver_rate', 'pred_bronze_rate'])
    df_preds = pd.concat([df_preds.drop(columns=['prediction', 'true_value']), pred_rates_df], axis=1)

    # --- 3. 准备合并 ---
    df_preds.rename(columns={
        'noc': 'NOC',
        'sex': 'Sex',
        'sport': 'Sport',
        'y_year': 'Year'
    }, inplace=True)

    # --- 4. merge操作 ---
    cols_to_merge = ['NOC', 'Sex', 'Sport', 'Year', 'Gold_rate', 'Silver_rate', 'Bronze_rate', 'Gold', 'Silver', 'Bronze']
    df_merged = pd.merge(
        df_preds,
        df_full_data[cols_to_merge],
        on=['NOC', 'Sex', 'Sport', 'Year'],
        how='left'
    )
    
    # 删除没有匹配到真实值的行
    df_merged.dropna(subset=['Gold_rate'], inplace=True)

    # --- 5. 向量化计算残差  ---

    df_merged['residual_gold'] = df_merged['Gold_rate'] - df_merged['pred_gold_rate']
    df_merged['residual_silver'] = df_merged['Silver_rate'] - df_merged['pred_silver_rate']
    df_merged['residual_bronze'] = df_merged['Bronze_rate'] - df_merged['pred_bronze_rate']

    # --- 6. 应用过滤器，筛选有意义的残差 ---
    meaningful_filter = (
        (df_merged['Gold'] > 0) | (df_merged['pred_gold_rate'] > 0.01) |
        (df_merged['Silver'] > 0) | (df_merged['pred_silver_rate'] > 0.01) |
        (df_merged['Bronze'] > 0) | (df_merged['pred_bronze_rate'] > 0.01)
    )
    df_final_residuals = df_merged[meaningful_filter]

    # --- 7. 提取残差并保存 ---
    residuals = {
        'gold': df_final_residuals['residual_gold'].tolist(),
        'silver': df_final_residuals['residual_silver'].tolist(),
        'bronze': df_final_residuals['residual_bronze'].tolist()
    }

    output_path = 'output/prediction_residuals.json'
    with open(output_path, 'w') as f:
        json.dump(residuals, f, indent=4)
        
    print(f"\n优化后的残差计算完成！共计 Gold: {len(residuals['gold'])}, Silver: {len(residuals['silver'])}, Bronze: {len(residuals['bronze'])} 条有效残差。")
    print(f"残差数据已保存到 {output_path}")

if __name__ == "__main__":
    main()