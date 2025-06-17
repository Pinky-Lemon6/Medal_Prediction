import pandas as pd
import json



def collect_sport_enum(df):
    print(df["Sport"].unique())


with open("data/cleaned/map_dic.json", "r") as f:
    map_dic = json.load(f)
def map_sport(sport):
    for key in map_dic.keys():
        if sport in map_dic[key]:
            return key
    return sport


def main():
    df_athletes = pd.read_csv("archive/summerOly_athletes.csv")
    df_athletes["Sport"] = df_athletes["Sport"].apply(map_sport)
    

    # --- 1. 精确计算奖牌数 ---
    df_medals_only = df_athletes[df_athletes['Medal'] != 'No medal'].copy()
    unique_medal_events = df_medals_only.drop_duplicates(
        subset=['Year', 'NOC', 'Sport', 'Event', 'Medal']
    )

    medal_dummies = pd.get_dummies(unique_medal_events['Medal'])
    df_processed = pd.concat([unique_medal_events[['Year', 'NOC', 'Sex', 'Sport']], medal_dummies], axis=1)
    final_medal_counts = df_processed.groupby(['Year', 'NOC', 'Sex', 'Sport']).sum().reset_index()
    
    # --- 2. 计算参与人数 ---
    df_participants = df_athletes.drop_duplicates(subset=['Year', 'NOC', 'Sport', 'Name'])
    participation_counts = df_participants.groupby(['Year', 'NOC', 'Sex', 'Sport']).size().reset_index(name='participates')

    # --- 3. 合并奖牌数与参与人数 ---
    data = pd.merge(
        final_medal_counts,
        participation_counts,
        on=['Year', 'NOC', 'Sex', 'Sport'],
        how='outer'
    )
    
    data[['Gold', 'Silver', 'Bronze']] = data[['Gold', 'Silver', 'Bronze']].fillna(0)
    data['participates'] = data['participates'].fillna(0)
    data[['Gold', 'Silver', 'Bronze', 'participates']] = data[['Gold', 'Silver', 'Bronze', 'participates']].astype(int)
    data = data.sort_values(by=["NOC", "Sex", "Sport", "Year"])
    
    print("处理后的数据示例:")
    print(data.head())
    print("\n检查中国队2024年游泳项目的例子:")
    print(data[(data['NOC']=='CHN') & (data['Year']==2024) & (data['Sport']=='Swimming')])
    
    data.to_csv("data/cleaned/sports_medals.csv", index=False)


if __name__ == "__main__":
    main()