import pandas as pd
import json
from tqdm import tqdm

def get_total_medals(medals, year, sex, sport, level):

    subset = medals[(medals["Year"] == year) & (medals["Sex"] == sex) & (medals["Sport"] == sport)]
    if subset.empty:
        return 0 
    
    index = subset.index[0]
    if level == "Gold":
        return medals.iloc[index]["Gold_total"]
    elif level == "Silver":
        return medals.iloc[index]["Silver_total"]
    else:
        return medals.iloc[index]["Bronze_total"]

def main():

    groundtruth = pd.read_csv("data/cleaned/sports_medals.csv")
    groundtruth = groundtruth[groundtruth["Year"] >= 2017].copy()
    groundtruth = groundtruth.groupby(["NOC", "Year"]).sum().reset_index()
    groundtruth = groundtruth[["NOC", "Year", "Gold", "Silver", "Bronze"]]
    groundtruth.to_csv("output/sports_medals_groundtruth.csv", index=False)

    
    medals = pd.read_csv("data/cleaned/sports_medals_padded.csv")
    medals = medals[medals["Year"] >= 2017].copy()
    medals = medals[["Year", "Sex", "Sport", "Gold_total", "Silver_total", "Bronze_total"]].drop_duplicates().reset_index(drop=True)

   
    with open("output/test_results.json", "r") as f:
        pred = json.load(f)

    ret = {} 
    
    for row in tqdm(pred["test_results"]):
        noc = row["noc"]
        year = row["y_year"]
        
        gold_rate = row["prediction"][0]
        silver_rate = row["prediction"][1]
        bronze_rate = row["prediction"][2]

        total_golds = get_total_medals(medals, year, row["sex"], row["sport"], "Gold")
        total_silvers = get_total_medals(medals, year, row["sex"], row["sport"], "Silver")
        total_bronzes = get_total_medals(medals, year, row["sex"], row["sport"], "Bronze")
        
       
        pred_gold = gold_rate * total_golds
        pred_silver = silver_rate * total_silvers
        pred_bronze = bronze_rate * total_bronzes
        
        
        if noc not in ret:
            ret[noc] = {}
        if year not in ret[noc]:
            ret[noc][year] = {"Gold": 0, "Silver": 0, "Bronze": 0}

        ret[noc][year]["Gold"] += pred_gold
        ret[noc][year]["Silver"] += pred_silver
        ret[noc][year]["Bronze"] += pred_bronze
    
    
    with open("output/test_results_eval.json", "w") as f:
        json.dump(ret, f, indent=4)

    print("评估完成！最终预测的各国奖牌数已保存。")
    return

if __name__ == "__main__":
    main()