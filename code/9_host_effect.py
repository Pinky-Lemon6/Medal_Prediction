import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os


def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


def get_host_year(NOC, host_info):
    for year, host in host_info.items():
        if host == NOC:
            return year
    return None


def plot_medals_over_years_from_host(df: pd.DataFrame, save_path: str = None):
    df_plot = df.T
    df_plot.index.name = 'Years_from_host'
    sns.set_style("whitegrid")
    plt.figure(figsize=(14, 8))
    num_nations = len(df_plot.columns)
    colors = sns.color_palette("husl", num_nations)
    df_plot.plot(ax=plt.gca(), marker='o', markersize=4, alpha=0.8, linewidth=1.5, color=colors)
    plt.axvline(x=0, color='red', linestyle='--', linewidth=1.5, label='Host Year')
    plt.xlabel('year_diff', fontsize=14)
    plt.ylabel('Total Medals', fontsize=14)
    # plt.title('Total Medals by NOC over Years from Host', fontsize=16)
    plt.title('', fontsize=1)
    plt.legend(title='NOC', bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout(rect=[0, 0, 0.88, 1])
    if save_path:
        output_dir = os.path.dirname(save_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()
    plt.close()


def plot_host_non_host_medals(df: pd.DataFrame, save_path: str = None):
    expected_cols = ['NOC', 'Average_medals_wo_host', 'Average_medals_w_host']
    if not all(col in df.columns for col in expected_cols):
        missing_cols = [col for col in expected_cols if col not in df.columns]
        raise ValueError(f"DataFrame must contain the following columns: {expected_cols}. Missing: {missing_cols}")
    df_melted = df.melt(id_vars='NOC', 
                        value_vars=['Average_medals_wo_host', 'Average_medals_w_host'],
                        var_name='Host Status', 
                        value_name='Average Medals')
    df_melted['Host Status'] = df_melted['Host Status'].replace({
        'Average_medals_w_host': 'Medals as Host',
        'Average_medals_wo_host': 'Medals as Non-Host'
    })
    sns.set_style("whitegrid")
    plt.figure(figsize=(14, 8))
    barplot = sns.barplot(x='NOC', y='Average Medals', hue='Host Status', data=df_melted, palette='viridis')
    plt.xlabel('National Olympic Committee (NOC)', fontsize=14)
    plt.ylabel('Average Medals', fontsize=14)
    # plt.title('Average Medals: Host vs. Non-Host Performance by NOC', fontsize=16)
    plt.title('', fontsize=1)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.legend(title='Host Status', bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout(rect=[0, 0, 0.88, 1]) 
    if save_path:
        output_dir = os.path.dirname(save_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()

    plt.close()




def main():
    host_info = pd.read_csv("archive/summerOly_hosts.csv").T.fillna("").to_dict()
    host_info = {row["Year"]: row["NOC"] for row in host_info.values()}
    
    medal_counts = pd.read_csv("archive/summerOly_medal_counts.csv")
    NOC_map = load_json("data/cleaned/NOC_map.json")
    medal_counts["NOC"] = medal_counts["NOC"].apply(lambda x: x.strip()).map(NOC_map)

    medal_counts["Host"] = medal_counts["Year"].map(host_info)
    medal_counts["Total_medals"] = medal_counts["Gold"] + medal_counts["Silver"] + medal_counts["Bronze"]
    medal_counts["is_host"] = medal_counts["NOC"] == medal_counts["Host"]
    hosted_nations = medal_counts[medal_counts["is_host"] == True]["NOC"].unique()
    medal_counts = medal_counts[medal_counts["NOC"].isin(hosted_nations)].copy().sort_values(by=["NOC", "Year"])
    medal_counts["host_year"] = medal_counts["NOC"].apply(lambda x: get_host_year(x, host_info))
    medal_counts["year_diff"] = medal_counts["Year"] - medal_counts["host_year"]

    df = pd.pivot_table(medal_counts, index=["NOC"], columns="year_diff", values="Total_medals", aggfunc="sum").fillna(0)
    plot_medals_over_years_from_host(df, "plots/medals_plot.png")

    country_medals_wo_host = medal_counts[medal_counts["is_host"] == False].groupby("NOC")["Total_medals"].mean().reset_index()
    country_medals_w_host = medal_counts[medal_counts["is_host"] == True].groupby("NOC")["Total_medals"].mean().reset_index()

    country_medals = country_medals_wo_host.merge(country_medals_w_host, on="NOC", suffixes=("_wo_host", "_w_host"))
    country_medals.columns = [i.replace("Total", "Average") for i in country_medals.columns]
    plot_host_non_host_medals(country_medals, "plots/host_non_host_medals.png")



if __name__ == "__main__":
    main()