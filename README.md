# 🥇 2028年奥运会奖牌榜预测模型 (MCM 2025 Problem C)

本项目旨在为[2025年美国大学生数学建模竞赛（MCM）](https://www.immchallenge.org/mcm/index.html)C题提供一个完整的、端到端的解决方案。项目利用历史奥运会数据，通过深度学习模型（LSTM）对2028年洛杉矶奥运会的奖牌榜进行预测，并结合统计学方法对预测结果的不确定性进行分析，最终生成一系列深度分析图表与报告。

---

## 📂 项目结构

```
Medal_Prediction/
│
├── archive/
│   ├── summerOly_athletes.csv         # 原始运动员数据
│   ├── summerOly_medal_counts.csv     # 原始国家奖牌榜数据
│   ├── summerOly_hosts.csv            # 原始奥运会主办地数据
│   └── summerOly_programs.csv         # 原始奥运会项目数据
│
├── code/
│   ├── 1_create_dataset.py            # 数据预处理
│   ├── 2_collect_LSTM_trainingdata.py # 特征工程与训练数据生成
│   ├── 3_training.py                  # LSTM模型训练
│   ├── 4_eval.py                      # (可选) 测试集评估脚本
│   ├── 5_eval_residuals.py            # 计算模型历史误差（残差）
│   ├── 6_prediction.py                # 生成2028年最终预测
│   └── 7_analysis_and_plot.py         # 最终分析与可视化
│
├── data/
│   └── cleaned/
│       ├── sports_medals.csv      # 预处理后的项目级数据
│       ├── sports_medals_padded.csv # 已补全和缩放的特征数据
│       └── lstm_training_data_3.jsonl # 最终的LSTM训练数据
│
├── models/
│   └── olympic_lstm.pth     # 训练好的最终模型权重
│
├── output/
│   ├── prediction_residuals.json      # 模型的历史误差分布
│   ├── prediction_2028_sport_level.csv # 2028年项目级预测结果
│   └── prediction_2028.csv      # 2028年国家级最终预测结果
│
├── plots/
│   ├── plot_A1_gold_table_with_intervals.png # 带预测区间的金牌榜
│   ├── plot_A2_total_table_with_intervals.png# 带预测区间的总奖牌榜
│   └── ...                            # 其他分析图表
│
├── requirements.txt                   # 项目依赖库
└── README.md                          # 本文档
```

---

## ⚙️ 环境依赖

本项目基于 Python 3.x 开发。请首先在您的环境中安装所有必要的依赖库。

```bash
pip install -r requirements.txt
```
主要依赖库包括: `pandas`, `numpy`, `torch`, `scikit-learn`, `matplotlib`, `seaborn`, `tqdm`, `scipy`。

---

## 📄 代码文件说明

本项目的核心逻辑由`code/`目录下的7个脚本按顺序执行构成。

### **`1_create_dataset.py`**
* **功能**: 数据预处理的第一步。读取原始的运动员数据 (`summerOly_athletes.csv`)，精确统计每个国家(NOC)在历届奥运会、每个运动大项、不同性别下的奖牌数和参赛人数。
* **输出**: `data/cleaned/sports_medals.csv`。

### **`2_collect_LSTM_trainingdata.py`**
* **功能**: 特征工程。在预处理后的数据基础上，计算并合并一系列用于模型训练的高级特征。
* **核心处理**: 设计了如“主办国优势”、“项目重要性”、“奖牌转化效率(HHI指数)”等8个关键特征。对数据进行补全、计算奖牌率、并进行特征缩放。
* **输出**: `data/cleaned/sports_medals_padded.csv` (包含所有特征的完整数据) 和 `data/cleaned/lstm_training_data_3.jsonl` (用于模型训练的时间序列数据)。

### **`3._training.py`**
* **功能**: 模型训练。定义并训练一个LSTM（长短期记忆网络）深度学习模型。
* **核心处理**: 使用PyTorch框架，加载`.jsonl`训练数据，训练模型以学习历史奖牌数据的模式。
* **输出**: `models/olympic_lstm.pth` (训练好的模型参数) 和 `output/test_results.json` (在测试集上的预测结果)。

### **`4_eval.py`**
* **功能**: (可选) 模型评估脚本。用于将测试集上预测的“奖牌率”转换回“奖牌数”，并与真实值对比，以评估模型性能。

### **`5_eval_residuals.py`**
* **功能**: 计算模型历史误差。这是实现不确定性分析的关键步骤。
* **核心处理**: 加载模型在测试集上的预测结果，与真实值进行比较，计算出预测误差（残差）的分布。
* **输出**: `output/prediction_residuals.json`。

### **`6_prediction.py`**
* **功能**: 生成2028年最终预测。
* **核心处理**: 加载训练好的最终模型 (`olympic_lstm.pth`)，准备2016、2020、2024年的数据作为输入序列，对2028年进行点预测（最可能的结果）。
* **输出**: `output/prediction_2028_sport_level.csv` (项目级预测) 和 `output/prediction_2028.csv` (国家级预测)。

### **`7_analysis_and_plot.py`**
* **功能**: 最终分析与可视化。
* **核心处理**: 整合6号脚本生成的**点预测**和5号脚本生成的**历史残差**，通过统计学方法为点预测加上**不确定性区间**。并生成所有MCM问题要求的分析图表，如奖牌榜、进步最快国家、关键优势项目等。
* **输出**: `plots/` 目录下的所有分析图表和报告。

---

## 🚀 运行流程

请严格按照以下顺序执行脚本，以完整复现项目结果。

### **第1步：准备工作**
将所有原始数据文件（4个`.csv`文件）放入`archive/`目录中。

### **第2步：数据预处理与特征工程**
依次运行脚本1和脚本2，生成模型训练所需的数据。
```bash
python code/1_create_dataset.py
python code/2_collect_LSTM_trainingdata.py
```

### **第3步：模型训练**
运行脚本3，训练LSTM模型并保存权重。
```bash
python code/3_training.py
```

### **第4步：测试模型效果**
运行脚本4，在测试集上验证模型效果。
```bash
python code/4_eval.py
```

### **第5步：计算历史误差**
运行脚本5，为不确定性分析做准备。
```bash
python code/5_eval_residuals.py
```

### **第6步：生成2028年预测**
运行脚本6，得到2028年奥运会的奖牌预测基准值。
```bash
python code/6_prediction.py
```

### **第7步：最终分析与可视化**
运行脚本7，整合所有结果，生成最终的分析图表。
```bash
python code/7_analysis_and_plot.py
```
完成后，您可以在 `plots/` 文件夹中找到所有生成的图表。

---

## 📈 项目总结

本项目通过构建一个包含丰富特征的时间序列数据集，并利用LSTM网络强大的序列学习能力，成功地对奥运奖牌榜进行了预测。项目的亮点在于不仅给出了点预测，还创新性地引入了基于历史残差的不确定性分析，使得预测结果更加科学、完整。最终的分析和可视化模块直面MCM竞赛的各项要求，为国家奥委会的决策提供了富有洞察力的数据支持。