# Oil-Temperature-Prediction

# 電力トランスフォーマにおけるオイル温度予測モデル (Oil Temperature Prediction)

## プロジェクト概要
電力トランスフォーマの故障予兆検知を目的とし、ETDatasetを用いたオイル温度（OT）の時系列予測モデルを構築しました。モデルに二層のLSTMを使用しました。**差分予測**を導入し、予測精度の改善を行いました。

##  技術スタック
* **Language:** Python 3.x
* **Modeling:** TensorFlow / Keras (LSTM)
* **Analysis:** Pandas, Matplotlib, Seaborn, Statsmodels
* **Preprocessing:** Scikit-learn (MinMaxScaler)

## 検証プロセスと結果
本プロジェクトでは、以下の段階的な検証を行いました。

### 1. 課題の特定 
データ分析の結果、データの周期性（24時間、168時間）を確認しました。また、Trainデータ（平均17.3℃）とTestデータ（平均7.7℃）の間に大きな分布シフトを確認しました。
### 2. 仮説検証 
以下の2つの仮説に基づいてモデルを比較検証しました。

| モデル | アプローチ | 結果 (RMSE) | 
| :--- | :--- | :--- |
| **Baseline** | ラグ特徴量のみ  | 1.48 | 
| **Hypothesis 1** |  月特徴量  | 1.72 | 
| **Hypothesis 2** | ** 差分予測 ** | **0.65** | 

### 3. 最終結果
差分予測モデルを採用することで季節変動の影響を排除し、**RMSEを約65%削減**することに成功しました。

##  実行方法
必要なライブラリをインストールします。
```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow statsmodels
python main.py
```
## ファイル構成
* `main.py`: 統合スクリプト
* `ett.csv`: 使用したデータセット (ETDataset)
* `README.md`: 本ドキュメント

## 参考文献
* Dataset: [ETDataset (GitHub)](https://github.com/zhouhaoyi/ETDataset)
* Reference: H. Zhou et al., "Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting", AAAI 2021.
