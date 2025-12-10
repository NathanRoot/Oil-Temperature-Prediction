import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import random
import tensorflow as tf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# 再現性の確保
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

set_seed(42)

# グラフ描画設定
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (15, 6)
plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'sans-serif'

# =============================================================================
# 1. データの読み込みと前処理
# =============================================================================
print("[Process] データの読み込みを開始します...")

try:
    df = pd.read_csv("drive/MyDrive/athena/oil_project/ett.csv", parse_dates=['date'], index_col='date')
except FileNotFoundError:
    print("Error: 指定されたファイル 'ett.csv' が見つかりません。")
    raise

# データクレンジング
df = df.dropna()
print(f"データ件数: {len(df)}")
print(f"期間: {df.index.min()} ~ {df.index.max()}")


# =============================================================================
# 2. 探索的データ分析 (EDA) & 特徴量選定の根拠
# =============================================================================
print("\n[Process] EDAを実行中 (周期性・相関分析)...")

# 2-1. 時系列成分分解 (トレンド・周期性の確認)
decomposition = seasonal_decompose(df['OT'], model='additive', period=24)
fig = decomposition.plot()
fig.set_size_inches(15, 10)
plt.suptitle('Seasonal Decomposition', fontsize=16)
plt.tight_layout()
plt.savefig('eda_seasonal_decompose.png')
plt.show()

# 2-2. 自己相関(ACF)による周期性の確認 (ラグ168時間の検証)
plt.figure(figsize=(15, 5))
plot_acf(df['OT'], lags=200, ax=plt.gca(), title='Autocorrelation Function (Check for 168h Seasonality)')
plt.axvline(x=168, color='red', linestyle='--', label='Lag 168 (1 Week)')
plt.legend()
plt.tight_layout()
plt.savefig('eda_acf_168h.png')
plt.show()

# 2-3. 相互相関(CCF)による外部変数ラグの探索
lags = range(1, 50)
corrs = [df['OT'].corr(df['HUFL'].shift(lag)) for lag in lags]

plt.figure(figsize=(15, 5))
plt.plot(lags, corrs, marker='o', markersize=4, label='Correlation OT vs HUFL(lag)')
plt.axvline(x=33, color='r', linestyle='--', alpha=0.6, label='Lag 33')
plt.axvline(x=38, color='g', linestyle='--', alpha=0.6, label='Lag 38')
plt.title('Cross-Correlation Analysis', fontsize=14)
plt.legend()
plt.tight_layout()
plt.savefig('eda_ccf_lags.png')
plt.show()

print("分析完了")

# =============================================================================
# 関数定義: データセット作成 & モデル学習・評価
# =============================================================================
def create_dataset(data, target_col='OT', use_month=False, use_diff=False):
    """
    EDAに基づくラグ特徴量を用いて学習用データセットを作成する
    """
    df_eng = data.copy()
    
    # 時間特徴量 (Hour)
    df_eng['hour_sin'] = np.sin(2 * np.pi * df_eng.index.hour / 24)
    df_eng['hour_cos'] = np.cos(2 * np.pi * df_eng.index.hour / 24)
    
    # 月特徴量 (Hypothesis 1以降で使用)
    if use_month:
        df_eng['month_sin'] = np.sin(2 * np.pi * df_eng.index.month / 12)
        df_eng['month_cos'] = np.cos(2 * np.pi * df_eng.index.month / 12)

    # 選定されたラグ特徴量
    ot_lags = [1, 24, 168]
    for lag in ot_lags:
        df_eng[f'OT_lag_{lag}'] = df_eng['OT'].shift(lag)
        
    # 選定された外部要因ラグ
    exog_cols = ['HUFL', 'HULL', 'MUFL', 'MULL']
    exog_lags = [1, 33, 38]
    for col in exog_cols:
        for lag in exog_lags:
            df_eng[f'{col}_lag_{lag}'] = df_eng[col].shift(lag)

    # ターゲット変数生成
    if use_diff:
        df_eng['Target'] = df_eng[target_col].diff().shift(-1)
    else:
        df_eng['Target'] = df_eng[target_col].shift(-1)

    df_eng = df_eng.dropna()
    feature_cols = [c for c in df_eng.columns if c != 'Target' and c not in ['OT', 'HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL']]
    
    return df_eng, feature_cols

def train_and_evaluate(df_dataset, feature_cols, model_name="Model", epochs=15):
    """
    モデルの学習、予測、評価を行う共通関数
    Train/Validation/Test に分割して学習・評価を行う
    """
    # データ分割比率 (Train: 70%, Val: 10%, Test: 20%)
    n = len(df_dataset)
    train_end = int(n * 0.7)
    val_end = int(n * 0.8)
    
    train_df = df_dataset.iloc[:train_end]
    val_df = df_dataset.iloc[train_end:val_end]
    test_df = df_dataset.iloc[val_end:]
    
    # スケーリング (Data Leakage防止: Trainの統計量のみを使用)
    scaler_x = MinMaxScaler().fit(train_df[feature_cols])
    scaler_y = MinMaxScaler(feature_range=(-1, 1)).fit(train_df[['Target']]) 
    
    # データ変換
    X_train = scaler_x.transform(train_df[feature_cols]).reshape(-1, 1, len(feature_cols))
    y_train = scaler_y.transform(train_df[['Target']])
    
    X_val = scaler_x.transform(val_df[feature_cols]).reshape(-1, 1, len(feature_cols))
    y_val = scaler_y.transform(val_df[['Target']])
    
    X_test = scaler_x.transform(test_df[feature_cols]).reshape(-1, 1, len(feature_cols))
    
    # モデル構築
    model = Sequential([
        LSTM(64, activation='relu', input_shape=(1, len(feature_cols)), return_sequences=True),
        Dropout(0.2),
        LSTM(32, activation='relu'),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    
    # 学習 
    history = model.fit(
        X_train, y_train, 
        epochs=epochs, 
        batch_size=32, 
        validation_data=(X_val, y_val),
        verbose=0
    )
    
    # 学習曲線の最終Lossを表示 
    final_val_loss = history.history['val_loss'][-1]
    print(f"    (Training Info) Final Validation Loss: {final_val_loss:.6f}")
    
    # 予測 
    pred_scaled = model.predict(X_test, verbose=0)
    pred = scaler_y.inverse_transform(pred_scaled).flatten()
    
    return pred, test_df


# =============================================================================
# 3. モデル検証プロセス
# =============================================================================

# --- 3-1. Baseline: 月特徴量なし / 絶対値予測 ---
print("\n[Process] Baselineモデルの検証 (No Month Features)...")
df_base, feats_base = create_dataset(df, use_month=False, use_diff=False)
pred_base, test_df_base = train_and_evaluate(df_base, feats_base, "Baseline")

# 評価 (Testデータ)
y_true_base = test_df_base['Target'].values
rmse_base = math.sqrt(mean_squared_error(y_true_base, pred_base))
mae_base = mean_absolute_error(y_true_base, pred_base)
print(f"  Baseline Result (Test Set) -> RMSE: {rmse_base:.4f}, MAE: {mae_base:.4f}")

# --- 3-2. Hypothesis 1: 月特徴量あり / 絶対値予測 ---
print("\n[Process] Hypothesis 1 (月特徴量追加) の検証...")
df_h1, feats_h1 = create_dataset(df, use_month=True, use_diff=False)
pred_h1, test_df_h1 = train_and_evaluate(df_h1, feats_h1, "Hypothesis 1")

# 評価
y_true_h1 = test_df_h1['Target'].values
rmse_h1 = math.sqrt(mean_squared_error(y_true_h1, pred_h1))
mae_h1 = mean_absolute_error(y_true_h1, pred_h1)
print(f"  Hypothesis 1 Result (Test Set) -> RMSE: {rmse_h1:.4f}, MAE: {mae_h1:.4f}")


# --- 誤差要因分析 (Concept Drift) ---
print("  [Analysis] Analyzing error source for Hypothesis 1...")
# 元のTrain(70%)とTest(Last 20%)を比較し、油温の平均値の乖離を確認する
n = len(df_h1)
train_mean = df_h1.iloc[:int(n*0.7)]['OT'].mean()
test_mean = df_h1.iloc[int(n*0.8):]['OT'].mean()
print(f"    Train Mean: {train_mean:.2f}C, Test Mean: {test_mean:.2f}C (Diff: {train_mean - test_mean:.2f}C)")

# --- 3-3. Hypothesis 2: 月特徴量あり / 差分予測 (Proposed) ---
print("\n[Process] Hypothesis 2 (差分予測導入) の検証...")
df_h2, feats_h2 = create_dataset(df, use_month=True, use_diff=True)
pred_diff, test_df_h2 = train_and_evaluate(df_h2, feats_h2, "Hypothesis 2")

# 復元処理 (Reconstruction)
# y_pred(t) = OT(t-1) + diff_pred(t)
last_ot_values = df.loc[test_df_h2.index]['OT'].values
pred_restored = last_ot_values + pred_diff

# 正解データの取得
y_true_h2 = df.loc[test_df_h2.index]['OT'].shift(-1).dropna().values
min_len = min(len(pred_restored), len(y_true_h2))
pred_restored = pred_restored[:min_len]
y_true_h2 = y_true_h2[:min_len]

# 評価
rmse_h2 = math.sqrt(mean_squared_error(y_true_h2, pred_restored))
mae_h2 = mean_absolute_error(y_true_h2, pred_restored)
print(f"  Hypothesis 2 Result (Test Set) -> RMSE: {rmse_h2:.4f}, MAE: {mae_h2:.4f}")


# =============================================================================
# 4. 最終結果まとめ
# =============================================================================
print("\n" + "="*50)
print("【最終評価結果一覧 (Test Data)】")
print(f"1. Baseline (Basic Lags)        : RMSE={rmse_base:.4f}, MAE={mae_base:.4f}")
print(f"2. Hypothesis 1 (+Month Feats)  : RMSE={rmse_h1:.4f}, MAE={mae_h1:.4f}")
print(f"3. Hypothesis 2 (+Differencing) : RMSE={rmse_h2:.4f}, MAE={mae_h2:.4f}")
print("="*50)


print("\n[Output] 結論モデル(Hypothesis 2)の予測結果を描画...")
limit = 300
plt.figure(figsize=(15, 6))

# 実測値
plt.plot(y_true_h2[:limit], label='Actual Data (Test Set)', color='black', alpha=0.6, linewidth=1.5)
# 結論モデル予測値
plt.plot(pred_restored[:limit], label=f'Predicted (Hypothesis 2) RMSE:{rmse_h2:.2f}', color='blue', linewidth=2)

plt.title('Final Prediction Result on Test Data', fontsize=16)
plt.ylabel('Oil Temperature (OT)')
plt.legend()
plt.tight_layout()
plt.show()