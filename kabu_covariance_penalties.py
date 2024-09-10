import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from scipy import linalg
from scipy.stats import laplace

# ランダムなリターンデータの生成
np.random.seed(42)
# パラメータ設定
n_samples = 1000
loc = 2000000  # 分布の中心を設定
scale = 1  # スケールパラメータ (標準偏差)

# ラプラス分布から確率変数を生成
returns = laplace.rvs(loc=loc, scale=scale, size=n_samples)

# ラグを用いた信号の生成
def create_lags(data, lags):
    lagged_data = pd.DataFrame({f'lag_{i}': data.shift(i) for i in range(1, lags + 1)})
    return lagged_data

# インプライド調整シャープレシオの計算
def imp_sharpe_ratio(X, y, predicted_returns):
    cov_matrix = np.cov(X.T)
    trace_M = np.trace(cov_matrix)  # 自由度 (行列のトレース)
    
    rho_in_squared = np.corrcoef(y, predicted_returns)[0, 1]**2
    N = len(y)  # データ数
    rho_out_squared = rho_in_squared - (2 / N) * trace_M
    return rho_out_squared

# MADの計算
def mad(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

# 相関係数の計算
def correlation(y_true, y_pred):
    return np.corrcoef(y_true, y_pred)[0, 1]

# MSEの計算
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# MAEの計算
def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

# AICの計算
def calculate_aic(model, y, y_pred):
    residuals = y - y_pred
    n = len(y)
    k = len(model.params)
    rss = np.sum(residuals**2)
    return n * np.log(rss / n) + 2 * k

# R-squaredの計算
def calculate_r_squared(y_true, y_pred):
    ss_total = np.sum((y_true - np.mean(y_true))**2)
    ss_residual = np.sum((y_true - y_pred)**2)
    return 1 - (ss_residual / ss_total)

# 評価指標を計算する関数
def evaluate_model(X_train, y_train, X_test, y_test, model_type):
    if model_type == 'ols':
        model = sm.OLS(y_train, X_train).fit()
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        aic_value = calculate_aic(model, y_train, y_pred_train)
        r_squared = calculate_r_squared(y_test, y_pred_test)
        imp_sharpe = imp_sharpe_ratio(X_test, y_test, y_pred_test)
        sr_p = np.mean(y_pred_test) / np.std(y_pred_test) if np.std(y_pred_test) != 0 else np.nan
        
        return aic_value, imp_sharpe, sr_p, r_squared
    
    elif model_type == 'tls':
        # TLSモデルの評価
        A_train = np.hstack([X_train, y_train.values.reshape(-1, 1)])
        U, s, Vt = linalg.svd(A_train, full_matrices=False)
        s[-1] = 0
        A_new = U @ np.diag(s) @ Vt
        X_tls = A_new[:, :-1]
        y_tls = A_new[:, -1]

        tls_beta = linalg.lstsq(X_tls, y_tls, lapack_driver='gelsy')[0]
        y_pred_test = X_test @ tls_beta
        
        aic_value = None  # TLSの場合、AICは適用しない
        r_squared = calculate_r_squared(y_test, y_pred_test)
        imp_sharpe = imp_sharpe_ratio(X_test, y_test, y_pred_test)
        sr_p = np.mean(y_pred_test) / np.std(y_pred_test) if np.std(y_pred_test) != 0 else np.nan
        
        return aic_value, imp_sharpe, sr_p, r_squared
    
    else:
        raise ValueError("Unsupported model type. Use 'ols' or 'tls'.")

# 全てのラグに対する評価を行う
from sklearn.model_selection import TimeSeriesSplit

def evaluate_lags(lags_list, n_splits=5):
    best_aic_lag = None
    best_imp_sharpe_lag = None
    best_sr_lag = None
    best_r_squared_lag = None

    best_aic = np.inf
    best_imp_sharpe = -np.inf
    best_sr = -np.inf
    best_r_squared = -np.inf

    tscv = TimeSeriesSplit(n_splits=n_splits)

    for lag in lags_list:
        # ラグを使ってデータを準備
        signals = create_lags(pd.Series(returns), lag)
        signals['returns'] = returns
        signals.dropna(inplace=True)
        
        X = signals.drop(columns=['returns'])
        y = signals['returns']
        
        for train_index, test_index in tscv.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            # OLSモデルの評価
            aic_value, imp_sharpe, sr_p, r_squared = evaluate_model(X_train, y_train, X_test, y_test, 'ols')
            if aic_value is not None and aic_value < best_aic:
                best_aic = aic_value
                best_aic_lag = lag

            if imp_sharpe > best_imp_sharpe:
                best_imp_sharpe = imp_sharpe
                best_imp_sharpe_lag = lag

            if sr_p > best_sr:
                best_sr = sr_p
                best_sr_lag = lag

            if r_squared > best_r_squared:
                best_r_squared = r_squared
                best_r_squared_lag = lag

            # TLSモデルの評価
            _, imp_sharpe_tls, sr_p_tls, r_squared_tls = evaluate_model(X_train, y_train, X_test, y_test, 'tls')
            
            if imp_sharpe_tls > best_imp_sharpe:
                best_imp_sharpe = imp_sharpe_tls
                best_imp_sharpe_lag = lag

            if sr_p_tls > best_sr:
                best_sr = sr_p_tls
                best_sr_lag = lag

            if r_squared_tls > best_r_squared:
                best_r_squared = r_squared_tls
                best_r_squared_lag = lag

    return best_aic_lag, best_imp_sharpe_lag, best_sr_lag, best_r_squared_lag


# ラグのリスト
lags_list = [3, 5, 7, 9, 12, 15, 18, 21, 26, 31, 36, 42, 49, 56, 63, 84, 105, 126]

# 評価を実行
best_aic_lag, best_imp_sharpe_lag, best_sr_lag, best_r_squared_lag = evaluate_lags(lags_list)

print(f'最良のラグ（AIC最小）: {best_aic_lag}')
print(f'最良のラグ（インプライドシャープレシオ最大）: {best_imp_sharpe_lag}')
print(f'最良のラグ（シャープレシオ最大）: {best_sr_lag}')
print(f'最良のラグ（R-squared最大）: {best_r_squared_lag}')

# インサンプルでのフィッティングとアウトサンプルでのテスト
# インサンプルでのフィッティングとアウトサンプルでのテスト
def fit_and_test_best_model(best_lag, model_type):
    signals_best = create_lags(pd.Series(returns), best_lag)
    signals_best['returns'] = returns
    signals_best.dropna(inplace=True)
    
    X_best = signals_best.drop(columns=['returns'])
    y_best = signals_best['returns']
    
    X_train_best, X_test_best, y_train_best, y_test_best = train_test_split(X_best, y_best, test_size=0.2, random_state=42)
    
    # OLS or TLSで学習
    if model_type == 'ols':
        # OLSによるアウトサンプル予測
        aic_value, imp_sharpe, sr_p, r_squared = evaluate_model(X_train_best, y_train_best, X_test_best, y_test_best, 'ols')
        y_pred_test = X_test_best @ np.linalg.lstsq(X_train_best, y_train_best, rcond=None)[0]
    
    elif model_type == 'tls':
        # TLSによるアウトサンプル予測
        A_train = np.hstack([X_train_best, y_train_best.values.reshape(-1, 1)])
        U, s, Vt = linalg.svd(A_train, full_matrices=False)
        s[-1] = 0  # 最小特異値を0に設定
        A_new = U @ np.diag(s) @ Vt
        X_tls = A_new[:, :-1]
        y_tls = A_new[:, -1]

        # TLSによる係数の推定
        tls_beta = linalg.lstsq(X_tls, y_tls, lapack_driver='gelsy')[0]
        y_pred_test = X_test_best @ tls_beta

    # テスト評価
    sr_test = np.mean(y_pred_test) / np.std(y_pred_test) if np.std(y_pred_test) != 0 else np.nan
    correlation_value = correlation(y_test_best, y_pred_test)
    mad_value = mad(y_test_best, y_pred_test)
    
    return sr_test, mad_value, correlation_value


# 最良のラグでのフィッティングとテスト
sr_test_aic, mad_test_aic, corr_test_aic = fit_and_test_best_model(best_aic_lag, 'ols')
sr_test_imp_sharpe, mad_test_imp_sharpe, corr_test_imp_sharpe = fit_and_test_best_model(best_imp_sharpe_lag, 'ols')
sr_test_sr, mad_test_sr, corr_test_sr = fit_and_test_best_model(best_sr_lag, 'ols')
sr_test_r_squared, mad_test_r_squared, corr_test_r_squared = fit_and_test_best_model(best_r_squared_lag, 'ols')

sr_test_aic_tls, mad_test_aic_tls, corr_test_aic_tls = fit_and_test_best_model(best_aic_lag, 'tls')
sr_test_imp_sharpe_tls, mad_test_imp_sharpe_tls, corr_test_imp_sharpe_tls = fit_and_test_best_model(best_imp_sharpe_lag, 'tls')
sr_test_sr_tls, mad_test_sr_tls, corr_test_sr_tls = fit_and_test_best_model(best_sr_lag, 'tls')
sr_test_r_squared_tls, mad_test_r_squared_tls, corr_test_r_squared_tls = fit_and_test_best_model(best_r_squared_lag, 'tls')

print(f'最良のラグ（AIC最小）のインサンプルシャープレシオ (OLS): {sr_test_aic}')
print(f'最良のラグ（AIC最小）のインサンプルMAD (OLS): {mad_test_aic}')
print(f'最良のラグ（AIC最小）のインサンプル相関係数 (OLS): {corr_test_aic}')

print(f'最良のラグ（インプライドシャープレシオ最大）のインサンプルシャープレシオ (OLS): {sr_test_imp_sharpe}')
print(f'最良のラグ（インプライドシャープレシオ最大）のインサンプルMAD (OLS): {mad_test_imp_sharpe}')
print(f'最良のラグ（インプライドシャープレシオ最大）のインサンプル相関係数 (OLS): {corr_test_imp_sharpe}')

print(f'最良のラグ（シャープレシオ最大）のインサンプルシャープレシオ (OLS): {sr_test_sr}')
print(f'最良のラグ（シャープレシオ最大）のインサンプルMAD (OLS): {mad_test_sr}')
print(f'最良のラグ（シャープレシオ最大）のインサンプル相関係数 (OLS): {corr_test_sr}')

print(f'最良のラグ（R-squared最大）のインサンプルシャープレシオ (OLS): {sr_test_r_squared}')
print(f'最良のラグ（R-squared最大）のインサンプルMAD (OLS): {mad_test_r_squared}')
print(f'最良のラグ（R-squared最大）のインサンプル相関係数 (OLS): {corr_test_r_squared}')

print(f'最良のラグ（AIC最小）のインサンプルシャープレシオ (TLS): {sr_test_aic_tls}')
print(f'最良のラグ（AIC最小）のインサンプルMAD (TLS): {mad_test_aic_tls}')
print(f'最良のラグ（AIC最小）のインサンプル相関係数 (TLS): {corr_test_aic_tls}')

print(f'最良のラグ（インプライドシャープレシオ最大）のインサンプルシャープレシオ (TLS): {sr_test_imp_sharpe_tls}')
print(f'最良のラグ（インプライドシャープレシオ最大）のインサンプルMAD (TLS): {mad_test_imp_sharpe_tls}')
print(f'最良のラグ（インプライドシャープレシオ最大）のインサンプル相関係数 (TLS): {corr_test_imp_sharpe_tls}')

print(f'最良のラグ（シャープレシオ最大）のインサンプルシャープレシオ (TLS): {sr_test_sr_tls}')
print(f'最良のラグ（シャープレシオ最大）のインサンプルMAD (TLS): {mad_test_sr_tls}')
print(f'最良のラグ（シャープレシオ最大）のインサンプル相関係数 (TLS): {corr_test_sr_tls}')

print(f'最良のラグ（R-squared最大）のインサンプルシャープレシオ (TLS): {sr_test_r_squared_tls}')
print(f'最良のラグ（R-squared最大）のインサンプルMAD (TLS): {mad_test_r_squared_tls}')
print(f'最良のラグ（R-squared最大）のインサンプル相関係数 (TLS): {corr_test_r_squared_tls}')
