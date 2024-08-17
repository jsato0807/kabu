import numpy as np

# 目的関数の定義
def objective_function(x):
    return (x - 2)**2

# ガウス過程の実装 (非常にシンプルな例)
class GaussianProcess:
    def __init__(self, X_init, Y_init):
        self.X = X_init
        self.Y = Y_init
        self.sigma_n = 1e-10  # ノイズの小さい値

    def kernel(self, x1, x2, l=1.0):
        # RBF カーネル
        return np.exp(-0.5 * np.sum((x1 - x2)**2) / l**2)

    def predict(self, X_s):
        K = self.kernel(self.X, self.X.T)
        K_s = self.kernel(self.X, X_s.T)
        K_ss = self.kernel(X_s, X_s.T) + self.sigma_n * np.eye(len(X_s))

        K_inv = np.linalg.inv(K + self.sigma_n * np.eye(len(self.X)))

        mu_s = K_s.T.dot(K_inv).dot(self.Y)
        cov_s = K_ss - K_s.T.dot(K_inv).dot(K_s)

        return mu_s, cov_s

# 獲得関数 (期待改善)
def expected_improvement(X_s, gp, Y_best):
    mu, sigma = gp.predict(X_s)
    Z = (mu - Y_best) / sigma
    return (mu - Y_best) * norm.cdf(Z) + sigma * norm.pdf(Z)




if __name__ == "__main__":
    # 初期データ
    X_init = np.array([[1.0], [3.0]])
    Y_init = objective_function(X_init)
    
    # ガウス過程の初期化
    gp = GaussianProcess(X_init, Y_init)
    
    # ベイズ最適化のプロセス
    for i in range(10):
        # 獲得関数の最大化で次のサンプルを選択
        X_next = np.array([[np.random.uniform(0, 5)]])
        EI = expected_improvement(X_next, gp, np.min(Y_init))
        
        # 新しいサンプル点の取得
        Y_next = objective_function(X_next)
        
        # データの更新
        X_init = np.vstack((X_init, X_next))
        Y_init = np.vstack((Y_init, Y_next))
        
        # ガウス過程の更新
        gp = GaussianProcess(X_init, Y_init)
        
        # 進捗の表示
        print(f"Iteration {i+1}: X = {X_next}, Y = {Y_next}, Best Y = {np.min(Y_init)}")

