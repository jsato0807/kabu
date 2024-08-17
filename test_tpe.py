import numpy as np
from sklearn.mixture import GaussianMixture
from scipy import stats
from kabu_backtest import traripi_backtest, initial_funds, grid_start, grid_end, entry_intervals, total_thresholds, data, strategies
from kabu_swap import get_html, parse_swap_points, rename_swap_points, SwapCalculator
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold
import random

default_density = 1.0
entry_interval = entry_intervals[0]
total_threshold = total_thresholds[0]

url = 'https://fx.minkabu.jp/hikaku/moneysquare/spreadswap.html'
html = get_html(url)
swap_points = parse_swap_points(html)
swap_points = rename_swap_points(swap_points)
calculator = SwapCalculator(swap_points)

def wrapped_objective(params, train_data):
    num_trap, profit_width, order_size, strategy, density = params
    effective_margin, _, _, _, _, _, _, _, _, _ = traripi_backtest(
        calculator, train_data, initial_funds, grid_start, grid_end, num_trap, profit_width, order_size,
        entry_interval, total_threshold, strategy, density
    )
    return -effective_margin

def encode_samples(samples, encoder):
    encoded_samples = []
    for sample in samples:
        encoded_sample = []
        for i, value in enumerate(sample):
            if isinstance(value, str):
                encoded_value = encoder.transform([[value]]).flatten()
                encoded_sample.extend(encoded_value)
            else:
                encoded_sample.append(value)
        encoded_samples.append(encoded_sample)
    return encoded_samples

def decode_sample(encoded_sample, encoder, original_length):
    num_categories = len(encoder.categories_[0])
    sample = []
    index = 0

    encoded_values = encoded_sample[3:num_categories+3]
    decoded_value = encoder.categories_[0][np.argmax(encoded_values)]
    index += num_categories

    sample.extend(encoded_sample[0:3])
    sample.append(decoded_value)
    sample.append(encoded_sample[-1])

    return sample

def generate_samples_from_gmm(samples, n_samples=100):
    if len(samples) > 1:
        gmm = GaussianMixture(n_components=min(len(samples), 5), covariance_type='full')
        gmm.fit(samples)
        new_samples = gmm.sample(n_samples)[0]
        return new_samples.tolist()
    return []

def adaptive_gamma(iteration, n_iterations):
    return 0.25 + 0.5 * (iteration / n_iterations)

def adaptive_startup_trials(iteration):
    return max(5, int(10 * np.log(iteration + 1)))

def tpe_optimization(train_data, n_iterations, gamma, n_startup_trials):
    samples = [[random.randint(1, 101), random.uniform(0.001, 100), random.randint(1, 10) * 1000, random.choice(strategies), random.uniform(1.0, 10)] for _ in range(n_startup_trials)]
    
    strategy_encoder = OneHotEncoder(categories=[strategies], sparse=False)
    strategy_encoder.fit([[s] for s in strategies])
    encoders = [strategy_encoder]

    encoded_samples = encode_samples(samples, encoders[0])

    scores = [wrapped_objective(decode_sample(s, encoders[0], len(samples[0])), train_data) for s in encoded_samples]
    best_params = samples[np.argmin(scores)]
    best_score = min(scores)

    for i in range(n_iterations):
        #gamma = adaptive_gamma(i, n_iterations)
        #n_startup_trials = adaptive_startup_trials(i)

        threshold = np.percentile(scores, gamma * 100)
        mask = np.array(scores) < threshold

        encoded_good_samples = [s for s, m in zip(encoded_samples, mask) if m]
        encoded_bad_samples = [s for s, m in zip(encoded_samples, mask) if not m]

        if len(encoded_good_samples) < len(samples[0]) or len(encoded_bad_samples) < len(samples[0]):
            # サンプル数が次元数より少ない場合の補完
            if len(encoded_good_samples) < len(samples[0]):
                augmented_good_samples = generate_samples_from_gmm(encoded_good_samples, n_samples=len(samples[0]) - len(encoded_good_samples))
                encoded_good_samples.extend(augmented_good_samples)
            if len(encoded_bad_samples) < len(samples[0]):
                augmented_bad_samples = generate_samples_from_gmm(encoded_bad_samples, n_samples=len(samples[0]) - len(encoded_bad_samples))
                encoded_bad_samples.extend(augmented_bad_samples)

        good_kde = stats.gaussian_kde(np.array(encoded_good_samples).T)
        bad_kde = stats.gaussian_kde(np.array(encoded_bad_samples).T)

        new_sample = []
        for j in range(len(samples[0])):
            if j == 0:
                x = np.linspace(1, 101, 101)
            elif j == 1:
                x = np.linspace(0.001, 100, 1000)
            elif j == 2:
                x = np.linspace(100, 10000, 1000)
            elif j == 3:
                x = np.eye(len(strategies))
            elif j == 4:
                x = np.linspace(0.1, 10, 1000)

            l_x = good_kde.evaluate(x)
            g_x = bad_kde.evaluate(x)
            ratio = l_x / g_x
            ratio /= np.sum(ratio)

            new_sample.append(np.random.choice(x, p=ratio))

        new_sample_decoded = decode_sample(new_sample, encoders[0], len(samples[0]))
        new_score = wrapped_objective(new_sample_decoded, train_data)

        if new_score < best_score:
            best_params = new_sample_decoded
            best_score = new_score

        samples.append(new_sample_decoded)
        scores.append(new_score)

    return best_params, best_score

"""
def cross_validate_and_optimize(n_splits, n_iterations, gamma, n_startup_trials):
    kf = KFold(n_splits=n_splits)
    margins = []
    best_params_list = []
    for train_index, test_index in kf.split(data):
        X_train, X_test = data[train_index], data[test_index]

        best_params, best_score = tpe_optimization(X_train, n_iterations, gamma, n_startup_trials)
        best_params_list.append(best_params)

        best_num_trap = best_params[0]
        best_profit_width = best_params[1]
        best_order_size = best_params[2]
        best_strategy = best_params[3]
        best_density = best_params[4] if len(best_params) > 4 else default_density

        effective_margin, _, _, _, _, _, _, _, _, _ = traripi_backtest(
            calculator, X_test, initial_funds, grid_start, grid_end, best_num_trap, best_profit_width, best_order_size,
            entry_interval, total_threshold, best_strategy, best_density
        )
        margins.append(effective_margin)

    return best_params_list[np.argmax(margins)]

"""



from sklearn.model_selection import KFold
import numpy as np

# 内側のクロスバリデーションの設定
inner_cv = KFold(n_splits=3, shuffle=True, random_state=1)

# 外側のクロスバリデーションの設定
outer_cv = KFold(n_splits=5, shuffle=True, random_state=1)

# ハイパーパラメータの範囲
param_grid = {
    'n_trials': [10, 20, 30],
    'gammas': [0.25, 0.30, 0.35, 0.40, 0.45, 0.50],
    'n_startup_trials': [10,20,30]

}

def nested_cross_validation(X, param_grid):
    outer_scores = []
    outer_best_params = []

    for train_idx, test_idx in outer_cv.split(X):
        X_train, X_test = X[train_idx], X[test_idx]

        best_inner_score = -np.inf
        best_inner_params = None

        for n_trials in param_grid['n_trials']:
            for gammas in param_grid['gammas']:
                for n_startup_trials in param_grid['n_startup_trials']:
                    inner_scores = []

                    for inner_train_idx, val_idx in inner_cv.split(X_train):
                        X_train_inner = X_train[inner_train_idx]
                        X_val = X_train[val_idx]

                        inner_best_params, inner_best_score = tpe_optimization(X_train_inner, n_trials, gammas, n_startup_trials)
                        
                        # 検証データでの評価
                        val_effective_margin, _, _, _, _, _, _, _, _, _ = traripi_backtest(
                            calculator, X_val, initial_funds, grid_start, grid_end, 
                            inner_best_params[0], inner_best_params[1], inner_best_params[2], 
                            entry_interval, total_threshold, inner_best_params[3], inner_best_params[4]
                        )
                        inner_scores.append(-val_effective_margin)

                    mean_inner_score = np.mean(inner_scores)

                    if abs(mean_inner_score) > abs(best_inner_score):
                        best_inner_score = mean_inner_score
                        best_inner_params = {
                            'n_trials': n_trials, 
                            'gammas': gammas, 
                            'n_startup_trials': n_startup_trials
                        }

        # 最良パラメータでモデルをトレーニングし、外側のテストデータで評価
        final_best_params, final_score = tpe_optimization(X_train, n_trials=best_inner_params['n_trials'], gammas=best_inner_params['gammas'], n_startup_trials=best_inner_params['n_startup_trials'])
        
        # テストデータでの評価
        test_effective_margin, _, _, _, _, _, _, _, _, _ = traripi_backtest(
            calculator, X_test, initial_funds, grid_start, grid_end, 
            final_best_params[0], final_best_params[1], final_best_params[2], 
            entry_interval, total_threshold, final_best_params[3], final_best_params[4]
        )
        outer_scores.append(-test_effective_margin)
        outer_best_params.append(final_best_params)

    # 最良パラメータの選定
    overall_best_params = max(set(tuple(params.items()) for params in outer_best_params), key=outer_best_params.count)
    overall_best_params = dict(overall_best_params)

    return np.mean(outer_scores), np.std(outer_scores), overall_best_params

    return np.mean(outer_scores), np.std(outer_scores), overall_best_params




if __name__ == "__main__":
    #cross_validate_and_optimize(10, 50, 0.25, 50)
    
     # ネストクロスバリデーションの実行
    mean_score, std_score, best_params = nested_cross_validation(data, param_grid)
    print(f"Nested CV Score: {mean_score:.4f} ± {std_score:.4f}")
    print(f"Best Parameters: {best_params}")
