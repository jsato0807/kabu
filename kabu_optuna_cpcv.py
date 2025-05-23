import optuna
import numpy as np
import pandas as pd
from typing import List, Tuple, Union
import math
from datetime import datetime
from dataclasses import dataclass
from functools import cached_property
from itertools import combinations
from typing import Iterable, List, Optional, Union
from numpy.typing import NDArray
from kabu_swap import get_html, parse_swap_points, rename_swap_points, SwapCalculator
from kabu_backtest import traripi_backtest, pair, initial_funds, grid_start, grid_end, entry_intervals, total_thresholds, data, start_date, end_date, interval
import yfinance as yf

# パラメータの設定
entry_interval = entry_intervals
total_threshold = total_thresholds

# Swapポイントの取得と計算
url = 'https://fx.minkabu.jp/hikaku/moneysquare/spreadswap.html'
html = get_html(url)
swap_points = parse_swap_points(html)
swap_points = rename_swap_points(swap_points)
calculator = SwapCalculator(swap_points)


@dataclass
class CombinatorialPurgedCrossValidation:
    """Combinatorial purged cross-validation."""

    n_splits: int = 5
    n_tests: int = 2
    purge_gap: Union[int, pd.Timedelta] = 0
    embargo_gap: Union[int, pd.Timedelta] = 0

    def __post_init__(self) -> None:
        """Post process."""
        if self.n_tests >= self.n_splits:
            raise ValueError("n_tests must be greater than n_splits.")

    @cached_property
    def _test_sets(self) -> List[List[int]]:
        """Return sets of group index for testing."""
        test_sets = []
        for comb in combinations(range(self.n_splits), self.n_tests):
            test_sets.append(list(comb))
        return test_sets

    @cached_property
    def _train_sets(self) -> List[List[int]]:
        """Return sets of group index for training."""
        train_sets = []
        for test_set in self._test_sets:
            train_sets.append(np.setdiff1d(np.arange(self.n_splits), test_set).tolist())
        return train_sets

    @cached_property
    def pathway_labeled(self) -> NDArray[np.integer]:
        """Labeled backtest pathways."""
        n_combs = math.comb(self.n_splits, self.n_tests)

        pathway_flags = np.zeros((n_combs, self.n_splits), bool)
        for i, comb in enumerate(combinations(range(self.n_splits), self.n_tests)):
            pathway_flags[i, comb] = True
        pathway_labeled = pathway_flags.cumsum(axis=0)
        pathway_labeled[~pathway_flags] = 0
        return pathway_labeled

    @cached_property
    def test_set_labels(self) -> List[List[int]]:
        """Return labels of test sets."""
        return [labels[labels > 0].tolist() for labels in self.pathway_labeled]

    def _is_valid_shape(
        self,
        X: Union[NDArray[np.floating], pd.DataFrame],
    ) -> None:
        if X.ndim != 2:
            raise ValueError("X.ndim must be 2.")

    def _is_valid(
        self,
        X: Union[NDArray[np.floating], pd.DataFrame],
    ) -> None:
        if X.ndim != 2:
            raise ValueError("X.ndim must be 2.")

    def _is_valid_gap_purge(self, X: pd.DataFrame) -> None:
        if isinstance(self.purge_gap, int):
            return
        if not isinstance(self.purge_gap, type(X.index[1] - X.index[0])):
            raise ValueError(
                "The type of purge_gap and the type of difference "
                "of index in X must be same."
            )

    def _is_valid_gap_embargo(self, X: pd.DataFrame) -> None:
        if isinstance(self.embargo_gap, int):
            return
        if not isinstance(self.embargo_gap, type(X.index[1] - X.index[0])):
            raise ValueError(
                "The type of embargo_gap and the type of difference "
                "of index in X must be same."
            )

    def purge(self, indices: pd.Index) -> pd.Index:
        if isinstance(self.purge_gap, int):
            return indices[: -self.purge_gap]

        flags = indices <= (indices.max() - self.purge_gap)
        return indices[flags]

    def embargo(self, indices: pd.Index) -> pd.Index:
        if isinstance(self.embargo_gap, int):
            return indices[self.embargo_gap :]
        flags = indices >= (indices.min() + self.embargo_gap)
        return indices[flags]

    def split(
        self,
        X: Union[NDArray[np.floating], pd.DataFrame],
        y: Optional[Union[NDArray[np.floating], pd.DataFrame, pd.Series]] = None,
        return_backtest_labels: bool = False,
    ) -> Iterable:
        """Split data.

        Parameters
        ----------
        X : (N, M) Union[NDArray[np.floating], pd.DataFrame]
            Explanatory variables to split, where N is number of data,
            M is number of features.
        y : (N,) Union[NDArray[np.floating], pd.DataFrame, pd.Series]
            Objective variables to split, where N is number of data.
        return_backtest_labels : bool, by default False.
            If True, return labels test set on backtest path.

        Returns
        -------
        Iterable that generate (Xtrain, ytrain, Xtest, ytest[, labels]) if y was given.
        If y wasn't given, this generates (Xtrain, Xtest[, labels]).
        """
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        if isinstance(y, np.ndarray):
            y = pd.DataFrame(y)
        self._is_valid_shape(X)
        self._is_valid_gap_purge(X)
        self._is_valid_gap_embargo(X)

        inds_unique = X.index.unique()

        inds_unique_splitted = np.array_split(inds_unique, self.n_splits)

        for train_gids, test_gids, labels in zip(
            self._train_sets, self._test_sets, self.test_set_labels
        ):
            inds_to_purge = np.array(test_gids) - 1
            inds_to_embargo = np.array(test_gids) + 1

            test_inds_list = [inds_unique_splitted[gid] for gid in test_gids]

            train_inds_list = []
            for gid in train_gids:
                inds = inds_unique_splitted[gid]
                if gid in inds_to_purge:
                    inds = self.purge(inds)
                if gid in inds_to_embargo:
                    inds = self.embargo(inds)
                train_inds_list.append(inds)

            train_inds = np.concatenate(train_inds_list).ravel()

            if y is None:
                if return_backtest_labels:
                    yield (
                        X.loc[train_inds],
                        [X.loc[inds] for inds in test_inds_list],
                        labels,
                    )
                else:
                    yield X.loc[train_inds], [X.loc[inds] for inds in test_inds_list]
            else:
                if return_backtest_labels:
                    yield (
                        X.loc[train_inds],
                        y.loc[train_inds],
                        [X.loc[inds] for inds in test_inds_list],
                        [y.loc[inds] for inds in test_inds_list],
                        labels,
                    )
                else:
                    yield (
                        X.loc[train_inds],
                        y.loc[train_inds],
                        [X.loc[inds] for inds in test_inds_list],
                        [y.loc[inds] for inds in test_inds_list],
                    )


# データ取得関数
def fetch_currency_data(pair: str, start: str, end: str, interval: str) -> pd.DataFrame:
    """Fetch historical currency pair data from Yahoo Finance."""
    data = yf.download(pair, start=start, end=end, interval=interval)
    data = data[['Close']]
    data.columns = ['Close']  # Rename column to 'Close'
    data.index.name = 'Date'  # Set index name to 'Date'
    print(f"Fetched data length: {len(data)}")
    return data


# wrapped_objective_functionの定義
def wrapped_objective_function(params, train_data):
    num_trap, profit_width, order_size, strategy, density = params
    
    # traripi_backtestをtrain_dataに対して実行し、effective_marginを計算
    effective_margin, _, _, _, _, _, _, _, _, _ = traripi_backtest(
        calculator, train_data, initial_funds, grid_start, grid_end, num_trap, profit_width, order_size,
        entry_interval, total_threshold, strategy, density
    )
    
    return effective_margin


# ベイズ最適化のための目的関数

def objective(trial, X_train):
    num_trap = trial.suggest_int('num_trap', 4, 101)
    profit_width = trial.suggest_float('profit_width', 0.001, 100.0)
    order_size = trial.suggest_int('order_size', 1, 10) * 1000
    density = trial.suggest_float('density', 1.0, 10.0)
    strategy = trial.suggest_categorical('strategy', ["long_only", "short_only", "half_and_half", "diamond"])

    params = [num_trap, profit_width, order_size, strategy, density]


        
    return wrapped_objective_function(params, X_train)

def idx_of_the_nearest(data, value):
    idx = np.argmin(np.abs(np.array(data) - value))
    return idx


if __name__ == "__main__":

    # データを取得
    pair = "AUDNZD=X"
    end_date = datetime.strptime("2024-01-01", "%Y-%m-%d")
    start_date = datetime.strptime("2019-06-01", "%Y-%m-%d")
    data = fetch_currency_data(pair, start_date, end_date, interval)

    # CombinatorialPurgedCrossValidation インスタンスの作成
    cv = CombinatorialPurgedCrossValidation(n_splits=10, n_tests=4)

    # 結果を保存するリスト
    results = []

    # 交差検証を行い、各分割でベイズ最適化
    for X_train, X_test in cv.split(X=data):
        X_train = X_train['Close']
        X_test = [X_test[i]['Close'] for i in range(len(X_test))]
        X_test = pd.concat(X_test)


        print(f"Optimizing on split with {len(X_train)} training samples and {len(X_test)} test samples...")

        # 各分割でベイズ最適化を実行
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: objective(trial, X_train), n_trials=10)

        best_params = study.best_params
        best_effective_margin = study.best_value

        # 最適なパラメータでテストセットを評価
        final_effective_margin = wrapped_objective_function([
            best_params['num_trap'],
            best_params['profit_width'],
            best_params['order_size'],
            best_params['strategy'],
            best_params['density']
        ], X_test)

        # 結果をリストに追加
        results.append({
            'train_size': len(X_train),
            'test_size': len(X_test),
            'best_params': best_params,
            'best_effective_margin': best_effective_margin,
            'final_effective_margin': final_effective_margin
        })

    # 結果を表示
# 結果を表示
with open(f"./txt_dir/kabu_optuna_cpcv_{pair}_{start_date}_{end_date}.txt", "w") as f:
    final_effective_margins = [result['final_effective_margin'] for result in results]
    
    for i, result in enumerate(results):
        f.write(f"Split {i + 1}:\n")
        f.write(f"  Training samples: {result['train_size']}\n")
        f.write(f"  Test samples: {result['test_size']}\n")
        f.write(f"  Best parameters: {result['best_params']}\n")
        f.write(f"  Best effective margin (training): {result['best_effective_margin']}\n")
        f.write(f"  Final effective margin (test): {result['final_effective_margin']}\n")
        f.write("\n")
    
    # 平均と標準偏差を計算して書き出す
    f.write(f"Final effective margin Mean (test): {np.mean(final_effective_margins)}\n")
    f.write(f"Final effective margin Standard Deviation (test): {np.std(final_effective_margins)}\n")
    
    # 平均に最も近いfinal_effective_marginを見つけて書き出す
    idx = idx_of_the_nearest(final_effective_margins, np.mean(final_effective_margins))
    f.write(f"Nearest effective margin to mean: {final_effective_margins[idx]}\n")
    f.write(f"Nearest effective margin params to mean: {results[idx]}\n")
    
