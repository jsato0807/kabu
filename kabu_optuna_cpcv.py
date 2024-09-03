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
from kabu_backtest import traripi_backtest, pair, initial_funds, grid_start, grid_end, entry_intervals, total_thresholds, data, start_date, end_date
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

from typing import Any, Dict, Tuple


# データ取得関数
def fetch_currency_data(pair: str, start: str, end: str, interval: str) -> pd.DataFrame:
    """Fetch historical currency pair data from Yahoo Finance."""
    data = yf.download(pair, start=start, end=end, interval=interval)
    data = data[['Close']]
    data.columns = ['Close']  # Rename column to 'Close'
    data.index.name = 'Date'  # Set index name to 'Date'
    print(f"Fetched data length: {len(data)}")
    return data


# ベイズ最適化のための目的関数
def optimize_on_split(X_train: pd.DataFrame) -> Tuple[Dict[str, Any], float]:
    def objective(trial: optuna.Trial) -> float:
        num_trap = trial.suggest_int('num_trap', 4, 101)
        profit_width = trial.suggest_float('profit_width', 0.001, 100.0)
        order_size = trial.suggest_int('order_size', 1, 10) * 1000
        density = trial.suggest_float('density', 1.0, 10.0)
        strategy = trial.suggest_categorical('strategy', ["long_only", "short_only", "half_and_half", "diamond"])

        # トレードシミュレーションのバックテスト
        effective_margin = traripi_backtest(
            calculator,
            X_train,
            initial_funds,
            grid_start,
            grid_end,
            num_trap,
            profit_width,
            order_size,
            entry_interval=0,
            total_threshold=0,
            strategy=strategy,
            density=density
        )

        
        return effective_margin

    # Optuna スタディの設定
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)
    return study.best_params, study.best_value

# データを取得
data = fetch_currency_data('USDJPY=X', '2022-01-01', '2023-01-01', '1d')

# CombinatorialPurgedCrossValidation インスタンスの作成
cv = CombinatorialPurgedCrossValidation(n_splits=5, n_tests=2)

# 結果を保存するリスト
results = []

# 交差検証を行い、各分割でベイズ最適化
for X_train, X_test in cv.split(X=data):
    X_train = X_train['Close']
    X_test = [X_test[i]['Close'] for i in range(len(X_test))]
    X_test = pd.concat(X_test)


    print(f"Optimizing on split with {len(X_train)} training samples and {len(X_test)} test samples...")
    
    # 各分割でベイズ最適化を実行
    best_params, best_effective_margin = optimize_on_split(X_train)

    # 最適なパラメータでテストセットを評価
    final_effective_margin = traripi_backtest(
        calculator,
        X_test,
        initial_funds,
        grid_start,
        grid_end,
        best_params['num_trap'],
        best_params['profit_width'],
        best_params['order_size'],
        entry_interval=0,
        total_threshold=0,
        strategy=best_params['strategy'],
        density=best_params['density']
    )

    # 結果をリストに追加
    results.append({
        'train_size': len(X_train),
        'test_size': len(X_test),
        'best_params': best_params,
        'best_effective_margin': best_effective_margin,
        'final_effective_margin': final_effective_margin
    })

# 結果を表示
for i, result in enumerate(results):
    print(f"Split {i + 1}:")
    print(f"  Training samples: {result['train_size']}")
    print(f"  Test samples: {result['test_size']}")
    print(f"  Best parameters: {result['best_params']}")
    print(f"  Best effective margin (training): {result['best_effective_margin']}")
    print(f"  Final effective margin (test): {result['final_effective_margin']}")
    print()
