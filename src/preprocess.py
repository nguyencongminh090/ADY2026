"""
preprocess.py
=============
Time-series preprocessing pipeline for the Zillow Metro ZHVI dataset.
Processes ALL metro regions simultaneously.

Provides an pipeline for data ingestion, missing value
imputation, feature engineering (log returns and lags), dataset splitting,
and train-dependent transformations (target encoding and scaling).
"""

from __future__ import annotations

import pathlib
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths & Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT  = pathlib.Path(__file__).resolve().parent.parent
RAW_FILE      = PROJECT_ROOT / "data" / "raw" / "Metro_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

META_COLS       = ["RegionID", "SizeRank", "RegionName", "RegionType", "StateName"]
TRAIN_RATIO     = 0.80
LAG_PERIODS     = [1, 2, 3, 6, 12]

LAG_COLS     = [f"lag_{l}" for l in LAG_PERIODS]
CITY_ENC_COL = "city_enc"
FEATURE_COLS = LAG_COLS + [CITY_ENC_COL]
TARGET_COL   = "log_return"


class DataPreprocessor:
    """Data pipeline for Zillow ZHVI data.

    This class handles the end-to-end preprocessing, ensuring that
    target variables and lag features are calculated before time-splitting,
    while scaler and city configurations are correctly fitted post-split
    to prevent target leakage.

    Attributes:
        raw_path (pathlib.Path): Path to the raw dataset.
        processed_dir (pathlib.Path): Path to output directory.
        train_ratio (float): Ratio for train/test split.
        lag_periods (list[int]): List of periods for generating lag features.
        scaler (StandardScaler): Fitted sklearn standard scaler.
        city_mean (pd.Series): Fitted target-encoded mean values for cities.
        global_mean (float): Global target mean for fallback encoding.
    """

    def __init__(
        self,
        raw_path     : pathlib.Path     = RAW_FILE,
        processed_dir: pathlib.Path     = PROCESSED_DIR,
        train_ratio  : float            = TRAIN_RATIO,
        lag_periods  : list[int] | None = None,
    ):
        """Initializes the DataPreprocessor.

        Args:
            raw_path: File path of the raw CSV.
            processed_dir: Directory where processed CSVs will be saved.
            train_ratio: Float indicating the split ratio for training data.
            lag_periods: List of lag periods. Defaults to [1, 2, 3, 6, 12].
        """
        self.raw_path      = raw_path
        self.processed_dir = processed_dir
        self.train_ratio   = train_ratio
        self.lag_periods   = lag_periods or LAG_PERIODS
        self.scaler        = StandardScaler()
        self.city_mean  : pd.Series | None = None
        self.global_mean: float     | None = None

    def load_data(self) -> pd.DataFrame:
        """Loads and filters raw data to retain only MSA rows.

        Returns:
            pd.DataFrame: A dataframe containing only 'msa' region records.
        """
        print(f"[1] Loading raw data from: {self.raw_path}")
        df = pd.read_csv(self.raw_path, low_memory=False, skipinitialspace=True)
        # Strip trailing whitespaces (caused by skipinitialspace leaving them behind)
        df.columns = df.columns.str.strip()
        for col in df.select_dtypes(['object']).columns:
            df[col] = df[col].str.strip()
            
        print(f"    Raw shape : {df.shape}")

        before  = len(df)
        df      = df[df["RegionType"] == "msa"].reset_index(drop=True)
        dropped = before - len(df)
        if dropped:
            print(f"    Dropped {dropped} non-MSA row(s) ('United States' country aggregate)")

        print(f"    MSA regions: {df['RegionName'].nunique()}")
        return df

    def melt_regions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Melts wide-format date columns into a long format.

        Args:
            df: Wide-format dataframe containing region data.

        Returns:
            pd.DataFrame: Long-format dataframe with 'date' and 'price' columns.
        """
        print("\n[2] Melting all regions: wide → long format …")

        keep_meta = ["RegionID", "RegionName", "StateName"]
        date_cols = [c for c in df.columns if c not in META_COLS]

        long = df[keep_meta + date_cols].melt(
            id_vars   =keep_meta,
            value_vars=date_cols,
            var_name  ="date",
            value_name="price",
        )
        long["date"] = pd.to_datetime(long["date"])
        long["price"] = pd.to_numeric(long["price"], errors="coerce")
        long = long.sort_values(["RegionID", "date"]).reset_index(drop=True)

        print(f"    Long shape : {long.shape}")
        print(f"    Regions    : {long['RegionName'].nunique()}")
        print(f"    Missing    : {long['price'].isna().sum():,}")
        return long

    def handle_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Interpolates missing price values per region.

        Args:
            df: Long-format dataframe with a 'price' column.

        Returns:
            pd.DataFrame: Dataframe with time-series interpolated prices.
        """
        print("\n[3] Interpolating missing values per region …")
        before = df["price"].isna().sum()

        def _interpolate_group(grp: pd.DataFrame) -> pd.DataFrame:
            grp = grp.copy().set_index("date")
            grp["price"] = (
                grp["price"]
                .interpolate(method="time", limit_direction="both")
                .ffill()
                .bfill()
            )
            return grp.reset_index()

        df = (
            df.groupby("RegionID", group_keys=False)
            .apply(_interpolate_group)
            .reset_index(drop=True)
        )

        print(f"    Missing before : {before:,} -> after : {df['price'].isna().sum():,}")
        return df

    def compute_targets_and_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Computes log returns and lag features for the entire dataset before splitting.

        Because shift-based lag features do not intrinsically leak future values
        when applied strictly backwards, computing them prior to the train/test
        split simplifies pipeline logic significantly.

        Args:
            df: Cleaned long-format dataframe.

        Returns:
            pd.DataFrame: Dataframe augmented with log return targets and lag features.
        """
        print("\n[4] Computing log return (target) and lag features per region …")

        # 1. Compute target
        df = df.copy()
        df[TARGET_COL] = (
            df.groupby("RegionID")["price"]
            .transform(lambda s: np.log(s / s.shift(1)))
        )

        before = len(df)
        df     = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)
        print(f"    Dropped {before - len(df):,} first-observation rows (missing targets).")

        # 2. Compute Lags
        def _add_lag_group(grp: pd.DataFrame) -> pd.DataFrame:
            grp = grp.copy().sort_values("date")
            for lag in self.lag_periods:
                grp[f"lag_{lag}"] = grp[TARGET_COL].shift(lag)
            return grp

        df = (
            df.groupby("RegionID", group_keys=False)
            .apply(_add_lag_group)
            .reset_index(drop=True)
        )

        # Drop rows with NaN from the maximum lag warm-up
        n_before = len(df)
        lag_cols = [f"lag_{l}" for l in self.lag_periods]
        df       = df.dropna(subset=lag_cols).reset_index(drop=True)
        print(f"    Dropped {n_before - len(df):,} warm-up rows due to lag generation.")

        return df

    def time_split(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Splits the dataframe into training and testing sets by time cutoff.

        Args:
            df: The complete, preprocessed dataframe spanning all dates.

        Returns:
            A tuple containing (train_df, test_df).
        """
        print(f"\n[5] Time-based split (train={self.train_ratio:.0%} / test={1-self.train_ratio:.0%})")

        all_dates   = df["date"].sort_values().unique()
        cutoff_date = all_dates[int(len(all_dates) * self.train_ratio)]

        train = df[df["date"] < cutoff_date].copy().reset_index(drop=True)
        test  = df[df["date"] >= cutoff_date].copy().reset_index(drop=True)

        print(f"    Cutoff date : {pd.Timestamp(cutoff_date).date()}")
        print(f"    Train size  : {len(train):,} rows")
        print(f"    Test size   : {len(test):,} rows")
        return train, test

    def apply_train_dependent_transforms(
        self, train: pd.DataFrame, test: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Applies operations that must be fitted exclusively on the training set.

        This includes City Target-Encoding and Standard Scaling, which are fitted
        on the training data and transform both training and testing environments,
        strictly preventing any target leakage.

        Args:
            train: Training dataframe.
            test: Testing dataframe.

        Returns:
            A tuple containing scaled (train_df, test_df).
        """
        print("\n[6] Applying train-dependent transforms (City Encoding & Scaling) …")

        # City Encoding
        self.city_mean   = train.groupby("RegionID")[TARGET_COL].mean()
        self.global_mean = float(train[TARGET_COL].mean())

        train[CITY_ENC_COL] = train["RegionID"].map(self.city_mean)
        test [CITY_ENC_COL] = test["RegionID"].map(self.city_mean).fillna(self.global_mean)

        # Standard Scaling
        feature_cols = [f"lag_{l}" for l in self.lag_periods] + [CITY_ENC_COL]
        train[feature_cols] = self.scaler.fit_transform(train[feature_cols])
        test [feature_cols] = self.scaler.transform(test[feature_cols])

        return train, test

    def save_processed(self, train: pd.DataFrame, test: pd.DataFrame) -> None:
        """Saves the output processed CSV files.

        Args:
            train: Fully processed training dataframe.
            test: Fully processed testing dataframe.
        """
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        train_path = self.processed_dir / "train.csv"
        test_path  = self.processed_dir / "test.csv"

        train.to_csv(train_path, index=False)
        test.to_csv(test_path, index=False)

        print(f"\n[7] Saved processed files to {self.processed_dir}:")
        print(f"    → train.csv ({train_path.stat().st_size / 1e6:.1f} MB)")
        print(f"    → test.csv  ({test_path.stat().st_size / 1e6:.1f} MB)")

    def run(self) -> dict:
        """Executes the full preprocessing pipeline in order.

        Returns:
            dict: Result container featuring final train/test matrices,
                  the scaler, and feature definitions.
        """
        print("=" * 60)
        print("  Preprocessing Pipeline — ALL metro regions")
        print("=" * 60)

        df = self.load_data()
        df = self.melt_regions(df)
        df = self.handle_missing(df)
        df = self.compute_targets_and_features(df)
        
        train, test = self.time_split(df)
        train, test = self.apply_train_dependent_transforms(train, test)

        self.save_processed(train, test)

        feature_cols = [f"lag_{l}" for l in self.lag_periods] + [CITY_ENC_COL]

        X_train = train[feature_cols]
        y_train = train[TARGET_COL]
        X_test  = test[feature_cols]
        y_test  = test[TARGET_COL]

        print("\n[8] Final dataset shapes:")
        print(f"     X_train : {X_train.shape}")
        print(f"     X_test  : {X_test.shape}")
        print(f"     y_train : {y_train.shape}  (log_return, unscaled)")
        print(f"     y_test  : {y_test.shape}   (log_return, unscaled)")

        # Recombine to save the full representation maintaining processed values
        final_long_df = pd.concat([train, test], ignore_index=True)
        final_long_df = final_long_df.sort_values(["RegionID", "date"]).reset_index(drop=True)
        final_long_df.to_csv('processed_data.csv', index=False)

        print(f"\n✅ Pipeline complete. Features: {feature_cols}\n")

        return {
            "X_train"     : X_train,
            "X_test"      : X_test,
            "y_train"     : y_train,
            "y_test"      : y_test,
            "scaler"      : self.scaler,
            "feature_cols": feature_cols,
            "target_col"  : TARGET_COL,
            "train_df"    : train,
            "test_df"     : test,
        }


if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    preprocessor.run()
