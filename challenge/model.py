import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import xgboost as xgb
from . import preprocess_utils


class DelayModel:
    def __init__(self):
        self._model: xgb.XGBClassifier | None = None
        self._top_10_features: list[str] = [
            "OPERA_Latin American Wings",
            "MES_7",
            "MES_10",
            "OPERA_Grupo LATAM",
            "MES_12",
            "TIPOVUELO_I",
            "MES_4",
            "MES_11",
            "OPERA_Sky Airline",
            "OPERA_Copa Air",
        ]

    def preprocess(
        self, data: pd.DataFrame, target_column: str | None = None
    ) -> tuple[pd.DataFrame, pd.DataFrame] | pd.DataFrame:
        """
        Prepare raw data for training or predict.

        Args:
            data (pd.DataFrame): raw data.
            target_column (str, optional): if set, the target is returned.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: features and target.
            or
            pd.DataFrame: features.
        """
        data["period_day"] = data["Fecha-I"].apply(preprocess_utils.get_period_day)
        data["high_season"] = data["Fecha-I"].apply(preprocess_utils.is_high_season)
        data["min_diff"] = data.apply(preprocess_utils.get_min_diff, axis=1)

        threshold_in_minutes = 15
        data["delay"] = np.where(data["min_diff"] > threshold_in_minutes, 1, 0)

        features = pd.concat(
            [
                pd.get_dummies(data["OPERA"], prefix="OPERA"),
                pd.get_dummies(data["TIPOVUELO"], prefix="TIPOVUELO"),
                pd.get_dummies(data["MES"], prefix="MES"),
            ],
            axis=1,
        )
        if target_column is None:
            return features[self._top_10_features]
        else:
            return features[self._top_10_features], data[[target_column]]

    def fit(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        """
        Fit model with preprocessed data.

        Args:
            features (pd.DataFrame): preprocessed data.
            target (pd.DataFrame): target.
        """
        pass

    def predict(self, features: pd.DataFrame) -> list[int]:
        """
        Predict delays for new flights.

        Args:
            features (pd.DataFrame): preprocessed data.

        Returns:
            (List[int]): predicted targets.
        """
        if self._model is None:
            raise Exception("Cannot predict on uninitialized model")

        self._model.predict(features)
