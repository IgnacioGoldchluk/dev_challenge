import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
from . import preprocess_utils


class ModelContainer:
    model = None


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

    def __init_model(self, scale):
        self._model = xgb.XGBClassifier(
            random_state=1, learning_rate=0.01, scale_pos_weight=scale
        )

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
        if "Fecha-I" in data.columns:
            # Preprocess for training
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

        # Fill the reqruired columsn that might not exist with empty values
        for col in self._top_10_features:
            features[col] = features.get(col, 0)

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
        target = target.iloc[:, 0]
        x_train, _, y_train, _ = train_test_split(
            features[self._top_10_features], target, test_size=0.33, random_state=42
        )

        n_y0 = len(y_train[y_train == 0])
        n_y1 = len(y_train[y_train == 1])
        scale = n_y0 / n_y1

        self.__init_model(scale)
        self._model.fit(x_train, y_train)

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

        return self._model.predict(features[self._top_10_features]).tolist()

    def preprocess_and_predict(self, features: pd.DataFrame) -> list[int]:
        features = self.preprocess(features)
        return self.predict(features)


def instantiate_model():
    ModelContainer.model = initialize_model()


def predict(features: pd.DataFrame) -> list[int]:
    if ModelContainer.model is None:
        ModelContainer.model = initialize_model()

    return ModelContainer.model.preprocess_and_predict(features)


def initialize_model() -> DelayModel:
    DATA_PATH = "data/data.csv"
    data = pd.read_csv(DATA_PATH)

    model = DelayModel()

    features, target = model.preprocess(data=data, target_column="delay")
    model.fit(features=features, target=target)

    return model
