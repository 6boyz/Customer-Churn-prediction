from typing import Callable
import pandas as pd
import numpy as np
import datetime
import math

from consts import PATH, TEST_PATH, TRAIN_PATH
from utils import print_log

FIRST_DATA_DATE = datetime.date(2016, 1, 2)
LAST_DATA_DATE = datetime.date(2023, 2, 23)


class TrainTestSplitter:
    _left_part_days: int
    _right_part_days: int
    _train_size: float
    _save_test_path: str
    _save_train_path: str
    _part_n: int

    _left_date: datetime
    _unreturn_date: datetime
    _right_date: datetime

    def __init__(
        self,
        left_part_days: int = 95,
        right_part_days: int = 95,
        train_size: float = 0.7,
        save_test_path: str = TEST_PATH,
        save_train_path: str = TRAIN_PATH,
        part_n: int = 3,
    ) -> None:
        self._left_part_days = left_part_days
        self._right_part_days = right_part_days
        self._train_size = train_size
        self._save_test_path = save_test_path
        self._save_train_path = save_train_path
        self._part_n = part_n + 1

        self._unreturn_date = LAST_DATA_DATE - datetime.timedelta(
            self._right_part_days * self._part_n
        )
        self._left_date = self._unreturn_date - datetime.timedelta(self._left_part_days)
        self._right_date = self._unreturn_date + datetime.timedelta(
            self._right_part_days
        )

    def _calc_parts_count(days_count: int) -> int:
        return math.ceil((LAST_DATA_DATE - FIRST_DATA_DATE).days / days_count)

    def _calc_alive(self) -> set[int]:
        print_log("Alive partners calculation")
        data = pd.read_parquet(PATH)

        alive_partners = set(
            data[
                (data["rep_date"] >= self._unreturn_date)
                & (data["rep_date"] < self._right_date)
            ]["partner"].values
        )
        return alive_partners

    def _split(
        self, data: pd.DataFrame
    ) -> dict["Y" : pd.DataFrame, "X" : pd.DataFrame]:
        print_log("Data spliting")
        data_len = len(data)
        X_len = int(data_len * self._train_size)
        X_data = data.head(X_len)

        Y_len = data_len - X_len
        Y_data = data.tail(Y_len)
        return {"Y": Y_data, "X": X_data}

    def _create_rfm(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        print_log("Creating RFM")
        grouping = raw_data.groupby("partner", as_index=True)
        rfm = pd.concat(
            [
                grouping.agg(monetary_value=("monetary", np.mean)),
                grouping.agg(first_buy=("rep_date", np.min)),
                grouping.agg(last_buy=("rep_date", np.max)),
                grouping.agg(count=("partner", np.size)),
                grouping.agg(alive=("is_alive", np.max)),
            ],
            axis=1,
        )

        rfm["frequency"] = rfm["count"] - 1
        rfm["recency"] = rfm["last_buy"] - rfm["first_buy"]
        rfm["T"] = LAST_DATA_DATE - rfm["first_buy"]

        rfm["recency"] = rfm["recency"].apply(lambda x: x.days)
        rfm["T"] = rfm["T"].apply(lambda x: x.days)

        return rfm

    def _get_alive_raw(self) -> pd.DataFrame:
        alive = self._calc_alive()
        data = pd.read_parquet(PATH)
        data = data[
            (data["rep_date"] < self._unreturn_date)
            & (data["rep_date"] >= self._left_date)
        ]
        data["is_alive"] = data["partner"].apply(lambda x: x in alive)
        return data

    def _save_data(self, get_data_func: Callable, train_path: str=None, test_path: str=None) -> None:
        if not train_path: train_path = TRAIN_PATH
        if not test_path: test_path = TEST_PATH
        Y, X = get_data_func().values()
        Y.to_parquet(test_path)
        X.to_parquet(train_path)

    def get_splited_raw(self) -> dict["Y" : pd.DataFrame, "X" : pd.DataFrame]:
        print_log("Getting splited alive partners data")
        return self._split(self._get_alive_raw())

    def get_splited_rfm(self) -> dict["Y" : pd.DataFrame, "X" : pd.DataFrame]:
        print_log("Getting splited alive partners rfm")
        return self._split(self._create_rfm(self._get_alive_raw()))

    def save_splited_raw(self, train_path: str=None, test_path: str=None) -> None:
        print_log("Save splitted raw")
        self._save_data(self.get_splited_raw, train_path, test_path)

    def save_splited_rfm(self, train_path: str=None, test_path: str=None) -> None:
        print_log("Save splitted rfm")
        self._save_data(self.get_splited_rfm, train_path, test_path)


if __name__ == "__main__":
    TrainTestSplitter(left_part_days=180).save_splited_rfm()
