import pandas as pd
import numpy as np
import datetime

from consts import PATH, TEST_PATH, TRAIN_PATH
from utils import print_log


LAST_DATA_DATE = datetime.date(2023, 2, 23)


class OldTrainTestSplitter:
    def __init__(
        self,
        days_before_die: int = 32,
        train_size: float = 0.7,
    ) -> None:
        self._days_before_die = days_before_die
        self._train_size = train_size

    def _calc_alive(self) -> set[int]:
        print_log("Alive partners calculation")
        data = pd.read_parquet(PATH)
        unreturn_date = LAST_DATA_DATE - datetime.timedelta(self._days_before_die)
        alive_partners = set(data[data["rep_date"] >= unreturn_date]["partner"].values)
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
        grouped = raw_data.groupby("partner", as_index=True)
        rfm = pd.concat(
            [
                grouped.agg(monetary_value=("monetary", np.mean)),
                grouped.agg(first_buy=("rep_date", np.min)),
                grouped.agg(last_buy=("rep_date", np.max)),
                grouped.agg(count=("partner", np.size)),
                grouped.agg(alive=("is_alive", np.max)),
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
        unreturn_date = LAST_DATA_DATE - datetime.timedelta(self._days_before_die)
        data = data[data["rep_date"] < unreturn_date]
        data["is_alive"] = data["partner"].apply(lambda x: x in alive)
        return data

    def get_splited_raw(self) -> dict["Y" : pd.DataFrame, "X" : pd.DataFrame]:
        print_log("Getting splited alive partners data")
        return self._split(self._get_alive_raw())

    def get_splited_rfm(self) -> dict["Y" : pd.DataFrame, "X" : pd.DataFrame]:
        print_log("Getting splited alive partners rfm")
        return self._split(self._create_rfm(self._get_alive_raw()))

    def save_splited_raw(self, train: str, test: str) -> None:
        print_log("Save splitted raw")
        Y, X = self.get_splited_raw().values()
        Y.to_parquet(test)
        X.to_parquet(train)

    def save_splited_rfm(self, train: str, test: str) -> None:
        print_log("Save splitted rfm")
        Y, X = self.get_splited_rfm().values()
        Y.to_parquet(test)
        X.to_parquet(train)


if __name__ == "__main__":
    OldTrainTestSplitter(days_before_die=90).save_splited_rfm()
