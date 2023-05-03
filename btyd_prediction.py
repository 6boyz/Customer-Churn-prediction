import numpy as np
import pandas as pd
from btyd import BetaGeoFitter
import cloudpickle

rfm_data = pd.read_parquet("./data/rfm.parquet.gzip")
bgf = cloudpickle.load(open("./model/rfm.model.pkl", "rb"))
future_purchases_matrix = bgf.conditional_expected_number_of_purchases_up_to_time(
    95, rfm_data["frequency"], rfm_data["recency"], rfm_data["T"]
)
print(future_purchases_matrix)
