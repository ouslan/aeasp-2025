from .data_pull import DataPull
import numpy as np
import polars as pl
import pandas as pd
from scipy.spatial import distance


class DataReg(DataPull):
    def __init__(
        self,
        saving_dir: str = "data/",
        database_file: str = "data.ddb",
        log_file: str = "data_process.log",
    ):
        super().__init__(saving_dir, database_file, log_file)

    def data_set(self) -> pl.DataFrame:
        df_min = self.pull_min_wage()
        df_min = df_min.with_columns(
            min_wage=pl.col("min_wage").str.replace("$", "", literal=True),
            year=pl.col("year").cast(pl.String),
        )
        df_shape = pl.from_pandas(self.pull_states_shapes())
        df_min = df_min.join(df_shape, on="state_name", how="inner", validate="m:1")

        df = self.conn.sql(
            """
            SELECT * FROM 'QCEWTable' 
                WHERE agglvl_code=74 AND own_code=5;
            """
        ).pl()
        df = df.with_columns(area_fips=pl.col("area_fips").str.zfill(5))
        df = df.with_columns(
            fips=pl.col("area_fips").str.slice(0, 2),
        )
        df = df.join(df_min, on=["fips", "year"], how="inner", validate="m:1")
        df = df.with_columns(
            qtr=pl.col("qtr").cast(pl.Int32), year=pl.col("year").cast(pl.Int32)
        )
        return df

    def controls_list(self) -> list:
        df = self.data_set()
        df = df.filter((pl.col("industry_code") == "72") & (pl.col("year") == 2015))
        df_dp03 = self.pull_dp03()
        df_dp03 = df_dp03.with_columns(
            area_fips=pl.col("geoid"),
        )
        df = df.group_by(["area_fips", "year"]).agg(
            employment=(
                (
                    pl.col("month1_emplvl")
                    + pl.col("month2_emplvl")
                    + pl.col("month3_emplvl")
                )
                / 3
            ).mean()
        )
        data = df.join(
            df_dp03, on=["area_fips", "year"], how="left", validate="m:1"
        ).sort(by=["area_fips", "year"])
        selected_cols = ["commute_car", "employment", "total_population"]

        data_np = data.select(selected_cols).to_numpy()

        # Compute the mean and covariance matrix
        mean_vec = np.mean(data_np, axis=0)
        cov_matrix = np.cov(data_np, rowvar=False)
        inv_cov_matrix = np.linalg.inv(cov_matrix)

        # Compute Mahalanobis distance of each row from the mean
        mahalanobis_distances = [
            distance.mahalanobis(row, mean_vec, inv_cov_matrix) for row in data_np
        ]

        # Add distances to the DataFrame
        data = data.with_columns(
            mahalanobis=mahalanobis_distances, area_fips="i" + pl.col("area_fips")
        )
        controls = data.sort("mahalanobis").select("area_fips").to_series().to_list()

        return controls

    def synth_data(self, controls: list):
        df = self.data_set()
        df = df.filter((pl.col("industry_code") == "72") & (pl.col("year") < 2020))

        df = df.with_columns(
            date=pl.col("year").cast(pl.String) + "Q" + pl.col("qtr").cast(pl.String),
            dummy=pl.lit(1),
            area_fips="i" + pl.col("area_fips"),
            total_employment=(
                (
                    pl.col("month1_emplvl")
                    + pl.col("month2_emplvl")
                    + pl.col("month3_emplvl")
                )
                / 3
            ).log(),
            # after_treatment=pl.when((pl.col("year") >= 2016) & (pl.col("qtr") > 1)).then(True).otherwise(False)
        )
        data = (
            df.select(pl.col("area_fips", "date", "total_employment", "avg_wkly_wage"))
            .with_columns(
                controls=pl.when(pl.col("area_fips") == "i06081")
                .then(True)
                .otherwise(False)
            )
            .to_pandas()
        )
        data["date"] = pd.PeriodIndex(df["date"], freq="Q").to_timestamp()
        data["after_treatment"] = data["date"] > pd.to_datetime("2016-01-01")
        data = data[
            (data["area_fips"].isin(controls)) | (data["area_fips"] == "i06081")
        ].reset_index(drop=True)
        return data
