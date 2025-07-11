from .data_pull import DataPull
import numpy as np
import polars as pl
import pandas as pd
from scipy.spatial import distance
import matplotlib.pyplot as plt
from toolz import partial
from scipy.optimize import fmin_slsqp
from joblib import Parallel, delayed
import seaborn as sns


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

    def controls_list(self, target: str) -> list:
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

        data = data.with_columns(
            area_fips="i" + pl.col("area_fips"),
        )

        data_np = data.select(selected_cols).to_numpy()

        # Separate the reference group (area_fips == '06081') and the target group (others)
        reference_df = data.filter(pl.col("area_fips") == target)
        target_df = data.filter(pl.col("area_fips") != target)

        # Convert to numpy arrays
        reference_np = reference_df.select(selected_cols).to_numpy()
        target_np = target_df.select(selected_cols).to_numpy()

        # Compute reference mean and inverse covariance matrix
        mean_vec = np.mean(reference_np, axis=0)
        cov_matrix = np.cov(target_np, rowvar=False)
        inv_cov_matrix = np.linalg.inv(cov_matrix)

        mahalanobis_distances = [
            distance.mahalanobis(row, mean_vec, inv_cov_matrix) for row in data_np
        ]

        data = data.with_columns(
            mahalanobis=pl.Series("mahalanobis", mahalanobis_distances),
        )
        controls = (
            data.filter(pl.col("area_fips") != target)
            .sort("mahalanobis")
            .head(200)
            .select("area_fips")
            .to_series()
            .to_list()
        )
        return controls

    def synth_data(self, controls: list, target: str, date: str) -> pd.DataFrame:
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
        )
        data = (
            df.select(pl.col("area_fips", "date", "total_employment", "avg_wkly_wage"))
            .with_columns(
                controls=pl.when(pl.col("area_fips") == target)
                .then(True)
                .otherwise(False)
            )
            .to_pandas()
        )
        data["date"] = pd.PeriodIndex(df["date"], freq="Q").to_timestamp()
        data["after_treatment"] = data["date"] > pd.to_datetime(date)
        data = data[
            (data["area_fips"].isin(controls)) | (data["area_fips"] == target)
        ].reset_index(drop=True)
        data = data[
            (data["avg_wkly_wage"] != 0) & (data["avg_wkly_wage"].notnull())
        ].reset_index(drop=True)
        return pd.DataFrame(data)

    def synth_freq(self, controls: list, target: str, date: str):
        # Helper functions
        def loss_w(W, X, y) -> float:
            return np.sqrt(np.mean((y - X.dot(W)) ** 2))

        def get_w(X, y):
            w_start = [1 / X.shape[1]] * X.shape[1]

            weights = fmin_slsqp(
                partial(loss_w, X=X, y=y),
                np.array(w_start),
                f_eqcons=lambda x: np.sum(x) - 1,
                bounds=[(0.0, 1.0)] * len(w_start),
                disp=False,
            )
            return weights

        def synthetic_control(area_fips: int, data: pd.DataFrame) -> np.array:
            features = ["total_employment"]

            inverted = (
                data.query("~after_treatment")
                .pivot(index="area_fips", columns="date")[features]
                .T
            )

            y = inverted[area_fips].values  # treated
            X = inverted.drop(columns=area_fips).values  # donor pool

            weights = get_w(X, y)
            synthetic = (
                data.query(f'~(area_fips=="{area_fips}")')
                .pivot(index="date", columns="area_fips")["total_employment"]
                .values.dot(weights)
            )

            data = (
                data.query(f'area_fips=="{area_fips}"')[
                    ["area_fips", "date", "total_employment", "after_treatment"]
                ].assign(synthetic=synthetic)
            ).reset_index(drop=True)
            return data

        def pre_treatment_error(area_fips):
            pre_treat_error = (
                area_fips.query("~after_treatment")["total_employment"]
                - area_fips.query("~after_treatment")["synthetic"]
            ) ** 2
            return pre_treat_error.mean()

        data = self.synth_data(controls=controls, target=target, date=date)

        # Remove invalide area fips
        features = ["total_employment"]
        pre_df = (
            data.query("~after_treatment")
            .pivot(index="area_fips", columns="date", values=features)
            .T
        ).dropna(axis=1)

        post_df = (
            data.query("after_treatment")
            .pivot(index="area_fips", columns="date", values=features)
            .T
        )
        pre_df = pre_df.dropna(axis=1)
        post_df = post_df.dropna(axis=1)

        controls = list(set(pre_df.columns) & set(post_df.columns))

        data = data[data["area_fips"].isin(controls)].reset_index(drop=True)

        features = ["total_employment"]

        inverted = (
            data.query("~after_treatment")
            .pivot(index="area_fips", columns="date", values=features)
            .T
        )

        y = inverted[target].values
        X = inverted.drop(columns=target).values

        synth_weights = get_w(X, y)
        synth_control = (
            data.query("~controls")
            .pivot(index="date", columns="area_fips")["total_employment"]
            .values.dot(synth_weights)
        )
        control_pool = data["area_fips"].unique()

        parallel_fn = delayed(partial(synthetic_control, data=data))

        synthetic_states = Parallel(n_jobs=8)(
            parallel_fn(area_fips) for area_fips in control_pool
        )

        # Graph the Target and Donner pool
        fig, ax = plt.subplots()

        (
            data.groupby(["date", "controls"], as_index=False)
            .agg({"total_employment": "mean"})
            .pipe(
                (sns.lineplot, "data"),
                x="date",
                y="total_employment",
                hue="controls",
                marker="o",
                ax=ax,
            )
        )
        ax.axvline(
            x=pd.to_datetime(date),
            linestyle=":",
            lw=2,
            color="C2",
            label="Implementation of minimum wage",
        )

        ax.legend(loc="upper right")
        ax.set(title="Employment", ylabel="total employment trend Trend")

        # Greph the synthetic control
        plt.figure(figsize=(10, 6))
        plt.plot(
            data.query("controls")["date"],
            data.query("controls")["total_employment"],
            label="San Mateo",
        )
        plt.plot(
            data.query("controls")["date"], synth_control, label="Synthetic Control"
        )
        plt.ylabel("Log employment")
        plt.legend()

        # Plot the impact
        plt.figure(figsize=(10, 6))
        plt.plot(
            data.query("controls")["date"],
            data.query("controls")["total_employment"] - synth_control,
            label="San mateo Effect",
        )
        plt.hlines(
            y=0,
            xmin=pd.to_datetime("2014-01-01"),
            xmax=pd.to_datetime("2020-01-01"),
            lw=2,
        )
        plt.title("Counties - Synthetic Across Time")
        plt.ylabel("Log of Employment")
        plt.legend()

        # Greph lines
        plt.figure(figsize=(12, 7))
        for area_fips in synthetic_states:
            plt.plot(
                area_fips["date"],
                area_fips["total_employment"] - area_fips["synthetic"],
                color="C5",
                alpha=0.4,
            )

        plt.plot(
            data.query("controls")["date"],
            data.query("controls")["total_employment"] - synth_control,
            label="San mateo",
        )

        plt.vlines(
            x=pd.to_datetime(date),
            ymin=-0.25,
            ymax=0.25,
            linestyle=":",
            lw=2,
            label="polosy",
        )
        plt.hlines(
            y=0,
            xmin=pd.to_datetime("2014-01-01"),
            xmax=pd.to_datetime("2020-01-01"),
            lw=3,
        )
        plt.ylabel("Gap in log employment")
        plt.title("Counties - Synthetic Across Time")
        plt.legend()

        # Grph without outliers
        plt.figure(figsize=(12, 7))
        for area_fips in synthetic_states:
            if pre_treatment_error(area_fips) < 80:
                plt.plot(
                    area_fips["date"],
                    area_fips["total_employment"] - area_fips["synthetic"],
                    color="C5",
                    alpha=0.4,
                )

        plt.plot(
            data.query("controls")["date"],
            data.query("controls")["total_employment"] - synth_control,
            label="California",
        )

        plt.vlines(
            x=pd.to_datetime(date),
            ymin=-0.25,
            ymax=0.25,
            linestyle=":",
            lw=2,
            label="polosy",
        )
        plt.hlines(
            y=0,
            xmin=pd.to_datetime("2014-01-01"),
            xmax=pd.to_datetime("2020-01-01"),
            lw=3,
        )
        plt.ylabel("Gap in log Employment")
        plt.title("Distribution of Effects")
        plt.title(
            "counties - Synthetic Across Time (Large Pre-Treatment Errors Removed)"
        )
        plt.legend()
