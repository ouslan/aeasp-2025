import os

import arviz as az
import bambi as bmb
import numpy as np
import polars as pl

from src.data_pull import DataPull

dp = DataPull()


def main():
    df_min = dp.pull_min_wage()
    df_min = df_min.with_columns(
        min_wage=pl.col("min_wage").str.replace("$", "", literal=True)
    )
    df_shpae = pl.from_pandas(dp.pull_states_shapes().drop("geometry", axis=1))
    df_min = df_min.join(df_shpae, on="state_name", how="inner", validate="m:1")
    df_min = df_min.with_columns(pl.col("year").cast(pl.String))
    var = "area_fips,year,qtr,industry_code,agglvl_code,month1_emplvl,month2_emplvl,month3_emplvl,total_qtrly_wages,avg_wkly_wage,qtrly_estabs"
    df = dp.conn.sql(
        f"""
        SELECT {var} FROM 'QCEWTable' 
            WHERE agglvl_code=74;
        """
    ).pl()
    df = df.with_columns(area_fips=pl.col("area_fips").str.zfill(5))
    df = df.with_columns(fips=pl.col("area_fips").str.slice(0, 2))
    df = df.join(df_min, on=["fips", "year"], how="inner", validate="m:1")
    df = df.filter(pl.col("year").cast(pl.Int32) < 2024)
    fips_list = dp.pull_states_shapes()["fips"].to_list()
    naics_code = [
        "11",
        "21",
        "22",
        "31-33",
        "42",
        "48-49",
        "51",
        "55",
        "61",
        "71",
        "72",
        "81",
        "92",
        "23",
        "44-45",
        "52",
        "54",
        "56",
        "62",
    ]
    not_valid = [
        "17",
        "27",
        "50",
        "31",
        "22",
        "13",
        "01",
        "39",
        "45",
        "40",
        "47",
        "32",
        "26",
        "05",
        "28",
        "30",
        "18",
        "72",
        "51",
    ]
    fips_list = list(set(fips_list) - set(not_valid))

    for fips in fips_list:
        for naics in naics_code:
            data = df.filter(
                (pl.col("fips") == fips) & (pl.col("industry_code") == naics)
            )
            data = data.with_columns(
                k_index=(pl.col("min_wage").cast(pl.Float64) * 8 * 5)
                / pl.col("avg_wkly_wage"),
                employment=(
                    pl.col("month1_emplvl")
                    + pl.col("month2_emplvl")
                    + pl.col("month3_emplvl")
                )
                / 3,
            )
            data = data.with_columns(
                log_k_index=pl.col("k_index").log(),
                log_employment=pl.col("employment").log(),
            )
            data = data.to_pandas()
            data["date"] = data["year"].astype(int) * 10 + data["qtr"].astype(int)
            data["date"] = data["date"].astype("category")
            data["area_fips"] = data["area_fips"].astype("category")
            data = data.sort_values(["year", "qtr", "area_fips"]).reset_index(drop=True)
            data = data.replace([np.inf, -np.inf], np.nan)
            model = bmb.Model(
                "log_employment ~ 0 + area_fips + date + log_k_index",
                data,
                dropna=True,
            )

            results = model.fit(
                inference_method="nutpie",
                sample_kwargs={"target_accept": 0.9},
                tune=2000,
                cores=10,
                chains=10,
                random_seed=787,
            )
            az.to_netcdf(results, f"data/processed/results_{fips}_{naics}.nc")
            dp.notify(
                url=str(os.environ.get("URL")),
                auth=str(os.environ.get("AUTH")),
                msg=f"Successfully completed regression for NAICS {naics} for {fips}",
            )


if __name__ == "__main__":
    main()
