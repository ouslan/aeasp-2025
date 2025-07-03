from .data_pull import DataPull
import polars as pl


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
