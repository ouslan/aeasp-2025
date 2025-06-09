from data_pull import DataPull
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
        df_shape = pl.from_pandas(self.pull_states_shapes())
        var = "area_fips,year,qtr,industry_code,agglvl_code,month1_emplvl,month2_emplvl,month3_emplvl,total_qtrly_wages,avg_wkly_wage,qtrly_estabs"
        df = self.conn.sql(
            f"""
            SELECT {var} FROM 'QCEWTable' 
                WHERE agglvl_code=74;
            """
        ).pl()
        df_min = df_min.join(df_shape, on="state_name", how="inner", validate="m:1")

        return df
