import logging
import os
from datetime import datetime
from json import JSONDecodeError

import geopandas as gpd
import pandas as pd
import polars as pl
import requests
from dotenv import load_dotenv
from tqdm import tqdm

from .jp_qcew.src.data.data_process import cleanData
from .models import init_dp03_table, init_qcew_table, init_wage_table

load_dotenv()


class DataPull(cleanData):
    def __init__(
        self,
        saving_dir: str = "data/",
        database_file: str = "data.ddb",
        log_file: str = "data_process.log",
    ):
        super().__init__(saving_dir, database_file, log_file)

    def pull_query(self, params: list, year: int) -> pl.DataFrame:
        # prepare custom census query
        param = ",".join(params)
        base = "https://api.census.gov/data/"
        flow = "/acs/acs5/profile"
        url = f"{base}{year}{flow}?get={param}&for=county%20subdivision:*&in=state:72&in=county:*"
        df = pl.DataFrame(requests.get(url).json())

        # get names from DataFrame
        names = df.select(pl.col("column_0")).transpose()
        names = names.to_dicts().pop()
        names = dict((k, v.lower()) for k, v in names.items())

        # Pivot table
        df = df.drop("column_0").transpose()
        return df.rename(names).with_columns(year=pl.lit(year))

    def pull_dp03(self) -> pl.DataFrame:
        if "DP03Table" not in self.conn.sql("SHOW TABLES;").df().get("name").tolist():
            init_dp03_table(self.data_file)
        for _year in range(2012, datetime.now().year):
            if (
                self.conn.sql(f"SELECT * FROM 'DP03Table' WHERE year={_year}")
                .df()
                .empty
            ):
                try:
                    logging.info(f"pulling {_year} data")
                    tmp = self.pull_query(
                        params=[
                            "DP03_0051E",
                            "DP03_0052E",
                            "DP03_0053E",
                            "DP03_0054E",
                            "DP03_0055E",
                            "DP03_0056E",
                            "DP03_0057E",
                            "DP03_0058E",
                            "DP03_0059E",
                            "DP03_0060E",
                            "DP03_0061E",
                        ],
                        year=_year,
                    )
                    tmp = tmp.rename(
                        {
                            "dp03_0051e": "total_house",
                            "dp03_0052e": "inc_less_10k",
                            "dp03_0053e": "inc_10k_15k",
                            "dp03_0054e": "inc_15k_25k",
                            "dp03_0055e": "inc_25k_35k",
                            "dp03_0056e": "inc_35k_50k",
                            "dp03_0057e": "inc_50k_75k",
                            "dp03_0058e": "inc_75k_100k",
                            "dp03_0059e": "inc_100k_150k",
                            "dp03_0060e": "inc_150k_200k",
                            "dp03_0061e": "inc_more_200k",
                        }
                    )
                    tmp = tmp.with_columns(
                        geoid=pl.col("state")
                        + pl.col("county")
                        + pl.col("county subdivision")
                    ).drop(["state", "county", "county subdivision"])
                    tmp = tmp.with_columns(pl.all().exclude("geoid").cast(pl.Int64))
                    self.conn.sql("INSERT INTO 'DP03Table' BY NAME SELECT * FROM tmp")
                    logging.info(f"succesfully inserting {_year}")
                except JSONDecodeError:
                    logging.warning(f"The ACS for {_year} is not availabe")
                    continue
            else:
                logging.info(f"data for {_year} is in the database")
                continue
        return self.conn.sql("SELECT * FROM 'DP03Table';").pl()

    def pull_file(self, url: str, filename: str, verify: bool = True) -> None:
        """
        Pulls a file from a URL and saves it in the filename. Used by the class to pull external files.

        Parameters
        ----------
        url: str
            The URL to pull the file from.
        filename: str
            The filename to save the file to.
        verify: bool
            If True, verifies the SSL certificate. If False, does not verify the SSL certificate.

        Returns
        -------
        None
        """
        chunk_size = 10 * 1024 * 1024

        with requests.get(url, stream=True, verify=verify) as response:
            total_size = int(response.headers.get("content-length", 0))

            with tqdm(
                total=total_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc="Downloading",
            ) as bar:
                with open(filename, "wb") as file:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            file.write(chunk)
                            bar.update(len(chunk))

    def pull_min_wage(self) -> pl.DataFrame:
        url = "https://www.dol.gov/agencies/whd/state/minimum-wage/history"

        if "WageTable" not in self.conn.sql("SHOW TABLES;").df().get("name").tolist():
            init_wage_table(self.data_file)
        if self.conn.sql("SELECT * FROM 'WageTable';").df().empty:
            html = requests.get(url).content

            df_list = pd.read_html(html, match="2023")
            df_2024 = df_list[-1]

            df_list = pd.read_html(html, match="2014")
            df_2019 = df_list[-1]

            df = pd.merge(
                df_2024, df_2019, how="left", on="State or other jurisdiction"
            )
            df = pl.from_pandas(df)
            df = df.rename({"State or other jurisdiction": "state_name"})
            df = df.unpivot(
                index="state_name", variable_name="year", value_name="min_wage"
            )
            self.conn.sql("INSERT INTO 'WageTable' BY NAME SELECT * FROM df;")
        return self.conn.sql("SELECT * FROM 'WageTable';").pl()

    def pull_qcew_file(self, year: int, qtr: int, county: str) -> pl.DataFrame:
        url = f"http://data.bls.gov/cew/data/api/{year}/{qtr}/area/{county}.csv"
        filename = f"{self.saving_dir}raw/bls_{year}_{qtr}_{county}.csv"
        if not os.path.exists(filename):
            self.pull_file(url=url, filename=filename)
            logging.info(f"succesfully downloaded bls_{year}_{qtr}_{county}.csv")
        df = pl.read_csv(filename, ignore_errors=True)
        if len(df.columns) < 5:
            print(county)
            raise ValueError("File Did not download correctly")
        return df

    def pull_qcew(self):
        gdf = gpd.read_file("data/raw/tl_2024_us_county.zip")
        gdf["county_id"] = gdf["STATEFP"] + gdf["COUNTYFP"]
        remove_list_sates = ["66", "69", "60", "09", "15", "69", "02"]
        remove_list_counties = ["46102"]
        gdf = gdf[~gdf["STATEFP"].isin(remove_list_sates)]
        gdf = gdf[~gdf["county_id"].isin(remove_list_counties)]
        county_list = list(gdf["county_id"].values)

        if "QCEWTable" not in self.conn.sql("SHOW TABLES;").df().get("name").tolist():
            init_qcew_table(self.data_file)
        for year in range(2014, 2025):
            for qtr in range(1, 5):
                print(f"{year}-{qtr}")
                county_missing = self.conn.sql(
                    f"SELECT DISTINCT area_fips FROM 'QCEWTable' WHERE year={year} AND qtr={qtr};"
                ).df()
                county_missing["area_fips"] = county_missing["area_fips"].str.zfill(5)
                county_missing = county_missing["area_fips"].to_list()
                county_missing = list(set(county_list) - set(county_missing))
                if len(county_missing) == 0:
                    continue
                print(len(county_missing))
                for county in county_missing:
                    if (
                        self.conn.sql(
                            f"SELECT * FROM 'QCEWTable' WHERE year={year} AND area_fips={county} AND qtr={qtr} LIMIT(1);"
                        )
                        .df()
                        .empty
                    ):
                        df = self.pull_qcew_file(year=year, qtr=qtr, county=county)
                        print(f"{year}-{qtr}-{county}")
                        self.conn.sql(
                            "INSERT INTO 'QCEWTable' BY NAME SELECT * FROM df;"
                        )
                        logging.info(
                            f"succesfully inserted qcew data for {year}-{qtr}-{county}"
                        )
                    else:
                        print(f"qcew data for {year}-{qtr}-{county} is in database")
                        logging.info(
                            f"qcew data for {year}-{qtr}-{county} is in database"
                        )
                self.notify(
                    url=str(os.environ.get("URL")),
                    auth=str(os.environ.get("AUTH")),
                    msg=f"Finished {year}-{qtr}",
                )

    def notify(self, url: str, auth: str, msg: str):
        requests.post(
            url,
            data=msg,
            headers={"Authorization": auth},
        )
