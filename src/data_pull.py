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
from shapely import wkt

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
        self.conn.install_extension("spatial")
        self.conn.load_extension("spatial")

    def pull_county_shapes(self):
        if "CountyTable" not in self.conn.sql("SHOW TABLES;").df().get("name").tolist():
            # Download the shape files
            if not os.path.exists(f"{self.saving_dir}external/county_shape.zip"):
                self.pull_file(
                    url="https://www2.census.gov/geo/tiger/TIGER2024/STATE/tl_2024_us_state.zip",
                    filename=f"{self.saving_dir}external/county_shape.zip",
                )
                logging.info("Downloaded zipcode shape files")

            gdf = gpd.read_file(f"{self.saving_dir}external/county_shape.zip")
            gdf = gdf.rename(
                columns={"GEOID": "geo_id", "NAME": "county_name", "STATEFP": "fips"}
            )
            gdf = gdf[["geo_id", "fips", "county_name", "geometry"]]
            df = gdf.drop(columns="geometry")

            geometry = gdf["geometry"].apply(lambda geom: geom.wkt)
            df["geometry"] = geometry
            self.conn.execute("CREATE TABLE CountyTable AS SELECT * FROM df")
            logging.info(
                f"The countytable is empty inserting {self.saving_dir}external/cousub.zip"
            )

        gdf = gpd.GeoDataFrame(self.conn.sql("SELECT * FROM CountyTable;").df())
        gdf["geometry"] = gdf["geometry"].apply(wkt.loads)
        gdf = gdf.set_geometry("geometry").set_crs("EPSG:4269", allow_override=True)
        gdf = gdf.to_crs("EPSG:3395")
        gdf["area_fips"] = gdf["geo_id"].astype(str)
        gdf["fips"] = gdf["fips"].astype(str)
        return gdf

    def pull_states_shapes(self):
        if "StateTable" not in self.conn.sql("SHOW TABLES;").df().get("name").tolist():
            # Download the shape files
            if not os.path.exists(f"{self.saving_dir}external/state_shape.zip"):
                self.pull_file(
                    url="https://www2.census.gov/geo/tiger/TIGER2024/STATE/tl_2024_us_state.zip",
                    filename=f"{self.saving_dir}external/state_shape.zip",
                )
                logging.info("Downloaded zipcode shape files")

            gdf = gpd.read_file(f"{self.saving_dir}external/state_shape.zip")
            gdf = gdf.rename(columns={"NAME": "state_name", "STATEFP": "fips"})

            gdf = gdf[["fips", "state_name", "geometry"]]
            df = gdf.drop(columns="geometry")

            geometry = gdf["geometry"].apply(lambda geom: geom.wkt)
            df["geometry"] = geometry
            self.conn.execute("CREATE TABLE StateTable AS SELECT * FROM df")
            logging.info(
                f"The countytable is empty inserting {self.saving_dir}external/cousub.zip"
            )
        return self.conn.sql("SELECT * FROM StateTable;").df()

    def pull_query(self, params: list, year: int) -> pl.DataFrame:
        # prepare custom census query
        param = ",".join(params)
        base = "https://api.census.gov/data/"
        flow = "/acs/acs5/profile"
        url = f"{base}{year}{flow}?get={param}&for=county:*&in=state:*"
        df = pl.DataFrame(requests.get(url).json())

        # get names from DataFrame
        names = df.select(pl.col("column_0")).transpose()
        names = names.to_dicts().pop()
        names = dict((k, v.lower()) for k, v in names.items())

        # Pivot table
        df = df.drop("column_0").transpose()
        return df.rename(names).with_columns(year=pl.lit(year))

    def pull_dp03(self):
        if "DP03Table" not in self.conn.sql("SHOW TABLES;").df().get("name").tolist():
            init_dp03_table(self.data_file)
        for _year in range(2012, datetime.now().year - 1):
            if (
                self.conn.sql(f"SELECT * FROM 'DP03Table' WHERE year={_year}")
                .df()
                .empty
            ):
                try:
                    logging.info(f"pulling {_year} data")
                    tmp = self.pull_query(
                        params=[
                            "DP03_0001E",
                            "DP03_0008E",
                            "DP03_0009E",
                            "DP03_0014E",
                            "DP03_0016E",
                            "DP03_0019E",
                            "DP03_0025E",
                            "DP03_0033E",
                            "DP03_0034E",
                            "DP03_0035E",
                            "DP03_0036E",
                            "DP03_0037E",
                            "DP03_0038E",
                            "DP03_0039E",
                            "DP03_0040E",
                            "DP03_0041E",
                            "DP03_0042E",
                            "DP03_0043E",
                            "DP03_0044E",
                            "DP03_0045E",
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
                            "DP03_0070E",
                            "DP03_0074E",
                        ],
                        year=_year,
                    )
                    tmp = tmp.rename(
                        {
                            "dp03_0001e": "total_population",
                            "dp03_0008e": "in_labor_force",
                            "dp03_0009e": "unemployment",
                            "dp03_0014e": "own_children6",
                            "dp03_0016e": "own_children17",
                            "dp03_0019e": "commute_car",
                            "dp03_0025e": "commute_time",
                            "dp03_0033e": "work_fish",
                            "dp03_0034e": "work_manage",
                            "dp03_0035e": "work_business",
                            "dp03_0036e": "work_finance",
                            "dp03_0037e": "work_computer",
                            "dp03_0038e": "work_architecture",
                            "dp03_0039e": "work_law",
                            "dp03_0040e": "work_education",
                            "dp03_0041e": "work_health",
                            "dp03_0042e": "in_education",
                            "dp03_0043e": "work_art_entertainment",
                            "dp03_0044e": "work_sales",
                            "dp03_0045e": "work_office_administration",
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
                            "dp03_0070e": "with_social_security",
                            "dp03_0074e": "food_stamp",
                        }
                    )
                    tmp = tmp.with_columns(
                        geoid=pl.col("state") + pl.col("county"), fips=pl.col("state")
                    ).drop(["state", "county"])
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
        gdf = self.pull_county_shapes()
        remove_list_sates = ["66", "69", "60", "09", "15", "69", "02"]
        remove_list_counties = ["46102"]
        gdf = gdf[~gdf["fips"].isin(remove_list_sates)]
        gdf = gdf[~gdf["geo_id"].isin(remove_list_counties)]
        county_list = list(gdf["geo_id"].values)

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
