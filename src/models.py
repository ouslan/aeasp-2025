import duckdb


def get_conn(db_path: str) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(db_path)


def init_dp03_table(db_path: str) -> None:
    conn = get_conn(db_path=db_path)

    conn.sql(
        """
        CREATE TABLE IF NOT EXISTS "DP03Table" (
            year INTEGER,
            fips VARCHAR(2),
            geoid VARCHAR(5),
            total_population INTEGER,
            in_labor_force INTEGER,
            unemployment INTEGER,
            own_children6 INTEGER,
            own_children17 INTEGER,
            commute_car INTEGER,
            commute_time FLOAT,
            total_house INTEGER,
            inc_less_10k INTEGER,
            inc_10k_15k INTEGER,
            inc_15k_25k INTEGER,
            inc_25k_35k INTEGER,
            inc_35k_50k INTEGER,
            inc_50k_75k INTEGER,
            inc_75k_100k INTEGER,
            inc_100k_150k INTEGER,
            inc_150k_200k INTEGER,
            inc_more_200k INTEGER,
            with_social_security INTEGER,
            food_stamp INTEGER
            );
        """
    )


def init_qcew_table(db_path: str) -> None:
    conn = get_conn(db_path=db_path)

    conn.sql(
        """
        CREATE TABLE IF NOT EXISTS "QCEWTable" (
            area_fips VARCHAR(5),
            own_code CHAR(1),
            industry_code VARCHAR(6),
            agglvl_code CHAR(2),
            size_code CHAR(1),
            year CHAR(4),
            qtr CHAR(1),
            disclosure_code CHAR(1),
            area_title VARCHAR(80),
            own_title VARCHAR(80),
            industry_title VARCHAR(80),
            agglvl_title VARCHAR(80),
            size_title VARCHAR(80),
            qtrly_estabs INTEGER,
            month1_emplvl INTEGER,
            month2_emplvl INTEGER,
            month3_emplvl INTEGER,
            total_qtrly_wages BIGINT,
            taxable_qtrly_wages BIGINT,
            qtrly_contributions BIGINT,
            avg_wkly_wage INTEGER,
            lq_disclosure_code CHAR(1),
            lq_qtrly_estabs FLOAT,
            lq_month1_emplvl FLOAT,
            lq_month2_emplvl FLOAT,
            lq_month3_emplvl FLOAT,
            lq_total_qtrly_wages FLOAT,
            lq_taxable_qtrly_wages FLOAT,
            lq_qtrly_contributions FLOAT,
            lq_avg_wkly_wage FLOAT,
            oty_disclosure_code CHAR(1),
            oty_qtrly_estabs_chg INTEGER,
            oty_qtrly_estabs_pct_chg FLOAT,
            oty_month1_emplvl_chg INTEGER,
            oty_month1_emplvl_pct_chg FLOAT,
            oty_month2_emplvl_chg INTEGER,
            oty_month2_emplvl_pct_chg FLOAT,
            oty_month3_emplvl_chg INTEGER,
            oty_month3_emplvl_pct_chg FLOAT,
            oty_total_qtrly_wages_chg BIGINT,
            oty_total_qtrly_wages_pct_chg FLOAT,
            oty_taxable_qtrly_wages_chg BIGINT,
            oty_taxable_qtrly_wages_pct_chg FLOAT,
            oty_qtrly_contributions_chg BIGINT,
            oty_qtrly_contributions_pct_chg FLOAT,
            oty_avg_wkly_wage_chg INTEGER,
            oty_avg_wkly_wage_pct_chg FLOAT
            );
        """
    )


def init_wage_table(db_path: str) -> None:
    conn = get_conn(db_path=db_path)

    conn.sql(
        """
        CREATE TABLE IF NOT EXISTS "WageTable" (
            state_name STRING,
            year INTEGER,
            min_wage STRING,
            );
        """
    )
