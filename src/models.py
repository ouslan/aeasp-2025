import duckdb


def get_conn(db_path: str) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(db_path)


def init_dp03_table(db_path: str) -> None:
    conn = get_conn(db_path=db_path)

    # Create sequence for primary keys
    conn.sql("DROP SEQUENCE IF EXISTS dp03_sequence;")
    conn.sql("CREATE SEQUENCE dp03_sequence START 1;")
    conn.sql(
        """
        CREATE TABLE IF NOT EXISTS "DP03Table" (
            id INTEGER PRIMARY KEY DEFAULT nextval('dp03_sequence'),
            year INTEGER,
            geoid VARCHAR(30),
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
            inc_more_200k INTEGER
            );
        """
    )


def init_geo_table(db_path: str) -> None:
    conn = get_conn(db_path=db_path)
    conn.install_extension("spatial")
    conn.load_extension("spatial")
    conn.sql("DROP SEQUENCE IF EXISTS geo_sequence;")
    conn.sql("CREATE SEQUENCE geo_sequence START 1;")
    conn.sql(
        """
        CREATE TABLE IF NOT EXISTS "GeoTable" (
            id INTEGER PRIMARY KEY DEFAULT nextval('geo_sequence'),
            geoid TEXT,
            name TEXT,
            geometry GEOMETRY,
            );
        """
    )
