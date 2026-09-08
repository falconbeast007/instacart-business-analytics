import duckdb
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

DB_PATH = BASE_DIR / "data" / "processed" / "instacart.duckdb"
SQL_DIR = BASE_DIR / "sql"

con = duckdb.connect(str(DB_PATH))

sql_files = sorted(SQL_DIR.glob("*.sql"))

print("=" * 80)
print("RUNNING SQL BUSINESS ANALYSIS")
print("=" * 80)

for sql_file in sql_files:

    print("\n" + "=" * 80)
    print(f"FILE: {sql_file.name}")
    print("=" * 80)

    sql = sql_file.read_text()

    statements = [
        statement.strip()
        for statement in sql.split(";")
        if statement.strip()
    ]

    for i, statement in enumerate(statements, start=1):

        try:
            result = con.execute(statement).df()

            print(f"\nQuery {i}")
            print("-" * 60)

            print(result.head(20).to_string(index=False))

        except Exception as e:

            print(f"\nERROR in Query {i}")
            print(e)

con.close()

print("\n" + "=" * 80)
print("SQL BUSINESS ANALYSIS COMPLETE")
print("=" * 80)