from quantforge.experiments.database import ExperimentDatabase


db = ExperimentDatabase(":memory:")

tables = set(db.tables())

assert "experiments" in tables
assert "metrics" in tables
assert "artifacts" in tables

db.close()

print("PASS")
