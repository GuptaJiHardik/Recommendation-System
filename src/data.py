"""Validate CSVs and import/read the local recommendation database."""

import argparse
from contextlib import contextmanager
import csv
from datetime import datetime
import json
import math
from pathlib import Path
import sqlite3

import pandas as pd


DEFAULT_DB = Path("data/recommendations.db")
EVENT_WEIGHTS = {"view": 1, "add_to_cart": 3, "purchase": 5}
REQUIRED = {
    "items": ("item_id", "category", "brand", "price", "text"),
    "impressions": ("impression_id", "timestamp", "visitor_id", "item_id"),
    "events": ("event_id", "timestamp", "visitor_id", "item_id", "event_type"),
}
COLUMNS = {
    "items": (*REQUIRED["items"], "created_at"),
    "impressions": (*REQUIRED["impressions"], "session_id", "position"),
    "events": (*REQUIRED["events"], "session_id", "impression_id"),
}
SCHEMA = (
    """CREATE TABLE IF NOT EXISTS items (
        item_id TEXT PRIMARY KEY NOT NULL, category TEXT NOT NULL,
        brand TEXT NOT NULL, price REAL NOT NULL CHECK (price >= 0),
        text TEXT NOT NULL, created_at TEXT)""",
    """CREATE TABLE IF NOT EXISTS impressions (
        impression_id TEXT PRIMARY KEY NOT NULL, timestamp TEXT NOT NULL,
        visitor_id TEXT NOT NULL, item_id TEXT NOT NULL REFERENCES items(item_id),
        session_id TEXT, position INTEGER CHECK (position > 0))""",
    """CREATE TABLE IF NOT EXISTS events (
        event_id TEXT PRIMARY KEY NOT NULL, timestamp TEXT NOT NULL,
        visitor_id TEXT NOT NULL, item_id TEXT NOT NULL REFERENCES items(item_id),
        event_type TEXT NOT NULL CHECK (event_type IN ('view','add_to_cart','purchase')),
        session_id TEXT, impression_id TEXT REFERENCES impressions(impression_id))""",
    "CREATE INDEX IF NOT EXISTS events_time ON events(timestamp)",
    "CREATE INDEX IF NOT EXISTS events_visitor_time ON events(visitor_id, timestamp)",
    "CREATE INDEX IF NOT EXISTS impressions_time ON impressions(timestamp)",
)


class DataValidationError(ValueError):
    """The input cannot be imported without changing its meaning."""


def as_utc(value):
    """Require an explicit timezone; never guess a timezone for raw data."""
    if not isinstance(value, (str, datetime, pd.Timestamp)):
        raise ValueError("timestamp must be a timezone-aware date/time")
    result = pd.Timestamp(value)
    if pd.isna(result) or result.tzinfo is None:
        raise ValueError("timestamp must include a timezone")
    if result.nanosecond:
        raise ValueError("timestamp precision finer than microseconds is unsupported")
    return result.tz_convert("UTC")


def sql_timestamp(value):
    # Fixed precision makes SQLite's lexical comparisons match chronological order.
    return as_utc(value).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def validate_csvs(data_dir=Path("data/synthetic")):
    """Return cleaned DataFrames and counts, or fail with CSV row diagnostics."""
    frames, counts, errors = {}, {}, []
    for table, required in REQUIRED.items():
        try:
            with (Path(data_dir) / f"{table}.csv").open(newline="", encoding="utf-8-sig") as handle:
                reader = csv.reader(handle, strict=True)
                header = next(reader, [])
                if not header or len(header) != len(set(header)):
                    raise DataValidationError(f"{table}: missing or duplicate CSV headers")
                rows = []
                for row in reader:
                    if len(row) != len(header):
                        raise DataValidationError(
                            f"{table} CSV row {reader.line_num}: expected {len(header)} fields, got {len(row)}")
                    rows.append(row)
            frame = pd.DataFrame(rows, columns=header)
        except csv.Error as exc:
            raise DataValidationError(f"{table}: unreadable CSV: {exc}") from exc
        missing = set(required) - set(frame.columns)
        if missing:
            raise DataValidationError(f"{table}: missing columns: {', '.join(sorted(missing))}")
        input_count = len(frame)
        frame = frame.drop_duplicates().copy()
        counts[table] = {"input": input_count, "duplicates": input_count - len(frame),
                         "stored": len(frame)}
        for column in COLUMNS[table]:
            if column not in frame:
                frame[column] = None
        frames[table] = frame
        primary = required[0]
        conflicting_ids = frame[primary].duplicated(keep=False)

        for index, row in frame.iterrows():
            def problem(message):
                errors.append(f"{table} CSV row {index + 2}: {message}")

            identifiers = [c for c in required if c.endswith("_id")]
            for column in identifiers:
                value = row[column]
                if not isinstance(value, str) or not value.strip() or value != value.strip():
                    problem(f"{column} must be a nonempty ID without surrounding whitespace")
            if conflicting_ids.loc[index]:
                problem(f"conflicting {primary}: {row[primary]!r}")
            if table == "items":
                for column in ("category", "brand"):
                    if not isinstance(row[column], str) or not row[column].strip():
                        problem(f"{column} must be nonempty")
                try:
                    price = float(row["price"])
                    if not math.isfinite(price) or price < 0:
                        raise ValueError
                    frame.at[index, "price"] = price
                except (ValueError, TypeError):
                    problem("price must be finite and nonnegative")
            if table == "events" and row["event_type"] not in EVENT_WEIGHTS:
                problem(f"unsupported event_type: {row['event_type']!r}")
            time_column = "created_at" if table == "items" else "timestamp"
            value = row[time_column]
            if time_column == "timestamp" or value is not None:
                try:
                    frame.at[index, time_column] = as_utc(value)
                except (ValueError, TypeError, OverflowError) as exc:
                    problem(f"invalid {time_column}: {exc}")
            if table == "impressions" and row["position"] is not None:
                try:
                    if not str(row["position"]).isdigit() or int(row["position"]) < 1:
                        raise ValueError
                    frame.at[index, "position"] = int(row["position"])
                except (ValueError, TypeError):
                    problem("position must be a positive integer")

    if errors:
        raise DataValidationError("\n".join(errors))

    items = frames["items"].set_index("item_id").to_dict("index")
    impressions = frames["impressions"].set_index("impression_id").to_dict("index")
    for table in ("impressions", "events"):
        for index, row in frames[table].iterrows():
            prefix = f"{table} CSV row {index + 2}: "
            item = items.get(row["item_id"])
            if item is None:
                errors.append(prefix + f"unknown item_id: {row['item_id']!r}")
            elif item["created_at"] is not None and row["timestamp"] < item["created_at"]:
                errors.append(prefix + "interaction before item creation")
            if table == "events" and row["impression_id"] not in (None, ""):
                impression = impressions.get(row["impression_id"])
                if impression is None:
                    errors.append(prefix + f"unknown impression_id: {row['impression_id']!r}")
                else:
                    for column in ("visitor_id", "item_id", "session_id"):
                        if column == "session_id" and row[column] is None:
                            continue
                        if row[column] != impression[column]:
                            errors.append(prefix + f"impression {column} mismatch")
                    if row["timestamp"] <= impression["timestamp"]:
                        errors.append(prefix + "action must follow its impression")
    if errors:
        raise DataValidationError("\n".join(errors))

    for table, frame in frames.items():
        frame = frame.loc[:, list(COLUMNS[table])].copy()
        time_column = "created_at" if table == "items" else "timestamp"
        frame[time_column] = pd.to_datetime(frame[time_column], utc=True)
        if table == "items":
            frame["price"] = frame["price"].astype(float)
        if table == "impressions":
            frame["position"] = frame["position"].astype("Int64")
        frames[table] = frame
    return frames, counts


def import_csvs(data_dir=Path("data/synthetic"), db_path=DEFAULT_DB):
    """Atomically replace database contents after validating all three CSVs."""
    frames, counts = validate_csvs(data_dir)
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path)
    try:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("BEGIN IMMEDIATE")
        for statement in SCHEMA:
            connection.execute(statement)
        for table in ("events", "impressions", "items"):
            connection.execute(f"DELETE FROM {table}")
        for table in ("items", "impressions", "events"):
            frame = frames[table]
            columns = list(COLUMNS[table])
            time_column = "created_at" if table == "items" else "timestamp"
            records = []
            for row in frame.to_dict("records"):
                values = []
                for column in columns:
                    value = row[column]
                    if pd.isna(value) or value == "":
                        value = "" if column == "text" else None
                    elif column == time_column:
                        value = sql_timestamp(value)
                    values.append(value)
                records.append(tuple(values))
            placeholders = ",".join("?" for _ in columns)
            connection.executemany(
                f"INSERT INTO {table} ({','.join(columns)}) VALUES ({placeholders})", records)
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()
    return counts


@contextmanager
def read_connection(db_path=DEFAULT_DB):
    path = Path(db_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Database not found: {path}; run python -m src.data first")
    connection = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
    try:
        yield connection
    finally:
        connection.close()


def _load_table(table, db_path):
    with read_connection(db_path) as connection:
        frame = pd.read_sql_query(f"SELECT * FROM {table} ORDER BY {COLUMNS[table][0]}", connection)
    time_column = "created_at" if table == "items" else "timestamp"
    frame[time_column] = pd.to_datetime(frame[time_column], utc=True)
    if table == "impressions":
        frame["position"] = frame["position"].astype("Int64")
    return frame


def load_items(db_path=DEFAULT_DB):
    return _load_table("items", db_path)


def load_events(db_path=DEFAULT_DB):
    return _load_table("events", db_path)


def load_impressions(db_path=DEFAULT_DB):
    return _load_table("impressions", db_path)


def visitor_history(visitor_id, as_of_time, db_path=DEFAULT_DB):
    with read_connection(db_path) as connection:
        frame = pd.read_sql_query(
            "SELECT * FROM events WHERE visitor_id = ? AND timestamp < ? "
            "ORDER BY timestamp, event_id", connection,
            params=(visitor_id, sql_timestamp(as_of_time)))
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/synthetic"))
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    args = parser.parse_args()
    try:
        counts = import_csvs(args.data_dir, args.db)
    except (DataValidationError, FileNotFoundError) as exc:
        parser.exit(1, f"Import failed: {exc}\n")
    print(json.dumps({"database": str(args.db), "counts": counts}, indent=2))


if __name__ == "__main__":
    main()
