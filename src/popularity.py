"""Recommend by weighted pre-snapshot popularity, excluding prior purchases."""

import argparse
import json
from pathlib import Path

import pandas as pd

from src.data import DEFAULT_DB, EVENT_WEIGHTS, read_connection, sql_timestamp


def recommend(visitor_id, k, as_of_time, db_path=DEFAULT_DB):
    """Return item_id, raw weighted score, and source for at most k items."""
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError("k must be a positive integer")
    if not isinstance(visitor_id, str) or not visitor_id.strip():
        raise ValueError("visitor_id must be nonempty")
    cutoff = sql_timestamp(as_of_time)
    with read_connection(db_path) as connection:
        result = pd.read_sql_query(
            """WITH scores AS (
                SELECT item_id,
                    SUM(CASE event_type WHEN 'view' THEN ?
                        WHEN 'add_to_cart' THEN ? WHEN 'purchase' THEN ? END) AS score
                FROM events WHERE timestamp < ? GROUP BY item_id
            )
            SELECT i.item_id, COALESCE(s.score, 0) AS score, 'popularity' AS source
            FROM items AS i LEFT JOIN scores AS s ON i.item_id = s.item_id
            WHERE (i.created_at IS NULL OR i.created_at <= ?)
                AND NOT EXISTS (
                    SELECT 1 FROM events AS e
                    WHERE e.visitor_id = ? AND e.item_id = i.item_id
                        AND e.event_type = 'purchase' AND e.timestamp < ?)
            ORDER BY score DESC, i.item_id ASC LIMIT ?""",
            connection, params=(*EVENT_WEIGHTS.values(), cutoff, cutoff, visitor_id, cutoff, k))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("visitor_ids", nargs="+")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--as-of", required=True)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    args = parser.parse_args()
    try:
        output = [{"visitor_id": visitor,
                   "recommendations": recommend(visitor, args.k, args.as_of, args.db).to_dict("records")}
                  for visitor in args.visitor_ids]
    except (ValueError, FileNotFoundError) as exc:
        parser.exit(1, f"Recommendation failed: {exc}\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
