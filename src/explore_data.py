"""Profile the catalog, exposures, and observed actions in a data directory."""

import argparse
import csv
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from statistics import median


SCHEMAS = {
    "items": ["item_id", "category", "brand", "price", "text", "created_at"],
    "impressions": ["impression_id", "timestamp", "visitor_id", "item_id", "session_id", "position"],
    "events": ["event_id", "timestamp", "visitor_id", "item_id", "event_type", "session_id", "impression_id"],
}
ACTION_ORDER = {"view": 0, "add_to_cart": 1, "purchase": 2}


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return reader.fieldnames, list(reader)


def parsed(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def profile(data_dir):
    tables = {}
    problems = []
    for name, expected in SCHEMAS.items():
        path = data_dir / f"{name}.csv"
        fields, rows = read_csv(path)
        tables[name] = rows
        if fields != expected:
            problems.append(f"{name}: unexpected columns {fields}")
        missing = sum(value is None or value.strip() == "" for row in rows for value in row.values())
        print(f"{name}: {len(rows)} rows; missing values: {missing}")
        if missing:
            problems.append(f"{name}: {missing} missing values")
        time_field = "created_at" if name == "items" else "timestamp"
        times = [row[time_field] for row in rows]
        if times:
            print(f"  time range: {min(times)} to {max(times)}")
        ids = [row[expected[0]] for row in rows]
        if len(ids) != len(set(ids)):
            problems.append(f"{name}: duplicate primary IDs")

    items = {row["item_id"]: row for row in tables["items"]}
    impressions = {row["impression_id"]: row for row in tables["impressions"]}
    events = tables["events"]
    print("categories:", dict(sorted(Counter(row["category"] for row in items.values()).items())))
    print("event types:", dict(sorted(Counter(row["event_type"] for row in events).items())))
    print("events by category:", dict(sorted(Counter(items[row["item_id"]]["category"]
                                                     for row in events if row["item_id"] in items).items())))
    visitor_counts = Counter(row["visitor_id"] for row in events)
    print(f"visitors with events: {len(visitor_counts)}; events per visitor: "
          f"min={min(visitor_counts.values(), default=0)}, "
          f"median={median(visitor_counts.values()) if visitor_counts else 0:g}, "
          f"max={max(visitor_counts.values(), default=0)}")
    print("most active visitors:", visitor_counts.most_common(5))
    item_counts = Counter(row["item_id"] for row in events)
    print("most popular items by events:", item_counts.most_common(10))
    print("items with zero events:", len(items.keys() - item_counts.keys()))
    first_time = min((parsed(row["created_at"]) for row in items.values()), default=None)
    if first_time:
        print("items introduced after day 42:", sum(parsed(row["created_at"]) >= first_time + timedelta(days=42)
                                                    for row in items.values()))
    print("least active items:", sorted(((item_id, item_counts[item_id]) for item_id in items),
                                        key=lambda pair: (pair[1], pair[0]))[:10])
    acted = {row["impression_id"] for row in events}
    print(f"impressions with action: {len(acted)}/{len(impressions)} "
          f"({len(acted)/len(impressions):.1%}); no action: {len(impressions)-len(acted)}"
          if impressions else "impressions with action: 0/0")

    for row in tables["impressions"]:
        item = items.get(row["item_id"])
        if item is None:
            problems.append(f"unknown item in {row['impression_id']}")
        elif parsed(row["timestamp"]) < parsed(item["created_at"]):
            problems.append(f"item shown before creation in {row['impression_id']}")
        if not row["visitor_id"] or not row["session_id"]:
            problems.append(f"missing visitor/session in {row['impression_id']}")
        if not row["position"].isdigit() or int(row["position"]) < 1:
            problems.append(f"invalid position in {row['impression_id']}")

    action_sequences = defaultdict(list)
    for row in events:
        impression = impressions.get(row["impression_id"])
        if row["event_type"] not in ACTION_ORDER:
            problems.append(f"invalid event type in {row['event_id']}")
        if impression is None:
            problems.append(f"unknown impression in {row['event_id']}")
            continue
        if any(row[field] != impression[field] for field in ("visitor_id", "item_id", "session_id")):
            problems.append(f"impression mismatch in {row['event_id']}")
        if parsed(row["timestamp"]) <= parsed(impression["timestamp"]):
            problems.append(f"action before impression in {row['event_id']}")
        action_sequences[row["impression_id"]].append(row)
    for impression_id, sequence in action_sequences.items():
        sequence.sort(key=lambda row: row["timestamp"])
        actual = [row["event_type"] for row in sequence]
        if actual not in (["view"], ["view", "add_to_cart"], ["view", "add_to_cart", "purchase"]):
            problems.append(f"invalid action sequence for {impression_id}: {actual}")
        if any(parsed(later["timestamp"]) <= parsed(earlier["timestamp"])
               for earlier, later in zip(sequence, sequence[1:])):
            problems.append(f"non-increasing action times for {impression_id}")

    last_week_start = max((parsed(row["timestamp"]) for row in tables["impressions"]),
                          default=None)
    if last_week_start:
        cutoff = last_week_start - timedelta(days=21)
        later = Counter(row["event_type"] for row in events if parsed(row["timestamp"]) >= cutoff)
        print("events in final 21 days:", dict(sorted(later.items())))
    print(f"reference/chronology checks: {'PASS' if not problems else 'FAIL (' + str(len(problems)) + ' issues)'}")
    for problem in problems[:20]:
        print("  -", problem)
    return not problems


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", nargs="?", type=Path, default=Path("data/synthetic"))
    args = parser.parse_args()
    raise SystemExit(0 if profile(args.data_dir) else 1)
