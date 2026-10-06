"""Generate a small, reproducible catalog and exposure-linked behavior log."""

import argparse
import csv
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path


START = datetime(2025, 1, 6, tzinfo=timezone.utc)
CATEGORIES = {
    "electronics": ["wireless mouse", "portable charger", "desk lamp", "earbuds", "keyboard"],
    "home": ["linen pillow", "ceramic mug", "storage basket", "cotton throw", "table lamp"],
    "fashion": ["cotton shirt", "denim jacket", "canvas tote", "wool scarf", "running socks"],
    "beauty": ["skin serum", "hand cream", "face cleanser", "lip balm", "body lotion"],
    "sports": ["yoga mat", "water bottle", "resistance bands", "training towel", "gym bag"],
}
BRANDS = ["Boreal", "Elm", "Juniper", "Harbor", "Drift", "Fable", "Cedar", "Solstice"]
PRICE_RANGES = {
    "electronics": (24, 165), "home": (12, 92), "fashion": (15, 115),
    "beauty": (8, 76), "sports": (14, 125),
}


def stamp(value):
    return value.strftime("%Y-%m-%dT%H:%M:%SZ")


def write_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def generate(seed, output_dir):
    rng = random.Random(seed)
    categories = list(CATEGORIES)
    items = []
    for index in range(1, 101):
        category = categories[(index - 1) % len(categories)]
        brand = BRANDS[(index + rng.randrange(len(BRANDS))) % len(BRANDS)]
        product = rng.choice(CATEGORIES[category])
        low, high = PRICE_RANGES[category]
        price = round(rng.uniform(low, high), 2)
        # The last 15 items arrive during the second half of the observation window.
        created = START if index <= 85 else START + timedelta(days=rng.randint(42, 68))
        items.append({
            "item_id": f"item_{index:04d}", "category": category, "brand": brand,
            "price": f"{price:.2f}",
            "text": f"{brand} {product} for everyday {category} use",
            "created_at": stamp(created),
        })

    # Stable item appeal is shared across visitors. A small tail stays hard to discover.
    appeal = {item["item_id"]: (3.5 if n < 12 else 0.18 if n >= 95 else 1.0)
              * rng.uniform(0.7, 1.3) for n, item in enumerate(items)}
    preferences = {}
    sessions = []
    for visitor_num in range(1, 51):
        visitor = f"visitor_{visitor_num:04d}"
        preferences[visitor] = set(rng.sample(categories, rng.choice([1, 1, 2, 2, 2])))
        first_day = rng.randint(0, 12) if visitor_num <= 45 else rng.randint(54, 68)
        # Different visit rates give both active and sparse histories.
        session_count = rng.randint(3, 6) if visitor_num > 45 else rng.randint(5, 15)
        available_days = range(first_day, 84)
        for day in sorted(rng.sample(available_days, min(session_count, len(available_days)))):
            when = START + timedelta(days=day, hours=rng.randint(8, 21), minutes=rng.randint(0, 59))
            sessions.append((when, visitor))
    sessions.sort()

    impressions = []
    events = []
    for session_num, (session_time, visitor) in enumerate(sessions, 1):
        session_id = f"session_{session_num:06d}"
        eligible = [item for item in items if item["created_at"] <= stamp(session_time)]
        shown = set()
        for position in range(1, rng.randint(6, 10) + 1):
            candidates = [item for item in eligible if item["item_id"] not in shown]
            weights = []
            for item in candidates:
                affinity = 3.2 if item["category"] in preferences[visitor] else 0.55
                # Some impressions explore outside the usual preference pattern.
                weights.append(appeal[item["item_id"]] * (1.0 if rng.random() < 0.12 else affinity))
            item = rng.choices(candidates, weights=weights, k=1)[0]
            shown.add(item["item_id"])
            impression_time = session_time + timedelta(minutes=2 * (position - 1))
            impression_id = f"impression_{len(impressions) + 1:07d}"
            impressions.append({
                "impression_id": impression_id, "timestamp": stamp(impression_time),
                "visitor_id": visitor, "item_id": item["item_id"],
                "session_id": session_id, "position": position,
            })
            preferred = item["category"] in preferences[visitor]
            view_probability = 0.36 if preferred else 0.12
            if rng.random() >= view_probability:
                continue
            action_time = impression_time + timedelta(seconds=rng.randint(8, 45))
            action_types = ["view"]
            if rng.random() < (0.25 if preferred else 0.10):
                action_types.append("add_to_cart")
                if rng.random() < (0.32 if preferred else 0.18):
                    action_types.append("purchase")
            for event_type in action_types:
                events.append({
                    "event_id": "", "timestamp": stamp(action_time), "visitor_id": visitor,
                    "item_id": item["item_id"], "event_type": event_type,
                    "session_id": session_id, "impression_id": impression_id,
                })
                action_time += timedelta(seconds=rng.randint(12, 55))

    events.sort(key=lambda row: (row["timestamp"], row["impression_id"]))
    for index, event in enumerate(events, 1):
        event["event_id"] = f"event_{index:07d}"
    write_csv(output_dir / "items.csv",
              ["item_id", "category", "brand", "price", "text", "created_at"], items)
    write_csv(output_dir / "impressions.csv",
              ["impression_id", "timestamp", "visitor_id", "item_id", "session_id", "position"], impressions)
    write_csv(output_dir / "events.csv",
              ["event_id", "timestamp", "visitor_id", "item_id", "event_type", "session_id", "impression_id"], events)
    print(f"Generated {len(items)} items, {len(impressions)} impressions, {len(events)} events in {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("data/synthetic"))
    args = parser.parse_args()
    generate(args.seed, args.output_dir)
