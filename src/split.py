"""Expanding-history snapshots with disjoint seven-day outcome windows."""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import pandas as pd

from src.data import DEFAULT_DB, as_utc, load_events, load_impressions, load_items


DEFAULT_SNAPSHOTS = (
    "2025-02-17T00:00:00Z", "2025-02-24T00:00:00Z", "2025-03-03T00:00:00Z",
    "2025-03-10T00:00:00Z", "2025-03-17T00:00:00Z", "2025-03-24T00:00:00Z",
)
DEFAULT_OBSERVATION_END = "2025-03-31T00:00:00Z"
OUTCOME_LENGTH = pd.Timedelta(7, unit="D")


@dataclass(frozen=True)
class Snapshot:
    role: str
    as_of_time: pd.Timestamp
    outcome_end: pd.Timestamp


def build_snapshots(snapshot_times=DEFAULT_SNAPSHOTS,
                    observation_end=DEFAULT_OBSERVATION_END):
    """Last two snapshots are validation/test; all earlier ones are training."""
    times = [as_utc(value) for value in snapshot_times]
    end = as_utc(observation_end)
    if len(times) < 3:
        raise ValueError("At least one training, one validation, and one test snapshot are required")
    for earlier, later in zip(times, times[1:]):
        if earlier + OUTCOME_LENGTH > later:
            raise ValueError("Snapshots must be chronological with nonoverlapping outcome windows")
    if times[-1] + OUTCOME_LENGTH > end:
        raise ValueError("Observation end does not cover the full final outcome window")
    roles = ["train"] * (len(times) - 2) + ["validation", "test"]
    return [Snapshot(role, time, time + OUTCOME_LENGTH) for role, time in zip(roles, times)]


def split_at(frame, snapshot):
    """The cutoff belongs to outcomes; the outcome end is excluded."""
    times = pd.to_datetime(frame["timestamp"], utc=True)
    history = frame.loc[times < snapshot.as_of_time].copy()
    outcomes = frame.loc[(times >= snapshot.as_of_time) & (times < snapshot.outcome_end)].copy()
    return history, outcomes


def eligible_items(items, as_of_time):
    # Missing creation metadata means the catalog has no known introduction cutoff.
    created = pd.to_datetime(items["created_at"], utc=True)
    return items.loc[created.isna() | (created <= as_utc(as_of_time))].copy()


def summarize_snapshot(snapshot, events, impressions, items):
    history, outcomes = split_at(events, snapshot)
    exposures_before, exposures_after = split_at(impressions, snapshot)
    return {
        "role": snapshot.role,
        "snapshot": snapshot.as_of_time.isoformat(),
        "outcome_end": snapshot.outcome_end.isoformat(),
        "history_events": len(history),
        "outcome_events": len(outcomes),
        "history_event_types": history["event_type"].value_counts().sort_index().to_dict(),
        "outcome_event_types": outcomes["event_type"].value_counts().sort_index().to_dict(),
        "history_visitors": int(history["visitor_id"].nunique()),
        "outcome_visitors": int(outcomes["visitor_id"].nunique()),
        "history_impressions": len(exposures_before),
        "outcome_impressions": len(exposures_after),
        "eligible_items": len(eligible_items(items, snapshot.as_of_time)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--snapshots", nargs="+", default=DEFAULT_SNAPSHOTS)
    parser.add_argument("--observation-end", default=DEFAULT_OBSERVATION_END)
    args = parser.parse_args()
    try:
        snapshots = build_snapshots(args.snapshots, args.observation_end)
        events, impressions, items = load_events(args.db), load_impressions(args.db), load_items(args.db)
        end = as_utc(args.observation_end)
        if any((frame["timestamp"] >= end).any() for frame in (events, impressions)):
            raise ValueError("Observed activity extends beyond the declared observation end")
    except (ValueError, FileNotFoundError) as exc:
        parser.exit(1, f"Snapshot inspection failed: {exc}\n")
    print(json.dumps([summarize_snapshot(s, events, impressions, items) for s in snapshots], indent=2))


if __name__ == "__main__":
    main()
