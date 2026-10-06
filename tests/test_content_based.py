"""Interpretable metadata retrieval, weighted profiles, and chronological comparisons."""

import json
import math
import sys

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import issparse

from src import content_based
from src.collaborative import CollaborativeModel, fit_collaborative
from src.content_based import fit_content_based
from src.data import COLUMNS, import_csvs, load_events, load_items
from src.evaluate import evaluate_snapshot
from src.popularity import recommend
from src.split import Snapshot


CUTOFF = pd.Timestamp("2025-03-17T00:00:00Z")
BEFORE = CUTOFF - pd.Timedelta(1, unit="D")


def events(rows=()):
    return pd.DataFrame(rows, columns=["visitor_id", "item_id", "event_type", "timestamp"])


def catalog(rows):
    return pd.DataFrame(rows, columns=["item_id", "category", "brand", "price", "text"]).assign(
        created_at=BEFORE).astype({"price": float})


@pytest.fixture
def tiny():
    items = catalog([("c", "home", "Elm", 10, ""),
                     ("b", "beauty", "Fable", 10, ""),
                     ("a", "home", "Elm", 10, "")])
    history = events([("u1", "a", "view", BEFORE), ("u1", "a", "add_to_cart", BEFORE),
                      ("u1", "b", "view", BEFORE)])
    return history, items, fit_content_based(history, items, CUTOFF)


def test_hand_calculated_profile_scores_and_trace(tiny):
    history, _, model = tiny
    assert model.item_ids == ("a", "b", "c")
    assert issparse(model.matrix) and model.matrix.dtype == np.float64
    norms = np.sqrt(model.matrix.multiply(model.matrix).sum(axis=1)).A.ravel()
    np.testing.assert_allclose(norms, 1)
    a, b = model.matrix[0].toarray(), model.matrix[1].toarray()
    strengths, _ = model._history("u1", history)
    profile, norm = model._profile(strengths)
    np.testing.assert_allclose(profile.toarray(), (4 * a + b) / 5)
    assert norm == pytest.approx(math.sqrt(59 / 3) / 5)
    assert float((model.matrix[0] @ model.matrix[1].T).toarray()[0, 0]) == pytest.approx(1 / 3)
    result = model.generate("u1", history, 50)
    assert result.item_id.tolist() == ["a", "c", "b"]
    assert set(result.source) == {"content"}
    assert result.set_index("item_id").loc["c", "score"] == pytest.approx(13 / math.sqrt(177))
    trace = model.explain("u1", history, "c")
    assert trace.source_item_id.tolist() == ["a", "b"]
    assert trace.interaction_strength.tolist() == [4, 1]
    assert trace.similarity.tolist() == pytest.approx([1, 1 / 3])
    assert trace.contribution.sum() == pytest.approx(result.set_index("item_id").loc["c", "score"])


@pytest.mark.parametrize("column,replacement", [("category", "beauty"), ("brand", "Fable"),
                                                  ("price", 100), ("text", "skin lotion")])
def test_each_feature_block_affects_similarity(column, replacement):
    items = catalog([(item, "home", "Elm", 10, "ceramic mug") for item in ("a", "b", "clone")])
    items.loc[items.item_id == "b", column] = replacement
    history = events([("u", "a", "view", BEFORE)])
    model = fit_content_based(history, items, CUTOFF)
    result = model.generate("u", history, 50).set_index("item_id")
    assert result.loc["clone", "score"] == pytest.approx(1)
    assert result.loc["b", "score"] == pytest.approx(0.75)
    start, end = model.feature_blocks["text" if column == "text" else column]
    assert (model.matrix[0, start:end] @ model.matrix[1, start:end].T).toarray()[0, 0] == 0


def test_text_is_lowercase_unigram_tfidf_and_blank_rows_are_safe():
    items = catalog([("a", "home", "Elm", 10, "MUG mug ceramic"),
                     ("b", "home", "Elm", 10, "mug"), ("blank", "home", "Elm", 10, None)])
    model = fit_content_based(events(), items, CUTOFF)
    assert model.vectorizer.get_feature_names_out().tolist() == ["ceramic", "mug"]
    text = model.vectorizer.transform(items.text.fillna(""))
    assert text[0, 1] / text[0, 0] == pytest.approx(2 * (math.log(4 / 3) + 1) / (math.log(4 / 2) + 1))
    start, end = model.feature_blocks["text"]
    assert model.matrix[2, start:end].nnz == 0
    assert np.isfinite(model.matrix.data).all()


def test_price_quantiles_boundary_ties_duplicates_and_constant_prices(tiny):
    items = catalog([(str(i), "home", "Elm", price, "") for i, price in enumerate(range(0, 60, 10))])
    model = fit_content_based(events(), items, CUTOFF)
    np.testing.assert_allclose(model.price_edges, [0, 10, 20, 30, 40, 50])
    bucket = np.searchsorted(model.price_edges[1:-1], items.price, side="right").reshape(-1, 1)
    assert bucket.ravel().tolist() == [0, 1, 2, 3, 4, 4]
    start, end = model.feature_blocks["price"]
    assert model.matrix[:, start:end].argmax(axis=1).A.ravel().tolist() == [0, 1, 2, 3, 4, 4]
    items["price"] = [0, 0, 0, 10, 10, 10]
    duplicate = fit_content_based(events(), items, CUTOFF)
    assert len(duplicate.price_edges) == len(np.unique(duplicate.price_edges))
    _, _, constant = tiny
    assert constant.price_edges.tolist() == [10]
    assert constant.feature_blocks["price"][1] - constant.feature_blocks["price"][0] == 1


def test_metadata_only_item_is_retrieved_but_not_collaboratively(tiny):
    history, items, model = tiny
    assert "c" not in set(history.item_id)
    assert "c" in set(model.generate("u1", history, 50).item_id)
    collaborative = fit_collaborative(history, items, CUTOFF)
    assert "c" not in set(collaborative.generate("u1", history, 50).item_id)


def test_purchase_evidence_and_view_cart_eligibility(tiny):
    history, items, _ = tiny
    history = pd.concat([history, events([("u1", "a", "purchase", BEFORE)])], ignore_index=True)
    model = fit_content_based(history, items, CUTOFF)
    result = model.generate("u1", history, 50)
    assert result.item_id.tolist() == ["c", "b"]
    assert model.explain("u1", history, "c").interaction_strength.tolist() == [9, 1]
    assert model.explain("u1", history, "a").empty


def test_cart_and_purchase_weights_shift_the_profile(tiny):
    _, items, _ = tiny
    model = fit_content_based(events(), items, CUTOFF)
    scores = []
    for action in ("view", "add_to_cart", "purchase"):
        history = events([("u", "a", action, BEFORE), ("u", "b", "view", BEFORE)])
        scores.append(model.generate("u", history, 50).set_index("item_id").loc["c", "score"])
    assert scores[0] < scores[1] < scores[2]


def test_future_data_cannot_change_features_or_past_results(tiny):
    history, items, model = tiny
    future_items = catalog([("future", "new category", "new brand", 10000, "future vocabulary")])
    future_items["created_at"] = CUTOFF + pd.Timedelta(1, unit="D")
    future_events = events([("u1", "b", "purchase", CUTOFF),
                            ("later", "future", "view", CUTOFF + pd.Timedelta(2, unit="D"))])
    later = fit_content_based(pd.concat([history, future_events]), pd.concat([items, future_items]), CUTOFF)
    assert later.item_ids == model.item_ids
    assert later.feature_names == model.feature_names
    assert later.popularity_scores == model.popularity_scores
    np.testing.assert_array_equal(later.price_edges, model.price_edges)
    np.testing.assert_array_equal(later.matrix.toarray(), model.matrix.toarray())
    pd.testing.assert_frame_equal(later.generate("u1", history, 50), model.generate("u1", history, 50))
    with_text = items.assign(text=["mug", "lotion", "mug"])
    before = fit_content_based(history, with_text, CUTOFF)
    after = fit_content_based(history, pd.concat([with_text, future_items]), CUTOFF)
    assert before.vectorizer.vocabulary_ == after.vectorizer.vocabulary_
    np.testing.assert_array_equal(before.vectorizer.idf_, after.vectorizer.idf_)
    future_items["created_at"] = CUTOFF
    assert "future" in fit_content_based(history, pd.concat([items, future_items]), CUTOFF).item_ids
    items["created_at"] = pd.NaT
    assert fit_content_based(history, items, CUTOFF).item_ids == model.item_ids


def test_ties_limits_shuffled_inputs_and_short_lists(tiny):
    history, items, model = tiny
    shuffled = fit_content_based(history.iloc[::-1], items.iloc[::-1], CUTOFF)
    pd.testing.assert_frame_equal(shuffled.generate("u1", history.iloc[::-1], 50), model.generate("u1", history, 50))
    assert model.generate("u1", history, 1).item_id.tolist() == ["a"]
    # No common features: zero-score items must not backfill personalized retrieval.
    items = catalog([("a", "home", "Elm", 0, "mug"), ("b", "beauty", "Fable", 100, "lotion")])
    user = events([("u", "a", "view", BEFORE)])
    assert fit_content_based(user, items, CUTOFF).generate("u", user, 50).item_id.tolist() == ["a"]


def test_empty_histories_catalog_and_exhausted_results(tiny):
    history, items, model = tiny
    empty = history.iloc[:0]
    fallback = model.generate("unknown", empty, 50)
    assert fallback.to_dict("records") == [
        {"item_id": "a", "score": 4.0, "source": "popularity"},
        {"item_id": "b", "score": 1.0, "source": "popularity"},
        {"item_id": "c", "score": 0.0, "source": "popularity"}]
    assert model.explain("unknown", empty, "c").empty
    untrained = fit_content_based(empty, items, CUTOFF)
    assert untrained.generate("unknown", empty, 50).item_id.tolist() == ["a", "b", "c"]
    no_catalog = fit_content_based(empty, items.iloc[:0], CUTOFF)
    assert no_catalog.matrix.shape == (0, 0)
    result = no_catalog.generate("unknown", empty, 50)
    assert result.empty and result.columns.tolist() == ["item_id", "score", "source"]
    purchases = events([("buyer", item, "purchase", BEFORE) for item in model.item_ids])
    assert model.generate("buyer", purchases, 50).empty


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_invalid_limit(tiny, limit):
    history, _, model = tiny
    with pytest.raises(ValueError, match="positive integer"):
        model.generate("u1", history, limit)


def test_invalid_visitor_history_and_candidate(tiny):
    history, _, model = tiny
    for visitor in ("", " ", None):
        with pytest.raises(ValueError, match="nonempty"):
            model.generate(visitor, history.iloc[:0], 10)
    with pytest.raises(ValueError, match="another visitor"):
        model.generate("u2", history, 10)
    for timestamp in (CUTOFF, CUTOFF + pd.Timedelta(1, unit="D"), pd.NaT):
        with pytest.raises(ValueError, match="strictly before"):
            model.generate("u1", events([("u1", "a", "view", timestamp)]), 10)
    with pytest.raises(ValueError, match="Unknown event"):
        model.generate("u1", events([("u1", "a", "click", BEFORE)]), 10)
    with pytest.raises(ValueError, match="unavailable"):
        model.generate("u1", events([("u1", "missing", "view", BEFORE)]), 10)
    with pytest.raises(ValueError, match="canonical"):
        model.generate("u1", pd.DataFrame(), 10)
    with pytest.raises(ValueError, match="unavailable"):
        model.explain("u1", history, "missing")


@pytest.mark.parametrize("bad", ["duplicate", "missing_column", "blank_brand", "bad_price", "unknown_history"])
def test_invalid_fit_inputs(tiny, bad):
    history, items, _ = tiny
    if bad == "duplicate":
        items = pd.concat([items, items.iloc[:1]])
    elif bad == "missing_column":
        items = items.drop(columns="text")
    elif bad == "blank_brand":
        items.loc[0, "brand"] = " "
    elif bad == "bad_price":
        items.loc[0, "price"] = np.inf
    else:
        history = events([("u1", "missing", "view", BEFORE)])
    with pytest.raises(ValueError):
        fit_content_based(history, items, CUTOFF)


def test_evaluator_and_diagnostics(tiny):
    history, items, model = tiny
    all_events = pd.concat([history, events([("u1", "c", "add_to_cart", CUTOFF),
                                             ("cold", "a", "purchase", CUTOFF)])])
    result = evaluate_snapshot(Snapshot("validation", CUTOFF, CUTOFF + pd.Timedelta(7, unit="D")),
                               all_events, items, model.generate)
    assert result["segments"]["personalized"]["candidate_recall_at_50"] == 1
    assert result["segments"]["cold_start"]["candidate_recall_at_50"] == 1
    diagnostics = content_based._diagnostics(result, model, all_events, items)
    assert diagnostics["zero_history_catalog_items"] == ["c"]
    assert diagnostics["zero_history_items_retrieved"] == ["c"]
    assert diagnostics["visitor_results"][0]["top10_category_counts"] == {"beauty": 1, "home": 2}
    assert diagnostics["median_largest_category_share"] == pytest.approx(2 / 3)
    assert diagnostics["visitor_results"][0]["new_item_example"]["item_id"] == "c"


@pytest.fixture
def database(tiny, tmp_path):
    history, items, _ = tiny
    directory = tmp_path / "csv"
    directory.mkdir()
    all_events = pd.concat([history, events([("u1", "c", "purchase", CUTOFF),
                                             ("u1", "b", "purchase", CUTOFF + pd.Timedelta(7, unit="D"))])],
                           ignore_index=True)
    all_events = all_events.assign(event_id=[f"e{i}" for i in range(len(all_events))],
                                   session_id="", impression_id="")
    for name, frame in (("items", items), ("events", all_events),
                         ("impressions", pd.DataFrame(columns=COLUMNS["impressions"]))):
        frame.to_csv(directory / f"{name}.csv", index=False)
    db = tmp_path / "data.db"
    import_csvs(directory, db)
    return db


def test_sql_popularity_fallback_parity(database):
    events_frame, items = load_events(database), load_items(database)
    model = fit_content_based(events_frame, items, CUTOFF)
    assert model.generate("unknown", events_frame.iloc[:0], 50).to_dict("records") == recommend(
        "unknown", 50, CUTOFF, database).to_dict("records")


def test_cli_recommend_and_validation_only_without_artifact_writes(database, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["content_based", "recommend", "u1", "unknown",
                                      "--as-of", CUTOFF.isoformat(), "--db", str(database)])
    content_based.main()
    output = json.loads(capsys.readouterr().out)
    assert output[0]["first_candidate_trace"]
    assert "category" in output[0]["recommendations"][0]
    assert output[1]["history_events"] == 0
    assert output[1]["recommendations"][0]["source"] == "popularity"
    snapshots = []
    original = content_based.evaluate_snapshot

    def capture(snapshot, *args):
        snapshots.append(snapshot.role)
        return original(snapshot, *args)

    def cannot_save(*args):
        pytest.fail("Phase 4 must not save collaborative artifacts")

    monkeypatch.setattr(content_based, "evaluate_snapshot", capture)
    monkeypatch.setattr(CollaborativeModel, "save", cannot_save)
    before = database.read_bytes()
    monkeypatch.setattr(sys, "argv", ["content_based", "evaluate", "--db", str(database)])
    content_based.main()
    output = json.loads(capsys.readouterr().out)
    assert snapshots == ["validation"] * 3
    for name in ("content", "collaborative", "popularity"):
        assert output[name]["snapshot"] == CUTOFF.isoformat()
        assert output[name]["counts"]["outcome_events"] == 1
    assert output["content_fit_ms"] >= 0
    assert database.read_bytes() == before


@pytest.mark.parametrize("failure", ["missing_db", "invalid_limit", "observation_end"])
def test_cli_errors(database, tmp_path, monkeypatch, capsys, failure):
    args = ["content_based", "evaluate", "--db", str(database)]
    if failure == "missing_db":
        args[-1] = str(tmp_path / "missing.db")
    elif failure == "invalid_limit":
        args = ["content_based", "recommend", "u1", "--as-of", CUTOFF.isoformat(),
                "--db", str(database), "--k", "0"]
    else:
        monkeypatch.setattr(content_based, "DEFAULT_OBSERVATION_END", CUTOFF.isoformat())
    monkeypatch.setattr(sys, "argv", args)
    before = database.read_bytes()
    with pytest.raises(SystemExit) as exc:
        content_based.main()
    assert exc.value.code == 1
    assert "Content command failed:" in capsys.readouterr().err
    assert database.read_bytes() == before
