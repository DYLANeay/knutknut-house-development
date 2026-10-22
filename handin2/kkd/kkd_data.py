"""KKD data: read the raw jsonl files, clean them, encode them for the price model.

The raw files in data/ are never modified, everything happens in memory.

    load_clean_rows()   -> cleaned houses (dicts, every column) for plots / insights
    load_dataset()      -> X, y, stats   numpy arrays for the price model
    encode_house(h, s)  -> one row of X  for the dashboard (same encoding as training)
"""
import json
import os

import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
REF_YEAR = 2022              # newest year in the data, used to compute ages
BAD_PRICE = 2147483646       # int32 max: placeholder, not a real price

# fixed category order -> the one-hot columns are always in the same order.
# the first value is the reference: it gets no column (its effect is inside the bias),
# otherwise one-hot + bias is redundant and the weights can't be read in part 5
ADVERTISEMENTS = ["no", "regular", "premium"]
COLORS = ["white", "blue", "red", "green", "black", "gray", "unknown"]


def _read_jsonl(name):
    rows = []
    with open(os.path.join(DATA_DIR, name)) as f:
        for line in f:
            if line.strip():
                # some lines have junk before the '{' (houses line 734 starts with "1{")
                rows.append(json.loads(line[line.index("{"):]))
    return rows


# lookup tables: id -> info, used for the joins
SCHOOLS = {s["id"]: s for s in _read_jsonl("schools.jsonl")}
DISTRICTS = {d["id"]: d for d in _read_jsonl("districts.jsonl")}
AGENTS = {a["agent_id"]: a["name"] for a in _read_jsonl("agents.jsonl")}
DISTRICT_IDS = list(DISTRICTS)   # file order; the first one is the reference district


def clean_house(h, median_year):
    """Clean one raw house (houses.jsonl format). Returns a new dict, or None if the row must be dropped."""
    if h.get("price", 0) >= BAD_PRICE:
        return None
    h = dict(h)
    h.pop("rooms", None)                                   # 71 % empty, size says the same thing
    h["year"] = h.get("year") or median_year               # 34 houses have no year
    h["color"] = {"reds": "red"}.get(h["color"], h["color"])
    h["kitchens"] = int(h["kitchens"])                     # mix of 1 / 1.0 / 2.0

    # remodeled = -1 means "never": split into a yes/no flag + how long ago
    h["was_remodeled"] = int(h["remodeled"] != -1)
    h["years_since_remodel"] = REF_YEAR - h["remodeled"] if h["was_remodeled"] else 0
    h["age"] = REF_YEAR - h["year"]

    # joins: school and district info, agent name
    school = SCHOOLS[h["school_id"]]
    district = DISTRICTS[h["district_id"]]
    h["school_rating"] = school["rating"]
    h["school_capacity"] = school["capacity"]
    h["crime_rating"] = district["crime_rating"]
    h["public_transport_rating"] = district["public_transport_rating"]
    h["agent_name"] = AGENTS[h["agent_id"]]
    return h


def load_clean_rows():
    """All cleaned houses with every column (sold, days_on_marked, agent... included)."""
    raw = _read_jsonl("houses.jsonl")
    median_year = int(np.median([h["year"] for h in raw if h.get("year")]))
    rows = [clean_house(h, median_year) for h in raw]
    return [h for h in rows if h is not None], median_year


# the price model inputs. Not used: sold / sold_in_month / days_on_marked (they happen
# after the price is set), agent (not a property of the house), crime / transport
# (2 districts have identical ratings, the district one-hot already covers them)
NUMERIC = ["size", "lot_w", "sun_factor", "bathrooms", "kitchens", "condition_rating",
           "storage_rating", "external_storage_m2", "age", "was_remodeled",
           "years_since_remodel", "school_rating", "school_capacity"]
FEATURE_NAMES = (NUMERIC + ["fireplace", "parking"]
                 + ["ad_" + a for a in ADVERTISEMENTS[1:]]
                 + ["color_" + c for c in COLORS[1:]]
                 + ["district_" + str(i) for i in range(1, len(DISTRICT_IDS))])


def _one_hot(value, categories):
    return [float(value == c) for c in categories[1:]]   # reference category = all zeros


def _features(h):
    """One cleaned house -> list of numbers, in the FEATURE_NAMES order (not scaled yet)."""
    return ([float(h[k]) for k in NUMERIC]
            + [float(h["fireplace"] == "yes"), float(h["parking"] == "yes")]
            + _one_hot(h["advertisement"], ADVERTISEMENTS)
            + _one_hot(h["color"], COLORS)
            + _one_hot(h["district_id"], DISTRICT_IDS))


def load_dataset():
    """X (n_houses, n_features) standardized, y = price in NOK (n_houses,), stats needed to encode new houses.

    stats is a dict of numpy arrays, so the model can save it as is: np.savez("model.npz", w=..., b=..., **stats)
    """
    rows, median_year = load_clean_rows()
    X_raw = np.array([_features(h) for h in rows])
    y = np.array([h["price"] for h in rows], dtype=float)

    # standardize: every column gets mean 0 / std 1, so SGD treats them equally
    # and the weights can be compared with each other (part 5, key drivers)
    mean = X_raw.mean(axis=0)
    std = X_raw.std(axis=0)
    std[std == 0] = 1.0
    stats = {"mean": mean, "std": std, "median_year": np.array(median_year),
             "feature_names": np.array(FEATURE_NAMES)}
    return (X_raw - mean) / std, y, stats


def encode_house(house, stats):
    """For the dashboard: a raw house dict (same keys as houses.jsonl) -> one standardized row of X."""
    h = clean_house(house, int(stats["median_year"]))
    return (np.array(_features(h)) - stats["mean"]) / stats["std"]


def train_test_split(X, y, test_frac=0.2, seed=0):
    idx = np.random.default_rng(seed).permutation(len(y))
    n_test = int(len(y) * test_frac)
    test, train = idx[:n_test], idx[n_test:]
    return X[train], X[test], y[train], y[test]


if __name__ == "__main__":
    # quick self-check: python kkd_data.py
    rows, median_year = load_clean_rows()
    X, y, stats = load_dataset()
    assert len(rows) == 2549, len(rows)              # 2551 lines - 2 fake prices
    assert X.shape == (2549, len(FEATURE_NAMES))
    assert not np.isnan(X).any() and y.max() < BAD_PRICE
    assert all("rooms" not in h and h["color"] != "reds" for h in rows)
    raw = _read_jsonl("houses.jsonl")[0]
    assert np.allclose(encode_house(raw, stats), X[0])   # dashboard encoding == training encoding

    print(f"{len(rows)} houses, {X.shape[1]} features, median year used for missing = {median_year}")
    print(f"price: min {y.min():,.0f}  median {np.median(y):,.0f}  max {y.max():,.0f} NOK")
    print("features:", ", ".join(FEATURE_NAMES))
