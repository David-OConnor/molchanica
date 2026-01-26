"""
Requires the (linux-only?) pytdc lib.

prints the indices to stdout, e.g. for ingesting into Rust via c+p.
"""

from tdc.single_pred import ADME, Tox
import pandas as pd


def _dataset_df(data_obj) -> pd.DataFrame:
    # Newer/older pytdc versions differ; try common APIs.
    if hasattr(data_obj, "get_data"):
        return data_obj.get_data()
    if hasattr(data_obj, "get_dataset"):
        return data_obj.get_dataset()
    raise RuntimeError("Couldn't find a way to fetch the full dataset dataframe from this TDC object.")


def train_test_split(name: str, tox: bool, method: str = "scaffold"):
    if tox:
        data = Tox(name=name)
    else:
        data = ADME(name=name)

    split = data.get_split(method=method)
    train = split["train"].copy()
    test = split["test"].copy()
    valid = split["valid"].copy()

    full = _dataset_df(data).copy()

    # Define original indices as positions in the full dataset.
    full = full.reset_index(drop=True)
    full["orig_idx"] = full.index

    # Join keys = all columns shared between full and split frames (excluding orig_idx).
    common_cols = [c for c in train.columns if c in full.columns and c != "orig_idx"]
    if not common_cols:
        raise RuntimeError("No shared columns between full dataset and train/test split to map indices.")

    # Map each split row back to its original index.
    train_m = train.merge(full[common_cols + ["orig_idx"]], on=common_cols, how="left", validate="m:1")
    test_m = test.merge(full[common_cols + ["orig_idx"]], on=common_cols, how="left", validate="m:1")
    valid_m = valid.merge(full[common_cols + ["orig_idx"]], on=common_cols, how="left", validate="m:1")

    if train_m["orig_idx"].isna().any() or test_m["orig_idx"].isna().any():
        missing_train = int(train_m["orig_idx"].isna().sum())
        missing_test = int(test_m["orig_idx"].isna().sum())
        missing_valid = int(valid_m["orig_idx"].isna().sum())

        raise RuntimeError(
            f"Failed to map some rows back to the original dataset. "
            f"Missing train={missing_train}, missing test={missing_test}. Missing valid={missing_valid} "
            f"This can happen if split frames were modified or if join keys aren't unique."
        )

    train_indices = train_m["orig_idx"].astype(int).tolist()
    test_indices = test_m["orig_idx"].astype(int).tolist()
    valid_indices = valid_m["orig_idx"].astype(int).tolist()

#     print(f"\n{name} Train indices:\n{train_indices}")
    print(f"\n\n{name} Validation indices:\n{valid_indices}")
#     print(f"\n\n{name} Test indices:\n{test_indices}")



sets = [
    # ADME
    ("caco2_wang", False),
    ("hia_hou", False),
    ("pampa_ncats", False),
    ("bioavailability_ma", False),
    ("lipophilicity_astrazeneca", False),
    ("solubility_aqsoldb", False),
    ("pgp_broccatelli", False),
    ("clearance_hepatocyte_az", False),
    ("bbb_martins", False),
    # Tox
    ("ld50_zhu", True),
    ("ames", True),
    ("carcinogens_lagunin", True),
    ("dili", True),
    ("skin_reaction", True),
    ("herg", True),
    ("hydrationfreeenergy_freesolv", False),
    ("vdss_lombardo", False),
    ("ppbr_az", False),
    ("cyp2c19_veith", False),
    ("cyp2d6_veith", False),
    ("cyp1a2_veith", False),
    ("cyp2c9_veith", False),
    ("cyp3a4_veith", False),
    ("half_life_obach", False),
]

for name, tox in sets:
    train_test_split(name, tox)