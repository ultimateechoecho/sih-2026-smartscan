"""BLOCK 1a - Data: download TSRD files, extract receiver + transmitter configs.

Input : TSRD .h5 files (scan mode) OR nothing (synthetic fallback)
Output: data/receiver.json          -> dwell centres (MHz) + dwell times (s)
        data/transmitter_stats.pkl  -> one row per transmitter config

CLI examples
    python -m smartscan.b1_data --list                # show scan/train files on the Hub
    python -m smartscan.b1_data --download 6          # download 6 small scan config files
    python -m smartscan.b1_data --extract             # h5 -> receiver.json + stats.pkl
    python -m smartscan.b1_data --inspect data/tsrd_raw/scan/train_scan/config_0.h5
    python -m smartscan.b1_data --synthetic           # no download: make fallback data
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (DATA_DIR, RAW_DIR, RECEIVER_PATH, STATS_PATH, TSRD_REPO_ID)


# --------------------------------------------------------------------------- #
# Download
# --------------------------------------------------------------------------- #
def list_repo_files(repo_id=TSRD_REPO_ID, prefix="scan/train_scan/", limit=20):
    from huggingface_hub import HfApi
    files = [f for f in HfApi().list_repo_files(repo_id, repo_type="dataset")
             if f.startswith(prefix) and f.endswith(".h5")]
    return sorted(files)[:limit] if limit else sorted(files)


def download(k=6, repo_id=TSRD_REPO_ID, prefix="scan/train_scan/", out_dir=RAW_DIR):
    """Download only k config files (scan files are small: ~ a few MB each)."""
    from huggingface_hub import hf_hub_download
    files = list_repo_files(repo_id, prefix, limit=k)
    paths = []
    for f in files:
        p = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=f,
                            local_dir=str(out_dir))
        print("downloaded:", p)
        paths.append(p)
    return paths


# --------------------------------------------------------------------------- #
# HDF5 reading (structure verified on scan/train_scan/config_0.h5)
# --------------------------------------------------------------------------- #
def inspect_h5(path):
    import h5py
    with h5py.File(path, "r") as f:
        def show(name, obj):
            if hasattr(obj, "shape"):
                print(f"{name:70s} {obj.shape}")
        f.visititems(show)
        print("feature_names:", [x.decode() if isinstance(x, bytes) else x
                                 for x in f["metadata/feature_names"][:]])


def read_receiver(path):
    import h5py
    with h5py.File(path, "r") as f:
        r = f["metadata/receiver"]
        return {
            "dwell_centres_mhz": [float(x) for x in r["dwell_centres_mhz"][:]],
            "dwell_times_s": [float(x) for x in r["dwell_times_s"][:]],
            "freq_range_mhz": [float(x) for x in r["freq_range_mhz"][:]],
            "source": str(path),
        }


def _get(group, key):
    return np.atleast_1d(group[key][:]).astype(float) if key in group else np.array([])


def extract_transmitters(path):
    """One record per transmitter: freq list, PRI list, PW list, scan-config keys.
    These are the emitters' TRUE generative parameters (receiver independent)."""
    import h5py
    rows = []
    with h5py.File(path, "r") as f:
        tx_root = f["metadata/transmitters"]
        for name in tx_root.keys():
            tx = tx_root[name]
            freqs = _get(tx, "frequency_config/freqs_mhz")
            pris = _get(tx, "pri_config/pris_us")
            pws = _get(tx, "pulse_width_config/pws_us")
            scan_keys = list(tx["scan_config"].keys()) if "scan_config" in tx else []
            rows.append({
                "transmitter": name, "source_file": Path(path).name,
                "freqs_mhz": freqs, "n_freqs": int(len(freqs)),
                "pris_us": pris, "n_pris": int(len(pris)),
                "pws_us": pws, "scan_keys": scan_keys,
            })
    return rows


def build_stats(paths):
    rows = []
    for p in paths:
        rows.extend(extract_transmitters(p))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Fallback data (lets every block run before the download is finished)
# --------------------------------------------------------------------------- #
def default_receiver():
    """36 dwell centres 250..17750 MHz (verified in TSRD). Dwell times: only the
    first four are documented (0.10,0.10,0.05,0.05); rest assumed 0.05 s."""
    centres = [250.0 + 500.0 * i for i in range(36)]
    times = [0.10, 0.10] + [0.05] * 34
    return {"dwell_centres_mhz": centres, "dwell_times_s": times,
            "freq_range_mhz": [500.0, 18000.0], "source": "fallback-assumed"}


def synthetic_stats(n_files=6, per_file=30, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for fi in range(n_files):
        for ti in range(per_file):
            nf = 1 if rng.random() < 0.6 else int(rng.integers(2, 7))
            npri = 1 if rng.random() < 0.7 else int(rng.integers(2, 4))
            rows.append({
                "transmitter": f"transmitters_{ti}", "source_file": f"synthetic_{fi}.h5",
                "freqs_mhz": rng.uniform(500, 18000, nf), "n_freqs": nf,
                "pris_us": np.sort(rng.uniform(200, 3000, npri)), "n_pris": npri,
                "pws_us": rng.uniform(0.5, 5, 1), "scan_keys": [],
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Load helpers used by every other block
# --------------------------------------------------------------------------- #
def save_receiver(rec, path=RECEIVER_PATH):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(rec, indent=2))


def load_receiver(path=RECEIVER_PATH):
    if Path(path).exists():
        return json.loads(Path(path).read_text())
    print("[b1_data] WARNING: receiver.json missing -> using assumed fallback receiver")
    return default_receiver()


def load_stats(path=STATS_PATH):
    if Path(path).exists():
        return pd.read_pickle(path)
    print("[b1_data] WARNING: transmitter_stats.pkl missing -> using SYNTHETIC stats")
    return synthetic_stats()


def split_stats(df, seed=0, frac=(0.6, 0.2, 0.2)):
    """Partition transmitter configs into train/val/test so test emitters are
    configs never seen in training. Groups by source file when >=3 files."""
    rng = np.random.default_rng(seed)
    n_files = df["source_file"].nunique()
    if len(df) < 10:
        return {"train": df, "val": df, "test": df}
    if n_files >= 3:
        files = np.array(sorted(df["source_file"].unique()))
        rng.shuffle(files)
        a = max(1, int(round(frac[0] * len(files))))
        b = max(a + 1, min(len(files) - 1, a + int(round(frac[1] * len(files)))))
        groups = {"train": files[:a], "val": files[a:b], "test": files[b:]}
        out = {k: df[df["source_file"].isin(v)] for k, v in groups.items()}
    else:
        idx = rng.permutation(len(df))
        a, b = int(frac[0] * len(df)), int((frac[0] + frac[1]) * len(df))
        out = {"train": df.iloc[idx[:a]], "val": df.iloc[idx[a:b]], "test": df.iloc[idx[b:]]}
    for k, v in out.items():                       # never return an empty split
        if len(v) == 0:
            out[k] = df
    return out


# --------------------------------------------------------------------------- #
# Optional: real occupancy grid from a TSRD file (sanity check / demo figure)
# --------------------------------------------------------------------------- #
def pulses_to_occupancy(path, receiver=None, slot_s=0.05):
    """ToA (us) + Frequency (MHz) -> boolean (n_bands, n_slots) occupancy grid.
    For inspection only - the scheduler never sees this."""
    import h5py
    receiver = receiver or read_receiver(path)
    centres = np.asarray(receiver["dwell_centres_mhz"])
    with h5py.File(path, "r") as f:
        data = f["data"][:]
    toa_s, freq = data[:, 0] * 1e-6, data[:, 1]
    band = np.abs(freq[:, None] - centres[None, :]).argmin(axis=1)
    slot = (toa_s / slot_s).astype(int)
    grid = np.zeros((len(centres), slot.max() + 1), bool)
    grid[band, slot] = True
    return grid


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--download", type=int, metavar="K")
    ap.add_argument("--extract", action="store_true")
    ap.add_argument("--inspect", type=str)
    ap.add_argument("--synthetic", action="store_true")
    a = ap.parse_args()

    if a.list:
        for f in list_repo_files():
            print(f)
    if a.download:
        download(a.download)
    if a.inspect:
        inspect_h5(a.inspect)
    if a.extract:
        paths = sorted(RAW_DIR.rglob("*.h5"))
        if not paths:
            sys.exit("No .h5 files under data/tsrd_raw. Run --download K first.")
        save_receiver(read_receiver(paths[0]))
        df = build_stats(paths)
        df.to_pickle(STATS_PATH)
        print(f"receiver.json saved; {len(df)} transmitter configs from {len(paths)} files")
        print(df[["n_freqs", "n_pris"]].describe().loc[["mean", "max"]])
    if a.synthetic:
        DATA_DIR.mkdir(exist_ok=True)
        save_receiver(default_receiver())
        synthetic_stats().to_pickle(STATS_PATH)
        print("synthetic fallback data written")
    if not any(vars(a).values()):
        rec, df = load_receiver(), load_stats()
        s = split_stats(df)
        print("bands:", len(rec["dwell_centres_mhz"]), "| stats rows:", len(df),
              "| split sizes:", {k: len(v) for k, v in s.items()})


if __name__ == "__main__":
    main()
