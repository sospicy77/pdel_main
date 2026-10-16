"""
Daily features derived from hourly weather, for v9 add-on modules.

  lwd_h : hours with RH >= 90 %  (leaf-wetness-duration proxy, dew wetting)
  dh183 : degree-hours above 18.3 °C (heat accumulation, as in Maryblyt EIP)

Uses the same reading / cleaning / gap filling as pear_fire_blight_fixed.py.

Usage:
  python3 make_hourly_features.py obs/hourly_99 2010 ... 2025 Output/pear_99/hourly_features.csv
"""

import sys

import numpy as np
import pandas as pd

from pear_fire_blight_fixed import read_hourly, season_hourly

LWD_RH = 90.0
DH_BASE = 18.3


def main():
    path, years, out = sys.argv[1], [int(y) for y in sys.argv[2:-1]], sys.argv[-1]
    hourly = read_hourly(path)
    rows = []
    for year in years:
        h, _ = season_hourly(hourly, year)
        h["date"] = h["tm"].dt.normalize()
        h["wet"] = (h["hm"] >= LWD_RH).astype(int)
        h["dh"] = np.maximum(h["ta"] - DH_BASE, 0)
        d = h.groupby("date").agg(lwd_h=("wet", "sum"), dh183=("dh", "sum")).reset_index()
        d.insert(0, "season", year)
        rows.append(d)
    pd.concat(rows).to_csv(out, index=False, float_format="%.4f")


if __name__ == "__main__":
    main()
