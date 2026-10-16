"""
Pear fire blight model (Maryblyt-type BHWT), fixed version of "pear fire blight".

Reads hourly weather from a local CSV (e.g. obs/ID1001.csv) instead of the
fireblight.org API, and fixes the errors found in the original code:

  1. Model['RH'] was never created (KeyError)       -> daily mean RH is added
  2. BBS marker loop appended extra rows (ValueError) -> one marker per day
  3. Missing days were filled with the station code  -> hourly gaps interpolated,
                                                        rain gaps set to 0
  4. Daily DH was grouped by UTC date                -> grouped by local date
  5. DH was attached to daily rows by position       -> joined by date
  6. EIP used an API column (GDH) that local data
     does not have                                   -> EIP uses the model's ADH
  7. Modified EIP polynomial decreases after its
     maximum (x > 262.5)                             -> held at its maximum
  8. `stage == 9` (comparison)                       -> `stage = 9`
  9. ADD/ADH kept accumulating after petal fall      -> 0 outside B1..PF
 10. Endodormancy release could switch off again
     when DVS1 dropped below 2                       -> released once, stays on
 11. Phenology / symptom dates crashed (.iloc[0])
     when not reached                                -> reported as None
 12. noteDay year taken from hourly row 3000         -> season year
 13. Negative indexes at the start of the series     -> guarded

Usage:
  python3 pear_fire_blight_fixed.py obs/ID1001.csv 2019 Output/pear_ID1001_2019
  python3 pear_fire_blight_fixed.py obs/hourly_99 2010 2011 ... 2025 Output/pear_99
  (add --chill-complete to assume endodormancy is released at season start;
   add --force-bloom START END to treat START..END as the bloom period,
   a what-if scenario for testing other models)
"""

import argparse
import datetime
import os

import numpy as np
import pandas as pd

# ------------------------------------------------------------------ #
# Model constants (unchanged from the original code)
# ------------------------------------------------------------------ #
DH_BASE = 18.3          # degree-hour base (°C)
DD_BASE = 4.4           # degree-day base for the EIP window (°C)
DD_WINDOW = 66.7        # degree-day length of the EIP window
EIP_DH = 110            # ADH giving EIP = 100 (before the modified equation)
W_RH = 80               # wetness: daily mean RH (%)
W_RAIN = 0.25           # wetness: daily rain (mm)
W_RAIN_PREV = 2.5       # wetness: previous-day rain (mm)
T_MEAN = 15.6           # temperature: daily mean (°C)
SYM_BASE = 12.7         # symptom degree-day base (°C)
CMS_DD, CBS_DD, BBS_DD = 109, 57, 57
GT_DVS, B1_DVS, FB_DVS = 0.4078, 0.6721, 0.8639
PF_DAYS = 15

EIP_A, EIP_B, EIP_C = -0.0031, 1.6273, -1.7298
EIP_X_MAX = -EIP_B / (2 * EIP_A)   # vertex of the modified EIP polynomial


MAX_GAP_HOURS = 6       # interpolation limit from each side of a gap (gaps up to 12 h filled)


def read_hourly(path):
    """Read hourly weather and return tm, ta, rn, hm (invalid values -> NaN).

    Supported formats
      - KMA download (cp949): 지점, 일시, 기온, ..., 강수량, ..., 습도, ...
      - Id, data_tm, ta, hm, rn
    """
    try:
        raw = pd.read_csv(path)
    except UnicodeDecodeError:
        raw = pd.read_csv(path, encoding="cp949")

    if "data_tm" in raw.columns:
        h = raw.rename(columns={"data_tm": "tm"})[["tm", "ta", "rn", "hm"]]
    else:
        cols = {raw.columns[1]: "tm", raw.columns[2]: "ta",
                raw.columns[5]: "rn", raw.columns[8]: "hm"}
        h = raw.rename(columns=cols)[["tm", "ta", "rn", "hm"]]
    h["tm"] = pd.to_datetime(h["tm"])          # local time (KST), no tz
    h = h.drop_duplicates("tm").sort_values("tm").reset_index(drop=True)

    # Missing-value codes (e.g. -999) and impossible values -> NaN.
    # A record with a missing temperature is treated as missing as a whole.
    bad_ta = (h["ta"] < -50) | (h["ta"] > 60)
    h.loc[bad_ta, ["ta", "rn", "hm"]] = np.nan
    h.loc[(h["hm"] < 0) | (h["hm"] > 100), "hm"] = np.nan
    h.loc[h["rn"] < 0, "rn"] = np.nan
    return h


def season_hourly(h, year):
    """Hourly series for one season (Oct 1 of year-1 .. Jun 30 of year).

    Missing hours and invalid values: temperature / RH interpolated, up to
    MAX_GAP_HOURS from each side of a gap (gaps up to 2 x MAX_GAP_HOURS are
    filled); rain set to 0. Longer gaps raise an error.
    """
    s = h[(h["tm"] >= pd.Timestamp(year - 1, 10, 1)) &
          (h["tm"] < pd.Timestamp(year, 7, 1))].set_index("tm")
    if s.empty:
        raise RuntimeError(f"No data for season {year}.")
    full = pd.date_range(s.index.min().normalize(),
                         s.index.max().normalize() + pd.Timedelta(hours=23),
                         freq="h")
    s = s.reindex(full)
    n_missing = int(s[["ta", "hm"]].isna().any(axis=1).sum())
    s[["ta", "hm"]] = s[["ta", "hm"]].interpolate(
        limit=MAX_GAP_HOURS, limit_direction="both", limit_area=None)
    s["rn"] = s["rn"].fillna(0)
    if s[["ta", "hm"]].isna().any().any():
        raise RuntimeError(f"Season {year}: gap longer than {MAX_GAP_HOURS} h.")
    s.index.name = "tm"
    return s.reset_index(), n_missing


def daily_from_hourly(h):
    """Daily weather and degree-hours, grouped by local calendar date."""
    h = h.copy()
    h["Date"] = h["tm"].dt.normalize()
    h["dh"] = np.maximum(h["ta"] - DH_BASE, 0)
    d = h.groupby("Date").agg(ta=("ta", "mean"), tx=("ta", "max"),
                              tn=("ta", "min"), rn=("rn", "sum"),
                              hm=("hm", "mean"), DH=("dh", "sum"))
    return d.reset_index().rename(columns={"Date": "tm"})


def phenology(h, year, chill_complete=False):
    """Hourly DVS1 (chilling) and DVS2 (heat) -> GT, B1, FB, PF dates."""
    T = h["ta"].to_numpy()

    dvr1 = np.where((T > -6) & (T < 12),
                    -2.533e-4 + 1.671e-3 *
                    np.exp(-0.5 * (np.log((T + 273) / 275.2) / 0.01595) ** 2),
                    0.0)
    dvr1[0] = 0
    dvs1 = np.cumsum(dvr1)

    note_day = pd.Timestamp(year, 2, 15)
    released = np.maximum.accumulate(dvs1 >= 2) | chill_complete
    active = (h["tm"].to_numpy() >= note_day.to_datetime64()) | released
    dvr2 = np.where(active & (T >= 5),
                    4.646e-4 + 5.022e-3 /
                    (1 + np.exp(-(T + 273 - 292.3) / 3.616)),
                    0.0)
    dvs2 = np.cumsum(dvr2)

    def first_date(threshold):
        idx = np.flatnonzero(dvs2 >= threshold)
        return None if len(idx) == 0 else h["tm"].iloc[idx[0]].normalize()

    gt, b1, fb = first_date(GT_DVS), first_date(B1_DVS), first_date(FB_DVS)
    pf = None if fb is None else fb + datetime.timedelta(days=PF_DAYS)
    release_time = (h["tm"].iloc[np.flatnonzero(released)[0]]
                    if released.any() else None)
    return dict(GT=gt, B1=b1, FB=fb, PF=pf, DVS1_max=float(dvs1.max()),
                chill_release=release_time)


def growth_stage(dates, ph):
    stage, fbd, out = 1, 0, []
    for day in dates:
        if stage == 1 and ph["GT"] is not None and day >= ph["GT"]:
            stage = 2
        elif stage == 2 and ph["B1"] is not None and day >= ph["B1"]:
            stage = 3
        elif stage == 3 and ph["FB"] is not None and day >= ph["FB"]:
            stage = 4
        elif stage == 4 and fbd >= 15:
            stage = 5
        elif stage == 5 and fbd >= 21:
            stage = 6
        elif stage == 6 and fbd >= 96:
            stage = 7
        elif stage == 7 and fbd >= 201:
            stage = 8
        elif stage == 8 and fbd >= 211:
            stage = 9
        if stage >= 4:
            fbd += 1
        out.append(stage)
    return out


def add_adh(d, B1, PF):
    """Sliding-window ADD (>4.4 °C, 66.7 DD) and weighted ADH, as in the
    original algorithm, restricted to B1 <= day < PF."""
    n = len(d)
    tx, tn = d["tx"].to_numpy(), d["tn"].to_numpy()
    tavg = (np.minimum(tx, 32.2) + np.maximum(tn, 4.4)) / 2
    dd = np.maximum(tavg - DD_BASE, 0)
    in_bloom = ((d["tm"] >= B1) & (d["tm"] < PF)).to_numpy()

    weight = np.zeros(n)
    snote = np.zeros(n, dtype=int)
    for i in range(n):
        tx1 = tx[i - 1] if i >= 1 else np.inf
        tx2 = tx[i - 2] if i >= 2 else np.inf
        if not in_bloom[i] or tn[i] < 0 or (tx2 < DH_BASE and tx1 < DH_BASE
                                            and tx[i] < DH_BASE):
            weight[i] = 0
        elif tx1 < DH_BASE and tx[i] < DH_BASE:
            weight[i] = 1 / 2
        elif tx[i] < DH_BASE:
            weight[i] = 2 / 3
        else:
            weight[i] = 1

        if not in_bloom[i]:
            snote[i] = 0
        elif tn[i] < 0:
            snote[i] = 3
        elif weight[i] == 0:
            snote[i] = 2
        elif tx1 < DH_BASE:
            snote[i] = 1

    dates = d["tm"].tolist()
    pos = {day: k for k, day in enumerate(dates)}
    ADD, ADH = np.zeros(n), np.zeros(n)
    add = adh = 0.0
    sday = B1
    for i in range(n):
        if not in_bloom[i]:
            add = adh = 0.0
            continue
        if snote[i] == 1 and adh == 0:
            sday = dates[i]
        elif snote[i] == 2:
            sday, add, adh = dates[i], 0.0, 0.0
        elif snote[i] == 3:
            sday, add, adh = dates[i] + datetime.timedelta(days=1), 0.0, 0.0

        while add >= DD_WINDOW and sday in pos:
            add -= dd[pos[sday]]
            sday += datetime.timedelta(days=1)

        dh_acc = dd_acc = 0.0
        if sday in pos:
            for j in range(pos[sday], i + 1):
                dh_acc = (dh_acc + d["DH"].iat[j]) * weight[j]
                dd_acc += dd[j]
        adh, add = dh_acc, dd_acc
        ADD[i], ADH[i] = add, adh
    return dd, weight, snote, ADD, ADH


def eip_from_adh(adh):
    x = np.asarray(adh) * 100 / EIP_DH
    x_capped = np.minimum(x, EIP_X_MAX)
    eip = np.maximum(EIP_A * x_capped ** 2 + EIP_B * x_capped + EIP_C, 0)
    eip[x == 0] = 0
    return eip, int((x > EIP_X_MAX).sum())


def symptoms(d, start, infection_days):
    """CMS, CBS (from GT) and BBS (from each infection day)."""
    tavg = (np.minimum(32.2, d["tx"]) + np.maximum(SYM_BASE, d["tn"])) / 2
    sdd = np.maximum(tavg - SYM_BASE, 0).to_numpy()
    after = (d["tm"] >= start).to_numpy()

    cms_acc = np.cumsum(np.where(after, sdd, 0))
    cms_idx = np.flatnonzero(after & (cms_acc >= CMS_DD))
    CMS = d["tm"].iat[cms_idx[0]] if len(cms_idx) else None

    CBS = None
    if CMS is not None:
        cbs_acc = np.cumsum(np.where(d["tm"] > CMS, sdd, 0))
        cbs_idx = np.flatnonzero(cbs_acc >= CBS_DD)
        CBS = d["tm"].iat[cbs_idx[0]] if len(cbs_idx) else None

    # BBS: one infection at a time, as in the original logic
    bbs = np.zeros(len(d))
    events = list(infection_days)
    a = 0
    for i in range(len(d)):
        if a >= len(events) or d["tm"].iat[i] <= events[a]:
            continue
        j0 = int(np.flatnonzero(d["tm"] == events[a])[0]) + 1
        bbs[i] = sdd[j0:i + 1].sum()
        if bbs[i] >= BBS_DD:
            a += 1
    BBS = d["tm"][bbs >= BBS_DD].tolist()
    return CMS, CBS, BBS, bbs


def run(path, year, chill_complete=False, force_bloom=None):
    hourly = read_hourly(path) if isinstance(path, str) else path
    h, n_missing = season_hourly(hourly, year)
    d = daily_from_hourly(h)

    ph = phenology(h, year, chill_complete)
    if force_bloom is not None:
        ph["B1"] = pd.Timestamp(force_bloom[0])
        ph["PF"] = pd.Timestamp(force_bloom[1]) + datetime.timedelta(days=1)
    if ph["B1"] is None or ph["FB"] is None:
        raise RuntimeError("Bloom was not reached in the data period.")

    d["growth"] = growth_stage(d["tm"], ph)
    d["Tavg-4.4"], d["weight"], d["snote"], d["ADD"], d["ADH"] = \
        add_adh(d, ph["B1"], ph["PF"])
    d["EIP"], n_capped = eip_from_adh(d["ADH"])

    rn_prev = d["rn"].shift(1, fill_value=0)
    d["B"] = (d["tm"] >= ph["B1"]) & (d["tm"] < ph["PF"])
    d["H"] = d["B"] & (d["EIP"] >= 100)
    d["W"] = d["B"] & ((d["hm"] >= W_RH) | (d["rn"] >= W_RAIN) |
                       (rn_prev >= W_RAIN_PREV))
    d["T"] = d["B"] & (d["ta"] >= T_MEAN)
    n_plus = d[["B", "H", "W", "T"]].sum(axis=1)
    d["BIR"] = np.where(d["B"], n_plus * 25, 0)
    d["level"] = np.select([n_plus == 1, n_plus == 2, n_plus == 3, n_plus == 4],
                           ["L", "M", "H", "I"], "")
    d["infection"] = n_plus == 4

    CMS, CBS, BBS, bbs = symptoms(d, ph["GT"], d.loc[d["infection"], "tm"])
    d["BBS_DD"] = bbs
    d["CMS"] = d["tm"] == CMS
    d["CBS"] = d["tm"] == CBS
    d["BBS"] = d["tm"].isin(BBS)

    info = dict(ph, CMS=CMS, CBS=CBS, BBS=BBS, missing_hours=n_missing,
                eip_capped_days=n_capped,
                data_start=h["tm"].min(), data_end=h["tm"].max())
    return d, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("weather")
    ap.add_argument("years", type=int, nargs="+",
                    help="season year(s): 2019 = 2018-10-01 .. 2019-06-30")
    ap.add_argument("out_dir")
    ap.add_argument("--chill-complete", action="store_true")
    ap.add_argument("--force-bloom", nargs=2, metavar=("START", "END"))
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    tag = ("_chillcomplete" if a.chill_complete else "") + \
          ("_forcebloom" if a.force_bloom else "")
    hourly = read_hourly(a.weather)
    rows = []
    for year in a.years:
        try:
            d, info = run(hourly, year, a.chill_complete, a.force_bloom)
        except RuntimeError as err:
            print(f"[skip] {err}")
            rows.append(dict(season=year, status=str(err)))
            continue
        d.to_csv(os.path.join(a.out_dir, f"pear_daily_{year}{tag}.csv"),
                 index=False, float_format="%.4f")
        with open(os.path.join(a.out_dir, f"pear_summary_{year}{tag}.txt"),
                  "w") as f:
            for k, v in info.items():
                f.write(f"{k}: {v}\n")
        b = d[d["B"]]
        rows.append(dict(season=year, status="ok",
                         GT=info["GT"].date(), B1=info["B1"].date(),
                         FB=info["FB"].date(), PF=info["PF"].date(),
                         chill_release=info["chill_release"],
                         bloom_days=len(b), H=int(b["H"].sum()),
                         W=int(b["W"].sum()), T=int(b["T"].sum()),
                         infection_days=int(b["infection"].sum()),
                         missing_hours=info["missing_hours"]))
        if len(a.years) == 1:
            for k, v in info.items():
                print(f"{k}: {v}")
    if len(a.years) > 1:
        summary = pd.DataFrame(rows)
        summary.to_csv(os.path.join(a.out_dir, f"pear_seasons{tag}.csv"),
                       index=False)
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
