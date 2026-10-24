"""Map the misses from the binary affected/not-affected validation.

Reproduces Integrator.validate_exposure_method -- the surveyed July 2021 depth
raster through the same exposure and FIS -- then joins the result to the
damage placards and asks, for every building the survey recorded as damaged but
the model called not_affected, WHY it was missed.

Each miss is attributed by recomputing the memberships from the same h_max the
FIS saw, so the attribution cannot drift from the live cutoffs:

  dry       depth_max below the wet cutoff -- the model saw no water at all.
            Floor level, sediment, non-flood damage or a geolocation error.
  shallow   h_max fuzzified low. Real but shallow water, called low by the ARR
            cutoffs themselves: the hazard-versus-impact gap, not a model
            error.

Those two are now exhaustive. A third class, "demoted", existed while a per-object FIS
ran a rule base in which a low h_mean pulled a wet ring back down to low; with
h_max the only indicator there is nothing left to do the demoting.

Unlike map_misclassifications.py this keeps the UNDAMAGED buildings, so the
false alarms are on the map too -- a miss map alone will always argue for
warning more.

Writes out/missed_buildings.csv and out/miss_map.html.
"""
import sys

PROJECT_ROOT = "/home/91/23016891/projects/westport-ibw"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import folium
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio as rio

from modules.lib import PLACRD_FILE_PATH, SURVEYED_FLOOD_DEPTH_JUL_2021
from modules.impact_module.risk_module import ImpactModule, WET_DEPTH_MIN
from modules.impact_module.fuzzy_risk import fuzzify
from modules.forecast_runner.forecast_runner_helper import (
    AFFECTED, NOT_AFFECTED, _damage_to_affected, _hazards_to_affected,
)

OUT_CSV = f"{PROJECT_ROOT}/out/missed_buildings.csv"
OUT_HTML = f"{PROJECT_ROOT}/out/miss_map.html"

# Parameters the validation was run with, so the map reproduces the numbers in
# the log rather than a differently-tuned run. NOTE buffer_m is 2.0, not the
# 0.0 that main.py defaults --buffer_multiplier to; it changes the contingency
# table materially (0.0 gives 417/230/239/1358 against 2.0's 428/219/256/1341),
# so it has to be stated rather than inherited from a default.
MEAN_THRESHOLD, MAX_THRESHOLD, BUFFER_M = 0.05, 0.15, 2.0

TIER_NAME = {1: "low", 2: "medium", 3: "high"}

# Why a miss was missed, in the order the tests are applied: a dry ring also
# fuzzifies low, so it has to be separated out first.
MISS_REASONS = {
    "dry": ("#7b241c", "Dry ring - model saw no water"),
    "shallow": ("#e67e22", "Shallow - h_max fuzzified low"),
}
OUTCOME_COLOURS = {
    "false_alarm": ("#f1c40f", "False alarm"),
    "hit": ("#27ae60", "Hit"),
    "correct_negative": ("#7f8c8d", "Correct negative"),
}

# ── PREDICTION ────────────────────────────────────────────────────────────────

with rio.open(SURVEYED_FLOOD_DEPTH_JUL_2021) as src:
    surveyed_depth = src.read(1).astype(float)

impact = ImpactModule("miss_analysis").get_impact_forecast(
    {"17-07-2021 00:00:00": [surveyed_depth]},
    MEAN_THRESHOLD, MAX_THRESHOLD, BUFFER_M,
    rasterfile=SURVEYED_FLOOD_DEPTH_JUL_2021,
)
predicted = next(iter(impact["building_impact"].values())).copy()
predicted = predicted[predicted["num_ring_cells"] > 0]
print(f"predicted for {len(predicted)} buildings with sampled cells")

# ── OBSERVATION ───────────────────────────────────────────────────────────────
# Every surveyed building, damaged or not: the undamaged ones are the correct
# negatives, without which false alarms cannot be counted.

placards = gpd.read_file(PLACRD_FILE_PATH)
placards = placards[placards["DamageNb"].isin([0.0, 1.0, 2.0, 3.0])]
placards["observed_affected"] = placards["DamageNb"].apply(_damage_to_affected)

scored = placards[["building_id", "DamageNb", "damage", "observed_affected"]].merge(
    predicted.drop(columns="geometry"), on="building_id", how="inner"
).dropna(subset=["building_hazard_level", "observed_affected"])
scored = scored.merge(predicted[["building_id", "geometry"]], on="building_id")
scored = gpd.GeoDataFrame(scored, geometry="geometry", crs=predicted.crs)

# Building tier alone, matching the headline validation; the property tier is
# carried through for inspection but no longer votes.
scored["predicted_affected"] = _hazards_to_affected(scored[["building_hazard_level"]])

is_obs = scored["observed_affected"].eq(AFFECTED)
is_pred = scored["predicted_affected"].eq(AFFECTED)
scored["outcome"] = np.select(
    [is_obs & is_pred, is_obs & ~is_pred, ~is_obs & is_pred],
    ["hit", "miss", "false_alarm"], default="correct_negative",
)

counts = scored["outcome"].value_counts()
hits, misses = int(counts.get("hit", 0)), int(counts.get("miss", 0))
fa, cn = int(counts.get("false_alarm", 0)), int(counts.get("correct_negative", 0))
pod = hits / (hits + misses) if hits + misses else 0.0
far = fa / (hits + fa) if hits + fa else 0.0
print(f"\nhits={hits} misses={misses} false_alarms={fa} correct_negatives={cn}")
print(f"POD={pod:.2f} FAR={far:.2f}   (should match the validation log)")

# ── WHY WERE THEY MISSED ──────────────────────────────────────────────────────
# Attribution from the memberships the FIS saw. With h_max the only indicator,
# a miss means its ring never fuzzified above low: either the model put no water
# there at all ("dry"), or it put some there but not enough ("shallow"). Dry is
# tested first because a dry ring also fuzzifies low.

h_max = scored["depth_max"].to_numpy(dtype=float)
h_max_is_low = fuzzify(h_max, "building").argmax(axis=-1) == 0
scored["miss_reason"] = np.where(
    scored["outcome"].ne("miss"), "",
    np.select([h_max < WET_DEPTH_MIN, h_max_is_low], ["dry", "shallow"],
              default=""),
)

missed = scored[scored["outcome"].eq("miss")]
print(f"\nwhy the {len(missed)} misses were missed:")
for reason, (_, label) in MISS_REASONS.items():
    n = int((missed["miss_reason"] == reason).sum())
    if n:
        sub = missed[missed["miss_reason"] == reason]
        print(f"  {label:48} {n:4} ({n / len(missed):5.1%})  "
              f"median h_max {sub['depth_max'].median():.2f} m, "
              f"h_mean {sub['depth_mean'].median():.2f} m")

print("\nmisses by observed damage class:")
by_damage = scored.groupby("damage").agg(
    n=("outcome", "size"), missed=("outcome", lambda s: (s == "miss").sum()))
by_damage = by_damage[by_damage["missed"] > 0].sort_values("n", ascending=False)
by_damage["miss_rate"] = (by_damage["missed"] / by_damage["n"]).round(2)
print(by_damage.to_string())

if "wet_fraction" in scored.columns:
    print("\nring wetted fraction (the coverage half of the old depth_mean):")
    print(scored.groupby("outcome")["wet_fraction"].describe()[
        ["count", "25%", "50%", "75%"]].round(2).to_string())

# ── RECORD ────────────────────────────────────────────────────────────────────

columns = ["building_id", "damage", "DamageNb", "miss_reason",
           "depth_max", "depth_mean", "num_ring_cells", "num_wet_cells",
           "wet_fraction", "building_hazard_level", "building_hazard_likelihood",
           "property_hazard_level", "property_known", "street_hazard_level",
           "overall_building_hazard_level", "overall_building_hazard_likelihood", "final_risk_level"]
columns = [c for c in columns if c in missed.columns]
missed[columns].sort_values(["miss_reason", "depth_max"]).to_csv(OUT_CSV, index=False)
print(f"\nwrote {OUT_CSV}: {len(missed)} rows")

# ── MAP ───────────────────────────────────────────────────────────────────────

drawn = scored.to_crs(epsg=4326)
for col in ("depth_max", "depth_mean", "wet_fraction", "building_hazard_likelihood"):
    if col in drawn:
        drawn[col] = drawn[col].round(2)
for col in ("property_hazard_level", "street_hazard_level"):
    drawn[col] = (drawn[col].fillna(-1).astype(int).map(TIER_NAME)
                  .fillna("not assessed"))

FIELDS = ["building_id", "damage", "miss_reason", "depth_max", "depth_mean",
          "wet_fraction", "num_ring_cells", "building_hazard_level",
          "property_hazard_level", "street_hazard_level", "overall_building_hazard_level"]
ALIASES = ["Building", "Damage record", "Why missed", "Rule fired",
           "Depth max (m)", "Depth mean, wet cells (m)", "Ring wetted fraction",
           "Ring cells", "Building tier", "Property tier", "Street tier",
           "Combined risk"]
keep = [i for i, f in enumerate(FIELDS) if f in drawn.columns]
FIELDS = [FIELDS[i] for i in keep]
ALIASES = [ALIASES[i] for i in keep]

centre = drawn.union_all().centroid
m = folium.Map(location=[centre.y, centre.x], zoom_start=15,
               tiles="Esri.WorldImagery", attr="Esri")

# The surveyed depth raster underneath, so a miss can be read against the water
# that was actually there rather than against the basemap alone.
with rio.open(SURVEYED_FLOOD_DEPTH_JUL_2021) as src:
    band = src.read(1).astype(float)
    bounds = src.bounds
    raster_crs = src.crs
wet = np.where(np.isnan(band) | (band < WET_DEPTH_MIN), np.nan, band)
if np.isfinite(wet).any():
    vmax = float(np.nanpercentile(wet, 98)) or 1.0
    norm = np.clip(wet / vmax, 0, 1)
    dry = np.isnan(norm)
    # Fill before the uint8 cast: casting NaN is undefined and leaves garbage in
    # the channel even where alpha hides it.
    shade = np.nan_to_num(norm, nan=0.0)
    rgba = np.zeros(norm.shape + (4,), dtype=np.uint8)
    rgba[..., 2] = 255                                   # blue ramp
    rgba[..., 1] = (120 * (1 - shade)).astype(np.uint8)
    rgba[..., 3] = np.where(dry, 0, 150).astype(np.uint8)
    sw = gpd.GeoSeries.from_xy([bounds.left], [bounds.bottom], crs=raster_crs
                               ).to_crs(4326)
    ne = gpd.GeoSeries.from_xy([bounds.right], [bounds.top], crs=raster_crs
                               ).to_crs(4326)
    folium.raster_layers.ImageOverlay(
        rgba, name=f"Surveyed depth (0-{vmax:.1f} m)", opacity=0.7,
        bounds=[[sw.y[0], sw.x[0]], [ne.y[0], ne.x[0]]],
    ).add_to(m)

def add_layer(subset, colour, label, show):
    if subset.empty:
        return
    folium.GeoJson(
        subset[FIELDS + ["geometry"]],
        name=f"{label} ({len(subset)})",
        show=show,
        style_function=lambda x, c=colour: {"fillColor": c, "color": "black",
                                            "weight": 0.6, "fillOpacity": 0.9},
        tooltip=folium.GeoJsonTooltip(fields=FIELDS, aliases=ALIASES),
    ).add_to(m)

# Misses split by reason and shown by default -- they are the question.
for reason, (colour, label) in MISS_REASONS.items():
    add_layer(drawn[drawn["miss_reason"] == reason], colour, label, show=True)
# Everything else is context, off by default.
for outcome, (colour, label) in OUTCOME_COLOURS.items():
    add_layer(drawn[drawn["outcome"] == outcome], colour, label, show=False)

folium.LayerControl(collapsed=False).add_to(m)

legend = "".join(
    f'<i style="background:{MISS_REASONS[r][0]};width:14px;height:14px;'
    f'display:inline-block;margin-right:6px;"></i> {MISS_REASONS[r][1]} '
    f'({int((missed["miss_reason"] == r).sum())})<br>'
    for r in MISS_REASONS if (missed["miss_reason"] == r).any()
)
m.get_root().html.add_child(folium.Element(f"""
<div style="position: fixed; bottom: 40px; left: 40px; z-index: 1000;
     background-color: white; padding: 12px; border-radius: 6px;
     border: 1px solid grey; font-size: 13px; max-width: 380px;">
  <b>Why the model missed damaged buildings</b><br>
  <span style="color:#7f8c8d">{len(scored)} surveyed buildings &middot;
  POD {pod:.2f} &middot; FAR {far:.2f} &middot; {len(missed)} misses</span><br><br>
  {legend}
  <hr style="margin:6px 0">
  <span style="color:#7f8c8d">Hits, false alarms and correct negatives are in
  the layer control, off by default. Toggle the surveyed depth raster to see
  the water each miss actually sat in.</span>
</div>
"""))

m.save(OUT_HTML)
print(f"saved {OUT_HTML}")
