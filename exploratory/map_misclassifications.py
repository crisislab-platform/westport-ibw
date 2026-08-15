"""Run the surveyed-damage validation and map what the model got wrong.

Reproduces Integrator.validate_exposure_method -- the surveyed July 2021 depth
raster through the same exposure and FIS -- then joins the result to the
damage placards and records every building whose predicted tier differs from the
observed one.

Scored on building_hazard_level, the building's own crisp cutoff tier, because the
placards record structural damage. The combined risk_level is carried along for
inspection but folds in street passability, which the survey never measured.

Writes out/misclassified_buildings.csv and out/misclassification_map.html.
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
from sklearn.metrics import cohen_kappa_score

from modules.lib import PLACRD_FILE_PATH, SURVEYED_FLOOD_DEPTH_JUL_2021
from modules.impact_module.impact_module import ImpactModule
from modules.forecast_runner.forecast_runner import _damage_to_true_risk_level

OUT_CSV = f"{PROJECT_ROOT}/out/misclassified_buildings.csv"
OUT_HTML = f"{PROJECT_ROOT}/out/misclassification_map.html"

TIER_NAME = {1: "Yellow", 2: "Orange", 3: "Red"}
# Signed tier error: how far the prediction sits from the observation, and in
# which direction. Over-prediction is the safe failure, under-prediction is not.
ERROR_COLOURS = {
    2: ("#7b241c", "Over by 2 tiers"),
    1: ("#e67e22", "Over by 1 tier"),
    0: ("#27ae60", "Correct"),
    -1: ("#5dade2", "Under by 1 tier"),
    -2: ("#1a5276", "Under by 2 tiers"),
}

# ── PREDICTION ────────────────────────────────────────────────────────────────

with rio.open(SURVEYED_FLOOD_DEPTH_JUL_2021) as src:
    surveyed_depth = src.read(1).astype(float)

impact = ImpactModule("misclassification").get_impact_forecast(
    {"17-07-2021 00:00:00": [surveyed_depth]},
    rasterfile=SURVEYED_FLOOD_DEPTH_JUL_2021,
)
predicted = next(iter(impact["building_impact"].values())).copy()
predicted = predicted[predicted["num_ring_cells"] > 0]
print(f"predicted for {len(predicted)} buildings with sampled cells")

# ── OBSERVATION ───────────────────────────────────────────────────────────────
# Same filter the validation uses: only surveyed buildings that were damaged.

placards = gpd.read_file(PLACRD_FILE_PATH)
placards = placards[placards["DamageNb"].isin([1.0, 2.0, 3.0])]
placards["true_risk_level"] = placards["DamageNb"].apply(_damage_to_true_risk_level)

scored = placards[["building_id", "DamageNb", "damage", "true_risk_level"]].merge(
    predicted.drop(columns="geometry"), on="building_id", how="inner"
).dropna(subset=["building_hazard_level", "true_risk_level"])
scored = scored.merge(predicted[["building_id", "geometry"]], on="building_id")
scored = gpd.GeoDataFrame(scored, geometry="geometry", crs=predicted.crs)

scored["predicted_tier"] = scored["building_hazard_level"].astype(int)
scored["observed_tier"] = scored["true_risk_level"].astype(int)
scored["tier_error"] = scored["predicted_tier"] - scored["observed_tier"]
scored["observed"] = scored["observed_tier"].map(TIER_NAME)
scored["predicted"] = scored["predicted_tier"].map(TIER_NAME)

qwk = cohen_kappa_score(scored["observed_tier"], scored["predicted_tier"],
                        weights="quadratic")
wrong = scored[scored["tier_error"] != 0]
print(f"\nscored {len(scored)} buildings | QWK {qwk:.3f}")
print(f"misclassified {len(wrong)} ({len(wrong) / len(scored):.1%})")
print("\ntier error:")
for error, (_, label) in sorted(ERROR_COLOURS.items(), reverse=True):
    n = int((scored["tier_error"] == error).sum())
    if n:
        print(f"  {label:18} {n:4} ({n / len(scored):5.1%})")
print("\nconfusion (rows observed, columns predicted):")
print(pd.crosstab(scored["observed"], scored["predicted"]).to_string())

# ── RECORD ────────────────────────────────────────────────────────────────────

columns = ["building_id", "observed", "predicted", "tier_error", "DamageNb", "damage",
           "depth_max", "depth_mean", "num_ring_cells",
           "building_hazard_level", "building_hazard_likelihood",
           "property_hazard_level", "property_known",
           "street_hazard_level", "street_known",
           "overall_building_hazard_level", "overall_building_hazard_likelihood", "overall_building_hazard_rule", "final_risk_level"]
columns = [c for c in columns if c in scored.columns]
wrong[columns].sort_values(["tier_error", "building_id"]).to_csv(OUT_CSV, index=False)
print(f"\nwrote {OUT_CSV}: {len(wrong)} rows")

# ── MAP ───────────────────────────────────────────────────────────────────────

drawn = scored.to_crs(epsg=4326)
for col in ("depth_max", "depth_mean", "building_hazard_likelihood", "overall_building_hazard_likelihood"):
    if col in drawn:
        drawn[col] = drawn[col].round(2)
for col in ("property_hazard_level", "street_hazard_level"):
    drawn[col] = drawn[col].fillna(-1).astype(int).map(TIER_NAME).fillna("not assessed")

FIELDS = ["building_id", "observed", "predicted", "tier_error", "damage",
          "depth_max", "depth_mean", "building_hazard_likelihood",
          "property_hazard_level", "street_hazard_level", "overall_building_hazard_level", "overall_building_hazard_rule"]
ALIASES = ["Building", "Observed", "Predicted", "Tier error", "Damage record",
           "Depth max (m)", "Depth mean (m)", "Confidence",
           "Property tier", "Street tier", "Combined risk", "Rule fired"]
FIELDS = [f for f in FIELDS if f in drawn.columns]
ALIASES = ALIASES[:len(FIELDS)]

centre = drawn.union_all().centroid
m = folium.Map(location=[centre.y, centre.x], zoom_start=15,
               tiles="Esri.WorldImagery", attr="Esri")

for error in (2, 1, 0, -1, -2):
    subset = drawn[drawn["tier_error"] == error]
    if subset.empty:
        continue
    colour, label = ERROR_COLOURS[error]
    folium.GeoJson(
        subset[FIELDS + ["geometry"]],
        name=f"{label} ({len(subset)})",
        # Correct ones are drawn but hidden: they are the backdrop, and the
        # question is what went wrong.
        show=error != 0,
        style_function=lambda x, c=colour: {"fillColor": c, "color": "black",
                                            "weight": 0.6, "fillOpacity": 0.85},
        tooltip=folium.GeoJsonTooltip(fields=FIELDS, aliases=ALIASES),
    ).add_to(m)

folium.LayerControl(collapsed=False).add_to(m)

legend = "".join(
    f'<i style="background:{ERROR_COLOURS[e][0]};width:14px;height:14px;'
    f'display:inline-block;margin-right:6px;"></i> {ERROR_COLOURS[e][1]} '
    f'({int((scored["tier_error"] == e).sum())})<br>'
    for e in (2, 1, 0, -1, -2) if (scored["tier_error"] == e).any()
)
m.get_root().html.add_child(folium.Element(f"""
<div style="position: fixed; bottom: 40px; left: 40px; z-index: 1000;
     background-color: white; padding: 12px; border-radius: 6px;
     border: 1px solid grey; font-size: 13px;">
  <b>Predicted vs surveyed damage tier</b><br>
  <span style="color:#7f8c8d">{len(scored)} surveyed buildings, QWK {qwk:.3f}</span><br><br>
  {legend}
  <hr style="margin:6px 0">
  <span style="color:#7f8c8d">Blue = model under-warned, orange/red = over-warned.<br>
  Hover for depths, the property and street tiers, and the rule that fired.</span>
</div>
"""))

m.save(OUT_HTML)
print(f"saved {OUT_HTML}")
