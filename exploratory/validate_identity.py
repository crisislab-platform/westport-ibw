"""Control experiment: feed the SIMULATED map back in as the forecast.

Replaces the flow-matching surrogate with an oracle that returns the ground
truth for each valid time, then runs the ordinary validation path. Everything
downstream is unchanged -- the same exposure sampling, the same FIS, the same
compute_metrics -- so the scores are a property of the harness alone.

A sound harness must return qwk = 1.0 and zero depth error. Anything else is a
bug in the validation rather than in the model: a timestep misalignment, a ring
sampled differently on the two sides, or an ensemble axis handled asymmetrically.

    python exploratory/validate_identity.py [scenario_id] [n_timesteps]
"""
import sys, os, logging
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch

from modules.bc_forecast_module.bc_module import BC_Module
from modules.impact_module.impact_module import ImpactModule
from modules.forecast_runner.forecast_runner import Integrator
from modules.utils.spatial_operators.operators import load_flood_map

logging.basicConfig(level=logging.WARNING)


class GroundTruthOracle:
    """Same interface as FlowMatchingModelClient, but returns the truth.

    One "member", so the ensemble axis is length 1 and the pooled reductions
    collapse to that single map -- which is what makes the comparison an
    identity rather than an ensemble-versus-deterministic test.
    """

    def __init__(self, bc_module):
        self.bc_module = bc_module

    def iter_hazard_forecast(self, bc_forecast):
        for timestamp in bc_forecast:
            gt = load_flood_map(
                self.bc_module.flood_map_dict[timestamp], self.bc_module.event_name
            )
            yield timestamp, torch.from_numpy(np.asarray(gt, dtype=np.float32)[None])


def main(scenario_id=57, n_timesteps=6):
    integrator = Integrator.__new__(Integrator)      # skip loading the surrogate
    integrator.scenario_id = scenario_id
    integrator.forecast_horizon = 1
    integrator.push_to_arcgis = False
    integrator.samples_per_member = 1
    integrator.num_deep_ensemble = 1
    integrator.max_area = -np.inf
    integrator.max_inundation_map = None
    integrator.max_extent_timestamp = None
    integrator.bc_module = BC_Module(scenario_id)
    integrator.impact_module = ImpactModule(scenario_id)
    integrator.hazard_module = GroundTruthOracle(integrator.bc_module)

    captured = []
    integrator.write_results_simulated = (
        lambda ts, metrics, model=None, valid_time=None, road_metrics=None:
        captured.append({"issue": ts, "valid": valid_time, **metrics})
    )

    steps = sorted(integrator.bc_module.bc_dict)
    # Start where the event is actually wet, so the tier metrics have something
    # to disagree about; a dry timestep agrees trivially and proves nothing.
    start = steps[len(steps) // 2]
    end = steps[min(len(steps) - 1, len(steps) // 2 + n_timesteps - 1)]
    print(f"scenario {scenario_id}: {start} -> {end}\n")
    integrator.run_hourly_forecast(start_time=str(start), end_time=str(end),
                                   test_mode=True)

    d = pd.DataFrame(captured)
    cols = [c for c in ("valid", "n", "n_wet_gt", "depth_max_bias_wet",
                        "depth_max_rmse_wet", "ensemble_mean_qwk",
                        "ensemble_mean_hit_rate_Red") if c in d.columns]
    print(d[cols].to_string(index=False))

    print("\n-- identity checks --")
    ok = True
    if "ensemble_mean_qwk" in d:
        q = d.ensemble_mean_qwk
        ok &= bool(np.allclose(q.dropna(), 1.0))
        print(f"  qwk == 1.0 everywhere         : {np.allclose(q.dropna(), 1.0)}"
              f"   (min {q.min():.6f})")
    for c in ("depth_max_bias_wet", "depth_max_rmse_wet"):
        if c in d:
            ok &= bool(np.allclose(d[c].dropna(), 0.0, atol=1e-9))
            print(f"  {c:26s}: max |value| {d[c].abs().max():.3e}")
    print(f"\n  HARNESS {'SOUND' if ok else 'HAS A BUG -- see above'}")
    return d


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 57,
         int(sys.argv[2]) if len(sys.argv) > 2 else 6)
