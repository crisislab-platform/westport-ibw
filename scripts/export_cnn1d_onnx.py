"""Export the westport 1D-CNN benchmark so this repo can run it with onnxruntime.

The CNN's checkpoint (CNN1D_V1/model.pth) holds weights only, so the network is
rebuilt here from westport's own class and saved as ONNX, next to it:

    CNN1D_V1/model.onnx (+ model.onnx.data)  the network; > 2 GB, so its weights
                                             sit in the one external data file
    CNN1D_V1/scalers/bc_scalers.joblib       input scalers, fitted on the
                                             training events as westport fits them
    CNN1D_V1/domain_mask.npy                 the cells the CNN predicts, (H, W) bool

Everything comes from westport's CNN1DDataManager built with the paper run's
settings (supplementary/configs/cnn1d_baseline.json), so the inputs, scalers and
mask are the ones the published CNN numbers were produced with. It also writes
each test event's input matrix to --check_dir, for checking that
SavedCnnInundation's input preparation reproduces them.

Run once, in westport's environment and from the westport repo:

    cd ~/projects/westport
    /data/miniconda3/envs/westport-env/bin/python \\
        ~/projects/westport-ibw/scripts/export_cnn1d_onnx.py --check_dir <dir>
"""
import argparse
import os
import shutil
import sys

import joblib
import numpy as np
import onnx
import torch
from sklearn.preprocessing import StandardScaler

WESTPORT_REPO = os.path.expanduser("~/projects/westport")
sys.path.insert(0, WESTPORT_REPO)

from constants import MODEL_CHECKPOINT_DIR  # noqa: E402  (westport's)
from modules.datamanager.cnn1d.cnn1d_dm import CNN1DDataManager  # noqa: E402
from modules.models.cnn1d.cnn1d_model import CNNModel  # noqa: E402
from modules.models.cnn1d.cnn1d_model_wrapper import CNN1DRUNConfig  # noqa: E402

# The paper run's settings (supplementary/configs/cnn1d_baseline.json).
LAG = 7
DROPOUT = 0.2
TRAIN_EVENTS = [1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 29,
                40, 41, 42, 44, 45, 46, 47, 48, 49, 50, 51, 52, 55, 56, 59, 60, 61, 62, 65, 66, 67, 68, 69,
                70, 71, 72, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 87, 88, 89, 90, 92, 94, 95, 96, 97,
                98, 99, 100, 101, 102, 104, 105]
VALIDATION_EVENTS = [58, 86]
TEST_EVENTS = [3, 13, 17, 37, 43, 54, 57, 64, 83, 91, 93, 106]

CNN_DIR = os.path.join(MODEL_CHECKPOINT_DIR, "CNN1D_V1")


def fit_scalers(dm):
    """The three StandardScalers CNN1DDataManager.preprocess_boundary_conditions
    fits (it keeps only their output), refitted the same way on the same rows,
    and checked against its scaled inputs."""
    columns = dm.unscaled_inputs[TRAIN_EVENTS[0]].columns
    flow_columns = [c for c in columns if "River_flow" in c]
    sea_level_columns = [c for c in columns if "Sea_level" in c]
    flow = StandardScaler().fit(np.vstack([dm.unscaled_inputs[e][flow_columns].values for e in TRAIN_EVENTS]))
    sea = StandardScaler().fit(np.vstack([dm.unscaled_inputs[e][sea_level_columns].values for e in TRAIN_EVENTS]))
    time_step = StandardScaler().fit(np.vstack([dm.unscaled_inputs[e]["timestep"].values.reshape(-1, 1)
                                                for e in TRAIN_EVENTS]))
    for event_id in TEST_EVENTS:
        raw = dm.unscaled_inputs[event_id]
        rebuilt = np.hstack([flow.transform(raw[flow_columns].values),
                             sea.transform(raw[sea_level_columns].values),
                             time_step.transform(raw["timestep"].values.reshape(-1, 1))]).astype(np.float32)
        assert np.array_equal(rebuilt, dm.event_input_map[event_id]), f"scalers differ for event {event_id}"
    return {"flow_scaler": flow, "sea_level_scaler": sea, "time_step_scaler": time_step, "lag": LAG,
            "flow_columns": flow_columns, "sea_level_columns": sea_level_columns}


def export_onnx(dm):
    """model.pth -> model.onnx with every weight in one model.onnx.data file.
    The exporter scatters a > 2 GB model's weights over one file per tensor, so
    it writes to a scratch folder first and the result is re-saved as one file."""
    model = CNNModel(dm.input_channels, dm.num_features, dm.outputs, dropout=DROPOUT)
    model.load_state_dict(torch.load(os.path.join(CNN_DIR, "model.pth"), map_location="cpu")["model_state_dict"])
    model.eval()

    scratch = os.path.join(CNN_DIR, "_onnx_export")
    os.makedirs(scratch, exist_ok=True)
    try:
        torch.onnx.export(model, torch.zeros(1, dm.input_channels, dm.num_features),
                          os.path.join(scratch, "model.onnx"), input_names=["bc"], output_names=["depth"],
                          dynamic_axes={"bc": {0: "batch"}, "depth": {0: "batch"}}, opset_version=17)
        graph = onnx.load(os.path.join(scratch, "model.onnx"))
        onnx.save_model(graph, os.path.join(CNN_DIR, "model.onnx"), save_as_external_data=True,
                        all_tensors_to_one_file=True, location="model.onnx.data")
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check_dir", required=True, help="where each test event's input matrix is written")
    args = parser.parse_args()

    config = CNN1DRUNConfig(model_name="CNN1D_V1", lag=LAG, batch_size=8, learning_rate=0.01, epochs=10,
                            patience=10, save_model=False, dropout=DROPOUT, train_events=TRAIN_EVENTS,
                            validation_events=VALIDATION_EVENTS, test_events=TEST_EVENTS,
                            indices_per_timestep=10, mode="test")
    dm = CNN1DDataManager(config)

    os.makedirs(os.path.join(CNN_DIR, "scalers"), exist_ok=True)
    joblib.dump(fit_scalers(dm), os.path.join(CNN_DIR, "scalers", "bc_scalers.joblib"))
    np.save(os.path.join(CNN_DIR, "domain_mask.npy"), dm.get_domain_mask().numpy().astype(bool))

    os.makedirs(args.check_dir, exist_ok=True)
    for event_id in TEST_EVENTS:
        np.savez(os.path.join(args.check_dir, f"event_{event_id}_inputs.npz"),
                 inputs=dm.event_input_map[event_id],
                 flood_map_files=np.array([str(f) for f in dm.flood_map_file_map[event_id]]))

    export_onnx(dm)
    print("exported to", CNN_DIR)


if __name__ == "__main__":
    main()
