# CV Experiment

MediaPipe hand detections are converted to a JSON skeleton representation in
`src/cv_experiment`. The generated file stores the fixed hand topology once and
the detected nodes separately for each frame.

From the `CV_Experiment` directory, run:

```powershell
uv run cv-experiment test_image.jpg --output output/hand_skeleton.json
```

The default model path is `hand_landmarker.task`. Use `--model` when the model
is stored somewhere else.

Convert a labelled directory while preserving its class folders:

```powershell
uv run cv-experiment data/raw/ASL/own_dataset `
  --output data/interim/landmarks_json
```

Use `--limit 10` for a small validation run. Existing JSON files are skipped;
pass `--overwrite` to regenerate them.

Each frame always contains stable `Left` and `Right` hand slots. A missing hand
has `present: false`, `uses_default_pose: true`, and an empty `nodes` list so a
renderer can apply its own neutral/default pose without treating it as a real
detection.

Create a balanced development subset and train the A-G baseline:

```powershell
uv run cv-experiment data/raw/BSL/train `
  --output data/interim/BSL/train `
  --limit-per-class 200

uv run cv-train data/interim/BSL/train
```

The validation split groups an image with its `flip` augmentation and groups
nearby numbered frames into blocks of 50. This reduces leakage from adjacent
frames of the same capture sequence.

Render a generated JSON file as a bound 3D skeleton:

```powershell
uv run cv-render-skeleton output/bsl_single.json `
  --output output/visualizations/bsl_single_3d.png
```

Detected hands use their MediaPipe nodes. A missing hand is rendered as a
transparent dashed neutral pose, proving that both renderer slots remain bound.
