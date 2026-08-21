# CV Experiment

MediaPipe or optional official HaMeR hand estimates are converted to schema-v2 JSON. Each document stores
the fixed `mediapipe_hand_21` graph once, then binds every detected hand to it
with image, world and canonical landmark coordinates.

From the `CV_Experiment` directory, run:

```powershell
uv run cv-experiment data/raw/BSL/train/A/A0.jpg `
  --output output/schema_v2/A0.json
```

The default model path is `hand_landmarker.task`. Use `--model` when the model
is stored somewhere else.

Convert a labelled directory while preserving its class folders:

```powershell
uv run cv-experiment data/raw/BSL/train `
  --output data/interim/BSL/train_v2
```

Use `--limit 10` for a small validation run. Existing JSON files are skipped;
pass `--overwrite` to regenerate them.

Schema v2 separates the pipeline layers:

```text
MediaPipe result
  -> image_landmarks (normalized image coordinates)
  -> world_landmarks (meters)
  -> canonical_landmarks (wrist-relative, palm-normalized)
  -> validation (basic and bone integrity now; temporal/PnP reserved)
  -> renderer binding (mediapipe_hand_21)
```

Each frame always contains stable `Left` and `Right` slots. A missing hand has
`present: false`, `skeleton: null`, and `renderer.uses_default_pose: true`.
Missing coordinates are never fabricated as detector output.

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
uv run cv-render-skeleton output/schema_v2/A0.json `
  --output output/schema_v2/A0_3d.png
```

Detected hands use their MediaPipe nodes. A missing hand is rendered as a
transparent dashed neutral pose, proving that both renderer slots remain bound.

## Optional HaMeR 3D backend

Select `--backend hamer` to keep MediaPipe as the 2D detector while using the
official pretrained HaMeR model for 3D reconstruction. The output still uses
the same `mediapipe_hand_21` topology, JSON v2 normalization, validation,
renderer and training features. See [docs/HAMER_SETUP.md](docs/HAMER_SETUP.md)
for the isolated environment, checkpoint and licensed MANO asset requirements.
