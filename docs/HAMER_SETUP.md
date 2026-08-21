# Official HaMeR backend setup on Windows

The project uses Python 3.10 for both backends. HaMeR remains an external,
editable dependency; its source and multi-gigabyte assets are not copied into
this repository.

```text
image -> MediaPipe 2D/handedness/box -> official HaMeR 3D
      -> mediapipe_hand_21 -> shared JSON v2/validation/renderer/training
```

## Verified environment

- Python: 3.10.11
- Final project environment: `D:/Code/COMP5615/CV_Experiment/.venv`
- Archived former Python 3.14 environment: `.venv314_archive`
- PyTorch: 2.11.0+cu130
- torchvision: 0.26.0+cu130
- GPU: NVIDIA GeForce RTX 5070
- MediaPipe: 1.0.1
- NumPy: 1.26.4
- HaMeR repository: `D:/Code/third_party/hamer`
- HaMeR commit: `3a01849f4148352e9260b69bf28b65d1671a4905`

The official README shows CUDA 11.7 as an example. This machine uses official
CUDA 13.0 PyTorch wheels because its RTX 5070 is a Blackwell GPU and the
installed NVIDIA driver supports CUDA 13.x.

## Recreate the project environment

```powershell
& "C:/Users/ADMIN/AppData/Local/Programs/Python/Python310/python.exe" -m venv .venv
& ./.venv/Scripts/python.exe -m pip install --upgrade pip setuptools wheel
& ./.venv/Scripts/python.exe -m pip install -e .
& ./.venv/Scripts/python.exe -m pip install `
  torch==2.11.0 torchvision==0.26.0 `
  --index-url https://download.pytorch.org/whl/cu130
```

## Install official HaMeR for this hybrid pipeline

```powershell
git clone --recursive https://github.com/geopavlakos/hamer.git D:/Code/third_party/hamer

& ./.venv/Scripts/python.exe -m pip install `
  gdown pyrender pytorch-lightning scikit-image smplx==0.1.28 `
  yacs timm einops pandas webdataset
& ./.venv/Scripts/python.exe -m pip install --no-build-isolation chumpy==0.70

Push-Location D:/Code/third_party/hamer
& D:/Code/COMP5615/CV_Experiment/.venv/Scripts/python.exe -m pip install -e . --no-deps
$env:PYOPENGL_PLATFORM = "win32"
& D:/Code/COMP5615/CV_Experiment/.venv/Scripts/python.exe -c `
  "from hamer.models import download_models; download_models()"
Pop-Location
```

The official `[all]` extra also installs detectron2, mmcv and ViTPose. They are
not installed here because this project's hybrid adapter receives hand boxes
from MediaPipe and does not import the official body/keypoint detectors. This
avoids unused Windows C++ builds while preserving official HaMeR model code and
weights. Do not use this minimal environment for the unmodified official demo
or for HaMeR training.

## Licensed MANO asset

Download the right-hand MANO model only from the official MANO website after
accepting its licence. Place it exactly here:

```text
D:/Code/third_party/hamer/_DATA/data/mano/MANO_RIGHT.pkl
```

The project must not download, redistribute, fabricate or substitute this
file. HaMeR is not operational until it is present and real inference succeeds.

## Verify the environment

```powershell
& ./.venv/Scripts/python.exe -m cv_experiment.env_check
& ./.venv/Scripts/python.exe -m unittest discover -s tests -v
```

## Run and render HaMeR

```powershell
& ./.venv/Scripts/python.exe -m cv_experiment test_image.jpg `
  --backend hamer `
  --model hand_landmarker.task `
  --hamer-root D:/Code/third_party/hamer `
  --hamer-checkpoint D:/Code/third_party/hamer/_DATA/hamer_ckpts/checkpoints/hamer.ckpt `
  --device cuda `
  --output output/backend_comparison/hamer.json

& ./.venv/Scripts/python.exe -m cv_experiment.render_skeleton `
  output/backend_comparison/hamer.json `
  --output output/backend_comparison/hamer_3d.png
```

HaMeR model-space values are labelled `model_unit` and
`equivalent_to_mediapipe_world: false`. Both backends use the same
wrist-relative, wrist-to-middle-MCP canonical normalization downstream.
