# Environment Notes — UV-Net Local Adaptation

Date: 2026-07-28
Machine: Ultra 7 270K + RTX 5070 12GB (Windows), conda env `uvnet`

## 1. Base environment

Cloned from the verified `brepnet` env:

| Package | Version |
|---|---|
| python | 3.10 |
| pytorch | 2.11.0+cu128 |
| pytorch-lightning | 1.9.5 |
| torchmetrics | 1.9.0 |
| occwl | 3.0.0 (git 8b536ea) |
| pythonocc-core (OCC) | 7.7.2 |
| numpy | 1.23.5 |
| libigl | 2.4.1 |

Additional installs for UV-Net: `trimesh`.
(joblib / matplotlib / scikit-learn / tqdm were already present.)

## 2. Differences from the official environment.yml

| Official pin | This env |
|---|---|
| python 3.9 | 3.10 |
| pytorch 1.8 | 2.11.0+cu128 |
| pytorch-lightning 1.3 | 1.9.5 |
| torchmetrics 0.3.2 | 1.9.0 |
| occwl 1.0 (channel `lambouj`) | 3.0.0 (git 8b536ea) |
| dgl-cuda11.0 0.6.1 | dgl 2.2.1+cu121 |
| pythonocc-core 7.4.x (implied) | 7.7.2 |

## 3. DGL 2.2.1+cu121 (critical)

- Install command:
  `pip install dgl -f https://data.dgl.ai/wheels/cu121/repo.html`
- Wheel used: `dgl-2.2.1+cu121-cp310-cp310-win_amd64.whl` (82.3 MB).
- WARNING: the PyPI build of dgl 2.2.1 is CPU-only. Do NOT use plain
  `pip install dgl`.
- DGL officially stopped providing Windows/macOS packages on 2024-06-27;
  2.2.1 is the last available version. Newer versions require building
  from source.

## 4. graphbolt stub (site-packages level; must be re-applied after
   reinstalling or upgrading dgl)

Replace `site-packages/dgl/graphbolt/__init__.py` entirely with:

```python
# graphbolt disabled: UV-Net uses classic DGLGraph only.
def __getattr__(name):
    return None
```

Reason: the graphbolt C++ binaries shipped with dgl 2.2.1 only support
torch 2.1-2.3 (`graphbolt_pytorch_2.x.x.dll`); loading under torch 2.11
fails (FileNotFoundError / WinError 127). UV-Net uses the classic
DGLGraph API only and never touches graphbolt. The original file can be
restored by reinstalling dgl.

## 5. Verification log (2026-07-28, RTX 5070 / sm_120)

- `import dgl` -&gt; 2.2.1+cu121 OK
- GraphConv forward on CPU OK
- GraphConv forward on CUDA OK

## 6. Code-level adaptations (no changes to model logic)

1. Replace deprecated np.int with np.int64 (numpy 1.23.5 compatibility)
   (commit bdf3d2ddf36c0e14af15a3a67af1666827e6c08e)
2. Adapt torchmetrics 0.3 API to 1.x: Accuracy(task/num_classes),
   IoU -> JaccardIndex, drop compute_on_step
   (commit 035f592883c28ad20b4f456ed17bbef3e13123d6)