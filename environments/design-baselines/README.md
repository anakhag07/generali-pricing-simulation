# Pinned Design-Bench environment

Use the upstream Design-Baselines environment at commit
`785dbcfa58107bfcc426257a1c2e69d7f71c3c27`. The integration does not copy its
requirements: the checked-out upstream repository remains the source of truth.

The upstream environment is large because `design-bench` and Design-Baselines
declare broad shared dependencies for all tasks and methods. This repository
does not add to that dependency set and invokes only
`design_baselines.gradient_ascent.gradient_ascent`. Ordinary project tests never
import this environment.

Two upstream assumptions no longer hold, so the plain
`conda env create -f environment.yml` from the upstream README cannot succeed
as-is:

- `mujoco-py==2.0.2.3` needs build-time inputs it does not declare.
- Design-Bench downloads its datasets and oracles from Google Cloud URLs that
  now return HTTP 404 (upstream issue #11).

The rest of this file is the working recipe, verified on ORCD Engaging
(RHEL 9, no root). Paths below are the ones actually used; adjust as needed.

## 1. MuJoCo 2.0 and its license

MuJoCo 2.0 became free in October 2021 and Roboti publishes an unlocked
activation key, so no proprietary license is required:

```bash
mkdir -p ~/.mujoco
curl -sSL -o ~/.mujoco/mjkey.txt https://www.roboti.us/file/mjkey.txt
```

The key is issued to "Everyone" and expires October 2031. `mujoco-py 2.0.2.3`
also needs the MuJoCo 2.0 binaries at `~/.mujoco/mujoco200`
(from `https://www.roboti.us/download/mujoco200_linux.zip`).

## 2. Helper prefix for build-time system libraries

`mujoco-py`'s CPU extension builder compiles an OSMesa shim and links
`OSMesa`, `GL`, and `glewosmesa`, then rewrites the result with `patchelf`.
`glewosmesa` ships inside `mujoco200/bin`, but OSMesa and `patchelf` do not
exist on this cluster and `libGL.so` (the linker symlink) is missing even
though `libGL.so.1` is present.

Do **not** add these to the pinned environment; keep them in a separate prefix
so the pinned dependency set stays untouched:

```bash
conda create -y -p ~/.conda/envs/osmesa-provider -c conda-forge --override-channels \
  "mesalib=21.2.5" patchelf
ln -sfn /lib64/libGL.so.1 ~/.conda/envs/osmesa-provider/lib/libGL.so
```

`mesalib=21.2.5` is pinned deliberately: modern `mesalib` (26.x) no longer
ships `libOSMesa`/`GL/osmesa.h`.

An EGL build path exists that would avoid OSMesa entirely, but `mujoco_py`
selects it only when `nvidia-smi` exists *and* a legacy `/usr/lib/nvidia-<NNN>`
or `/usr/local/nvidia/lib64` directory is present. That is an old Ubuntu
convention absent on RHEL 9, and there is no environment variable to force it,
so the OSMesa route is the supported one here.

## 3. Environment variables (needed at build *and* run time)

`mujoco_py` resolves `libmujoco200` and OSMesa when it is imported, not only
when it is compiled, so any Slurm job or test runner needs these exported too:

```bash
export OSM=~/.conda/envs/osmesa-provider
export PATH=$OSM/bin:$PATH
export CPATH=$OSM/include
export LIBRARY_PATH=$OSM/lib:$HOME/.mujoco/mujoco200/bin
export LD_LIBRARY_PATH=$OSM/lib:$HOME/.mujoco/mujoco200/bin:$LD_LIBRARY_PATH
export MUJOCO_PY_FORCE_CPU=1
```

`MUJOCO_PY_FORCE_CPU=1` is correct here: the exact oracles need physics, never
rendering.

## 4. Create the environment

```bash
git clone https://github.com/brandontrabucco/design-baselines.git
cd design-baselines
git checkout 785dbcfa58107bfcc426257a1c2e69d7f71c3c27
conda env create -f environment.yml   # conda stage succeeds; pip stage fails on mujoco-py
```

The pip stage fails on `mujoco-py`, which leaves the conda packages installed.
Finish it with the variables from step 3 exported, adding
`--no-build-isolation` so `setup.py` can see the already-installed Cython
(`mujoco-py` imports Cython at build time without declaring it):

```bash
PY=~/.conda/envs/design-baselines/bin/python
$PY -m pip install --no-build-isolation "mujoco-py==2.0.2.3"
$PY -m pip install --no-build-isolation -r requirements.txt
```

Verify physics actually steps, not just that the import resolves:

```bash
$PY -c "
import mujoco_py
m = mujoco_py.load_model_from_xml('<mujoco><worldbody><body pos=\"0 0 1\">'
    '<joint type=\"free\"/><geom type=\"sphere\" size=\".1\"/></body></worldbody></mujoco>')
s = mujoco_py.MjSim(m)
for _ in range(10): s.step()
print('z =', float(s.data.qpos[2]))  # < 1.0 if gravity is being integrated
"
```

## 5. Datasets and oracles

Design-Bench resolves its assets under
`<site-packages>/design_bench_data`, and `DiskResource.is_downloaded` is just
`os.path.exists(...)`. Placing the files there therefore skips the dead
download with no patching of upstream.

The official bucket returns 404, so these came from the community mirror
`https://huggingface.co/datasets/beckhamc/design_bench_data`, whose directory
layout already matches the expected paths:

```bash
DD=~/.conda/envs/design-baselines/lib/python3.7/site-packages/design_bench_data
HF=https://huggingface.co/datasets/beckhamc/design_bench_data/resolve/main
mkdir -p $DD/ant_morphology
curl -sSL -o $DD/smiles_vocab.txt $HF/smiles_vocab.txt
for f in ant_morphology-x-0.npy ant_morphology-y-0.npy ant_oracle.pkl; do
  curl -sSL -o $DD/ant_morphology/$f $HF/ant_morphology/$f
done
```

`smiles_vocab.txt` is required even for an Ant-only run: `design_bench/__init__.py`
eagerly builds the ChEMBL feature extractor for every registered task at import
time, so `import design_bench` fails without it.

D'Kitty needs `dkitty_morphology/{dkitty_morphology-x-0.npy,dkitty_morphology-y-0.npy,dkitty_oracle.pkl}`
from the same mirror.

**Provenance caveat.** This mirror is community-provided, not the official
Google bucket, which no longer serves these files. The Ant arrays have the
expected shape/dtype and are finite, and the exact oracle reproduces recorded
dataset values closely (see below), but they cannot be proven byte-identical to
the assets used in the Design-Bench paper. Note this if published numbers ever
depend on them. `scripts/design_bench.py` pins whatever is actually used via its
checksummed manifest ID.

## 6. Verification

```bash
$PY -c "
import design_bench, numpy as np
t = design_bench.make('AntMorphology-Exact-v0', relabel=False)
print(np.asarray(t.x).shape, np.asarray(t.y).shape)   # (10004, 60) (10004, 1)
print(t.predict(np.asarray(t.x)[:1]), np.asarray(t.y)[0])
"
```

`task.x` has 10004 rows even though `ant_morphology-x-0.npy` holds 25009:
Design-Bench subsamples to the official Ant task size. The raw file size is not
the task size.

On the verified run the exact oracle scored the first dataset design at
`-209.26` against its recorded `-210.13`, which is the signal that the oracle,
the policy pickle, and MuJoCo are wired together correctly.

Run the opt-in end-to-end checks with the step-3 variables exported:

```bash
DESIGN_BENCH_PYTHON=~/.conda/envs/design-baselines/bin/python \
  pytest -q -m design_bench_live tests/benchmarks/test_design_bench_live.py
```

A single exact-oracle `predict` call takes roughly a minute on a CPU node.
