import json
from pathlib import Path

import numpy as np
import pytest

from experiments.bootstrap_joint_coverage import load_manifest, run_dataset, collect
from experiments.bootstrap_band import features, refit_pairs
from experiments.bootstrap_band_sweep import dataset_streams

ROOT = Path(__file__).resolve().parents[2]


def manifests(tmp_path):
    p = json.loads((ROOT/"manifests/bootstrap_ols_joint_coverage.json").read_text())
    p.update(name="source", datasets=2, datasets_per_task=1,
             settings=[{"N":12,"B":9},{"N":20,"B":19}])
    path = tmp_path/"source.json"
    path.write_text(json.dumps(p))
    source = load_manifest(path)
    for index in range(2):
        run_dataset(source,index,tmp_path)
    p.pop("settings")
    p.update(name="cartesian", grid={"N":[12,20],"B":[5,9,19]},reuse_projects=["source"])
    path = tmp_path/"cartesian.json"
    path.write_text(json.dumps(p))
    return source,load_manifest(path)


def test_only_missing_suffix_is_refit_and_old_coverage_is_unchanged(tmp_path,monkeypatch):
    source,manifest=manifests(tmp_path)
    calls=[]
    def spy(design,y,indices):
        calls.append(indices.shape)
        return refit_pairs(design,y,indices)
    monkeypatch.setattr("experiments.bootstrap_cartesian_coverage.refit_pairs",spy)
    for index in range(2):
        saved=run_dataset(manifest,index,tmp_path)
        assert saved["reused_bootstrap_draws"]==28
        assert saved["new_bootstrap_draws"]==10
        assert run_dataset(manifest,index,tmp_path)==saved
        previous=json.loads((tmp_path/"source"/"datasets"/f"dataset-{index:04d}"/"summary.json").read_text())
        old={(r['N'],r['B']):r for r in previous['rows']}
        for row in saved['rows']:
            if (row['N'],row['B']) in old:
                assert row['covered']==old[row['N'],row['B']]['covered']
                assert row['critical_value']==old[row['N'],row['B']]['critical_value']
        spec={"seeds":manifest.payload['seeds'],"axes":{},"baseline":{"N":20,"B":19}}
        _,x,eps,u=dataset_streams(spec,index)
        with np.load(tmp_path/'cartesian'/'datasets'/f'dataset-{index:04d}'/'N12.npz') as data:
            ids=(12*u[:19,:12]).astype(np.int32)
            design=features(x[:12]); y=design@[0.,5.,-5.]+eps[:12]
            np.testing.assert_allclose(data['bootstrap_beta'],refit_pairs(design,y,ids),atol=1e-11)
        # Per-N checkpoints also prevent repeated work if only the final summary is missing.
        (tmp_path/'cartesian'/'datasets'/f'dataset-{index:04d}'/'summary.json').unlink()
        assert run_dataset(manifest,index,tmp_path)['rows']==saved['rows']
    assert calls==[(10,12),(10,12)]
    collect(manifest,tmp_path)
    summary=json.loads((tmp_path/'cartesian'/'summary.json').read_text())
    assert len(summary['metrics'])==6 and len(summary['plots'])==2
    assert all(Path(p['path']).read_bytes().startswith(b'%PDF-') for p in summary['plots'])
    assert {(r['N'],r['B']) for r in summary['metrics']}=={(n,b) for n in [12,20] for b in [5,9,19]}
    # Raw source corruption must be caught rather than hidden by the new cache.
    Path(previous['arrays']['path']).write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='Changed reused artifact'):
        run_dataset(manifest,1,tmp_path)


def test_full_saved_prefix_never_refits(tmp_path,monkeypatch):
    _,manifest=manifests(tmp_path)
    manifest.payload['grid']['N']=[20]
    manifest.payload['settings']=[s for s in manifest.payload['settings'] if s['N']==20]
    def forbidden(*args,**kwargs):
        raise AssertionError('Existing bootstrap draws must never be refitted')
    monkeypatch.setattr('experiments.bootstrap_cartesian_coverage.refit_pairs',forbidden)
    saved=run_dataset(manifest,0,tmp_path)
    assert saved['new_bootstrap_draws']==0


def test_cartesian_production_grid():
    m=load_manifest(ROOT/'manifests/bootstrap_ols_cartesian_coverage.json')
    assert len(m.payload['settings'])==81
    assert m.payload['datasets']==2000
    assert {(s['N'],s['B']) for s in m.payload['settings']}=={
        (n,b) for n in m.payload['grid']['N'] for b in m.payload['grid']['B']}
