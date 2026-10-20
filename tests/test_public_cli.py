"""Public entry points preserve results and support computed-only figures."""
import importlib.util
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
SCRIPTS=ROOT/'sptc_toolbox/scripts'


def command(script,*args):
    return subprocess.run([sys.executable,str(SCRIPTS/script),*map(str,args)],capture_output=True,text=True,cwd=ROOT)


def test_plot_without_publisher_image_and_preserve_existing_result(tmp_path):
    axis=np.linspace(0,.04,5)
    x,y=np.meshgrid(axis,axis,indexing='ij')
    data=tmp_path/'coupling.npz';np.savez(data,axis=axis,Vxy=np.stack([x,y],axis=-1).astype(complex))
    out=tmp_path/'figure'
    first=command('plot_jin2025_tensor.py','--dataset',data,'--output',out)
    assert first.returncode==0,first.stderr
    assert (out/'computed.svg').is_file()
    assert not (out/'paper_vs_legume.png').exists()
    original=(out/'plot_metrics.json').read_bytes()
    second=command('plot_jin2025_tensor.py','--dataset',data,'--output',out)
    assert second.returncode!=0
    assert (out/'plot_metrics.json').read_bytes()==original


def test_four_samples_rejected_for_missing_off_diagonal_reference():
    spec=importlib.util.spec_from_file_location('public_tensor_plot',SCRIPTS/'plot_jin2025_tensor.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    with pytest.raises(ValueError,match='at least five'):
        module.analyze(np.linspace(0,1,4),np.ones((4,4,2),complex),100)


def test_validator_checks_requested_root_before_calculation(tmp_path):
    out=tmp_path/'new-checks'
    result=command('validate_jin2025_tensor.py','--root',tmp_path/'missing','--output',out)
    assert result.returncode!=0 and 'Missing required regenerated inputs' in result.stderr
    assert not out.exists()


def test_regeneration_preserves_existing_scan(tmp_path):
    dataset=tmp_path/'n3.47_g4.1_N9';dataset.mkdir()
    marker=dataset/'coupling.npz';marker.write_bytes(b'previous calculation')
    result=command('reproduce_jin2025_tensor.py','--gmax','4.1','--grid-size','9','--output',tmp_path)
    assert result.returncode!=0 and 'already exists' in result.stderr
    assert marker.read_bytes()==b'previous calculation'
