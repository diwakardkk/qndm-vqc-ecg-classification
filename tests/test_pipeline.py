from pathlib import Path

from analysis.reporting import output_tree
from utils.config import load_config


def test_output_directories(tmp_path):
    output_tree(tmp_path, [4, 6, 8])
    assert (tmp_path / "4_qubits" / "gradient_validation").is_dir()
    assert (tmp_path / "cross_qubit_comparison").is_dir()


def test_reproducibility():
    cfg = load_config("configs/quick.yaml")
    assert cfg["training"]["seeds"] == [42]
    assert cfg["data"]["dataset_dir"] == "MIT-BIH Arrhythmia"

