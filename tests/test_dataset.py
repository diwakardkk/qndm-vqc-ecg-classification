from pathlib import Path

from data.loader import validate_dataset
from data.splitting import label_policy


def test_dataset_files_detected():
    root = Path("MIT-BIH Arrhythmia")
    frame = validate_dataset(root, ["100", "101"])
    assert frame[["dat", "hea", "atr"]].all().all()


def test_AAMI_mapping():
    policy = label_policy(strict=False)
    for symbol in ["N", "L", "R", "e", "j"]:
        assert policy.classify(symbol) == (0, -1)
    for symbol in ["V", "E"]:
        assert policy.classify(symbol) == (1, 1)
    assert policy.classify("A") is None

