import pandas as pd

from data.splitting import assert_no_leakage


def test_no_patient_leakage():
    metadata = pd.DataFrame(
        {
            "record": ["101", "106", "119", "200"],
            "split": ["train", "train", "val", "test"],
            "label": [0, 1, 0, 1],
        }
    )
    assert_no_leakage(metadata)

