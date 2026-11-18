# Tests de la logique anti-fuite (training/data_prep.py) et de l'alignement biblique
import numpy as np
import pandas as pd

from collect.align_bibles import parse_vpl
from training.data_prep import quantile_filter, split_corpus


def _corpus(n=4000):
    rows = []
    for i in range(n):
        rows.append((f"phrase source numéro {i} bien assez longue", f"tgt bambara numéro {i} assez long",
                     "francais", "bambara", "hf_djelia"))
    for i in range(n):
        rows.append((f"phrase mooré {i} assez longue ici", f"phrase française {i} assez longue ici",
                     "moore", "francais", "smol_sent"))
    return pd.DataFrame(rows, columns=["src", "tgt", "src_lang", "tgt_lang", "source"])


def test_test_et_validation_absents_du_train():
    train, valid, test = split_corpus(_corpus(), holdout_lang="bambara")
    assert len(test) > 0 and len(valid) > 0
    held = set(test["src"]) | set(test["tgt"]) | set(valid["src"]) | set(valid["tgt"])
    assert not (set(train["src"]) | set(train["tgt"])) & held


def test_quantile_filter_retire_les_pires_et_garde_les_sans_note():
    df = pd.DataFrame({"src_lang": "a", "tgt_lang": "b", "source": "x", "nll": list(np.arange(100.0)) + [np.nan]})
    out = quantile_filter(df, drop_frac=0.1)
    assert out["nll"].max() < 95 and out["nll"].isna().sum() == 1


def test_parse_vpl():
    v = parse_vpl("MAT 1:1 un\nMAT 1:2 deux\n\nligne cassée\nMAT 1:3   \n")
    assert v == {"MAT 1:1": "un", "MAT 1:2": "deux"}
