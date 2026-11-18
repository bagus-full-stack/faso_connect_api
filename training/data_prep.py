# training/data_prep.py — préparation des données (pandas seulement, sans torch)
"""
- split_corpus        : validation FLORES + jeu de test/validation pour une langue absente de FLORES
- temperature_sample  : rééquilibrage des langues par échantillonnage à température
- quantile_filter     : retire les paires les moins plausibles (score NLL) par direction
"""
import re
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PIVOTS = {"francais", "anglais"}


def norm(text: str) -> str:
    return re.sub(r"\W+", " ", str(text).lower()).strip()


# ── 1. Jeu de test pour une langue absente de FLORES ──────────────────

def split_corpus(
    df: pd.DataFrame,
    holdout_lang: str = "bambara",
    n_valid: int = 500,
    n_test: int = 1000,
    min_words: int = 3,
    seed: int = 42,
):
    """
    Retourne (df_train, df_valid, df_test).

    - FLORES (source == "flores_dev") reste la validation de base.
    - On retire du train `n_valid + n_test` paires holdout_lang<->francais, orientées
      holdout_lang -> francais (le `forced_bos_token_id` de l'éval est fra_Latn).
    - n_valid paires s'ajoutent à la validation (early stopping) ; n_test paires forment
      un test jamais vu ni pour l'entraînement ni pour le choix du modèle.
    - Toute ligne du train dont le texte holdout_lang est identique (après normalisation)
      à une phrase tenue à l'écart est retirée : pas de fuite.
    """
    flores = df["source"] == "flores_dev"
    rest = df[~flores]

    side_src = rest["src_lang"] == holdout_lang
    side_tgt = rest["tgt_lang"] == holdout_lang
    pair_fr = ((side_src & (rest["tgt_lang"] == "francais")) |
               (side_tgt & (rest["src_lang"] == "francais")))
    cand = rest[pair_fr].copy()

    # orientation holdout_lang -> francais
    swap = cand["tgt_lang"] == holdout_lang
    cand.loc[swap, ["src", "tgt"]] = cand.loc[swap, ["tgt", "src"]].to_numpy()
    cand["src_lang"], cand["tgt_lang"] = holdout_lang, "francais"

    cand["_key"] = cand["src"].map(norm)
    cand = cand[cand["src"].str.split().str.len() >= min_words]  # pas de mots isolés
    cand = cand.drop_duplicates("_key")

    n_total = min(n_valid + n_test, len(cand) // 2)  # garde au moins la moitié pour le train
    if n_total < n_valid + n_test:
        logger.warning(f"Seulement {len(cand)} candidats {holdout_lang} : holdout réduit à {n_total}")
    hold = cand.sample(n=n_total, random_state=seed)
    n_v = round(n_total * n_valid / (n_valid + n_test)) if (n_valid + n_test) else 0
    hold_valid, hold_test = hold.iloc[:n_v], hold.iloc[n_v:]

    keys = set(hold["_key"])
    involves = (rest["src_lang"] == holdout_lang) | (rest["tgt_lang"] == holdout_lang)
    holdout_text = np.where(rest["src_lang"] == holdout_lang, rest["src"], rest["tgt"])
    leaked = involves.to_numpy() & pd.Series(holdout_text, index=rest.index).map(norm).isin(keys).to_numpy()
    df_train = rest[~leaked]

    cols = [c for c in df.columns]
    df_valid = pd.concat([df[flores], hold_valid[cols]], ignore_index=True)
    df_test = hold_test[cols].reset_index(drop=True)
    logger.info(
        f"Split : train {len(df_train):,} | valid {len(df_valid):,} "
        f"(FLORES {int(flores.sum()):,} + {holdout_lang} {len(hold_valid):,}) | test {len(df_test):,}"
    )
    return df_train, df_valid, df_test


def both_directions(df: pd.DataFrame, cap: int = 250, seed: int = 42) -> pd.DataFrame:
    """
    Ajoute le sens inverse de chaque paire (fra->X en plus de X->fra), puis garde au plus `cap`
    lignes par direction : la validation (génération + beam search) reste rapide.
    """
    inv = df.rename(columns={"src": "tgt", "tgt": "src", "src_lang": "tgt_lang", "tgt_lang": "src_lang"})
    both = pd.concat([df, inv[df.columns]], ignore_index=True)
    parts = [g.sample(n=min(len(g), cap), random_state=seed) for _, g in both.groupby(["src_lang", "tgt_lang"])]
    return pd.concat(parts, ignore_index=True)


# ── 2. Échantillonnage à température ──────────────────────────────────

def temperature_sample(df: pd.DataFrame, T: float = 5.0, seed: int = 42) -> pd.DataFrame:
    """
    Rééquilibre les langues : proportion_l ∝ n_l^(1/T). T=1 garde les proportions brutes,
    T grand tend vers l'uniforme. La taille totale est conservée (même durée d'entraînement).
    Les langues rares sont suréchantillonnées avec remise, les dominantes sous-échantillonnées.
    La langue d'une ligne est son côté non-pivot (francais/anglais).
    """
    lang = np.where(df["tgt_lang"].isin(PIVOTS), df["src_lang"], df["tgt_lang"])
    df = df.assign(_lang=lang)
    counts = df["_lang"].value_counts()
    weights = counts.astype(float) ** (1.0 / T)
    target = (weights / weights.sum() * len(df)).round().astype(int)

    rng = np.random.RandomState(seed)
    parts = []
    for l, group in df.groupby("_lang"):
        n, k = len(group), int(target[l])
        if k <= n:
            parts.append(group.sample(n=k, random_state=rng))
        else:
            extra = group.sample(n=k - n, replace=True, random_state=rng)
            parts.append(pd.concat([group, extra]))
    out = pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)

    table = pd.DataFrame({"avant": counts, "après": out["_lang"].value_counts()}).fillna(0).astype(int)
    logger.info(f"Échantillonnage T={T} (part avant -> après) :\n"
                + table.assign(avant_pct=(100 * table["avant"] / table["avant"].sum()).round(1),
                               apres_pct=(100 * table["après"] / table["après"].sum()).round(1)).to_string())
    return out.drop(columns="_lang")


# ── 3. Filtre par score NLL ───────────────────────────────────────────

def quantile_filter(
    df: pd.DataFrame,
    score_col: str = "nll",
    drop_frac: float = 0.07,
    protect_sources=("flores_dev",),
) -> pd.DataFrame:
    """
    Retire, pour chaque direction (src_lang, tgt_lang), les `drop_frac` paires au NLL le plus élevé.
    Le seuil est calculé par direction : une langue plus mal modélisée ne perd pas tout son corpus.
    Les sources protégées (validation FLORES) ne sont jamais retirées.
    """
    thr = df.groupby(["src_lang", "tgt_lang"])[score_col].transform(lambda s: s.quantile(1 - drop_frac))
    keep = (df[score_col] <= thr) | df["source"].isin(protect_sources)
    return df[keep]
