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
    holdout_source: str = "hf_djelia",
    n_valid: int = 500,
    n_test: int = 1000,
    min_words: int = 3,
    block: int = 25,
    seed: int = 42,
):
    """
    Retourne (df_train, df_valid, df_test).

    - FLORES (source == "flores_dev") reste la validation de base.
    - Pour `holdout_lang` (absente de FLORES), on retire du train `n_valid` + `n_test` paires
      holdout_lang -> francais, prises par BLOCS de `block` lignes consécutives de `holdout_source`
      (source distincte de la source dominante du train). Les lignes voisines d'un même texte restent
      ainsi du même côté de la coupure. Les blocs entiers sont retirés du train.
      shortcut: on suppose que l'ordre des lignes du CSV suit l'ordre des documents ; à remplacer par
      un vrai identifiant de document si le corpus en reçoit un.
    - Les blocs du test et ceux de la validation sont disjoints.
    - Toute ligne du train dont le texte holdout_lang est identique (après normalisation)
      à une phrase tenue à l'écart est retirée : pas de fuite.
    - Le split se fait sur le corpus NON filtré : appliquer le filtre de qualité au train seulement.
    """
    flores = df["source"] == "flores_dev"
    rest = df[~flores]

    involves = (rest["src_lang"] == holdout_lang) | (rest["tgt_lang"] == holdout_lang)
    pair_fr = involves & ((rest["src_lang"] == "francais") | (rest["tgt_lang"] == "francais"))
    pool = rest[pair_fr]
    if holdout_source not in set(pool["source"]):
        logger.warning(f"Source {holdout_source!r} absente : holdout pris dans toutes les sources")
        holdout_source = None
    elif (pool["source"] == holdout_source).sum() < 2 * (n_valid + n_test):
        logger.warning(f"Source {holdout_source!r} trop petite : holdout pris dans toutes les sources")
        holdout_source = None
    pool = pool if holdout_source is None else pool[pool["source"] == holdout_source]

    # blocs de lignes consécutives (ordre d'origine du CSV)
    pool = pool.sort_index()
    blk = pd.Series(np.arange(len(pool)) // block, index=pool.index)

    cand = pool.copy()
    swap = cand["tgt_lang"] == holdout_lang          # orientation holdout_lang -> francais
    cand.loc[swap, ["src", "tgt"]] = cand.loc[swap, ["tgt", "src"]].to_numpy()
    cand["src_lang"], cand["tgt_lang"] = holdout_lang, "francais"
    cand["_key"] = cand["src"].map(norm)
    cand["_blk"] = blk
    cand = cand[cand["src"].str.split().str.len() >= min_words]  # pas de mots isolés
    cand = cand.drop_duplicates("_key")

    rng = np.random.RandomState(seed)
    blocks = rng.permutation(cand["_blk"].unique())
    sizes = cand.groupby("_blk").size()
    test_blk, valid_blk, n_t, n_v = [], [], 0, 0
    for b in blocks:
        if n_t < n_test:
            test_blk.append(b); n_t += sizes[b]
        elif n_v < n_valid:
            valid_blk.append(b); n_v += sizes[b]
        else:
            break
    if n_t < n_test or n_v < n_valid:
        logger.warning(f"Holdout {holdout_lang} incomplet : test {n_t}/{n_test}, valid {n_v}/{n_valid}")
    hold_test = cand[cand["_blk"].isin(test_blk)].sample(frac=1, random_state=seed).head(n_test)
    hold_valid = cand[cand["_blk"].isin(valid_blk)].sample(frac=1, random_state=seed).head(n_valid)

    # retire du train les blocs entiers + toute phrase identique ailleurs
    held_blk = test_blk + valid_blk
    block_rows = blk.index[blk.isin(held_blk)]
    keys = set(cand.loc[cand["_blk"].isin(held_blk), "_key"])
    holdout_text = np.where(rest["src_lang"] == holdout_lang, rest["src"], rest["tgt"])
    leaked = involves.to_numpy() & pd.Series(holdout_text, index=rest.index).map(norm).isin(keys).to_numpy()
    df_train = rest[~leaked & ~rest.index.isin(block_rows)]

    cols = list(df.columns)
    df_valid = pd.concat([df[flores], hold_valid[cols]], ignore_index=True)
    df_test = hold_test[cols].reset_index(drop=True)
    logger.info(
        f"Split : train {len(df_train):,} | valid {len(df_valid):,} "
        f"(FLORES {int(flores.sum()):,} + {holdout_lang} {len(hold_valid):,}) | test {len(df_test):,} "
        f"[source={holdout_source or 'toutes'}, blocs de {block}]"
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
    # lignes sans score (langue inconnue du scoring) : conservées
    keep = (df[score_col] <= thr) | df[score_col].isna() | df["source"].isin(protect_sources)
    return df[keep]


# ── 4. Test « usage réel » par langue ─────────────────────────────────

REAL_TEST_SOURCES = {"dioula": "hf_uvci", "fulfulde": "hf_arprim", "moore": "smol_sent"}


def split_real_test(
    df_train: pd.DataFrame,
    specs: dict = None,
    n: int = 150,
    min_words: int = 3,
    block: int = 25,
    seed: int = 42,
):
    """
    Retire du train un petit test « usage réel » par langue, pris dans une source de phrases courantes
    (specs = {langue: source}). Même méthode que split_corpus : blocs de `block` lignes consécutives de
    la source, blocs entiers retirés du train, et toute phrase identique ailleurs retirée aussi.
    La paire garde son orientation d'origine (francais->dioula, anglais->moore...).
    Retourne (df_train, df_real). À faire relire par un locuteur avant de s'y fier.
    """
    specs = REAL_TEST_SOURCES if specs is None else specs
    rng = np.random.RandomState(seed)
    drop_idx, keys, parts = set(), {}, []
    nonpivot = pd.Series(np.where(df_train["tgt_lang"].isin(PIVOTS), df_train["src_lang"], df_train["tgt_lang"]),
                         index=df_train.index)
    langtext = pd.Series(np.where(df_train["tgt_lang"].isin(PIVOTS), df_train["src"], df_train["tgt"]),
                         index=df_train.index)

    for lang, source in specs.items():
        pool = df_train[(df_train["source"] == source) & (nonpivot == lang)].sort_index()
        if pool.empty:
            logger.warning(f"Test réel {lang}: aucune ligne dans {source!r}, ignoré")
            continue
        blk = pd.Series(np.arange(len(pool)) // block, index=pool.index)
        cand = pool.assign(_blk=blk, _key=langtext.loc[pool.index].map(norm))
        cand = cand[langtext.loc[cand.index].str.split().str.len() >= min_words].drop_duplicates("_key")
        sizes = cand.groupby("_blk").size()
        chosen, got = [], 0
        for b in rng.permutation(sizes.index.to_numpy()):
            if got >= n:
                break
            chosen.append(b); got += sizes[b]
        if got < n:
            logger.warning(f"Test réel {lang}: {got}/{n} phrases seulement dans {source!r}")
        sel = cand[cand["_blk"].isin(chosen)].sample(frac=1, random_state=seed).head(n)
        parts.append(sel.drop(columns=["_blk", "_key"]))
        drop_idx |= set(blk.index[blk.isin(chosen)])
        keys[lang] = set(cand.loc[cand["_blk"].isin(chosen), "_key"])

    leaked = pd.Series(False, index=df_train.index)
    for lang, ks in keys.items():
        leaked |= (nonpivot == lang) & langtext.map(norm).isin(ks)
    out = df_train[~leaked & ~df_train.index.isin(drop_idx)]
    real = pd.concat(parts, ignore_index=True) if parts else df_train.iloc[0:0]
    logger.info(f"Test réel : {len(real)} phrases ({real.groupby('src_lang').size().to_dict() if len(real) else {}}) ; "
                f"train {len(df_train):,} -> {len(out):,}")
    return out, real
