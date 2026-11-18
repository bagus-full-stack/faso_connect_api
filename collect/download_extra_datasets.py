# collect/download_extra_datasets.py — sources supplémentaires (Hugging Face, accès sur demande)
"""
Ajoute trois sources au corpus, dans data/processed/corpus_burkina_extra.csv (lu par clean_corpus.py) :

  bia_moore    burkimbia/moore-parallel-corpus   fr->mooré  (hors religieux, hors mots isolés, hors SMOL)
  nllb_mined   burkimbia/nllb-burkina-parallel   fr/en->mooré, dioula, bambara, peul (filtré, plafonné)
  oza_bambara  oza75/bambara-mt (bm-fr-full)     fr->bambara, sous-ensembles humains seulement

Les trois datasets sont à accès manuel : demande l'accès sur la page de chaque dataset avec le compte
du HF_TOKEN. Sans accès, la source est ignorée avec un avertissement.

Licences : bia_moore = recherche / non commercial (MAFAND-MT CC BY-NC) ; nllb_mined = ODC-BY, avec
des phrases soumises au droit d'auteur de leurs éditeurs ; oza_bambara = MIT (sources Corbama /
Bamadaba à vérifier). Ne publie pas de modèle commercial sans vérifier chaque source.

Usage : python collect/download_extra_datasets.py
Réglages (variables d'environnement) : EXTRA_KEEP_RELIGIOUS=1, NLLB_MAX_PER_CONFIG=30000
"""
import os
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

OUT_FILE = Path("data/processed/corpus_burkina_extra.csv")
COLS = ["src", "tgt", "src_lang", "tgt_lang", "source"]
KEEP_RELIGIOUS = os.getenv("EXTRA_KEEP_RELIGIOUS", "0") == "1"
NLLB_MAX_PER_CONFIG = int(os.getenv("NLLB_MAX_PER_CONFIG", "30000"))
NLLB_MIN_RATIO = 0.5
NLLB_FULA = {"fuh_Latn", "ffm_Latn"}  # fulfulde du Burkina / Niger et Maasina (labels GlotLID)

CODE = {"fra": "francais", "fr": "francais", "eng": "anglais", "en": "anglais", "mos": "moore",
        "dyu": "dioula", "bam": "bambara", "bm": "bambara", "fuv": "fulfulde", "ff": "fulfulde"}


def _lang(code: str) -> str:
    return CODE.get(str(code).split("_")[0].lower(), "")


def _out(df: pd.DataFrame, src, tgt, src_lang, tgt_lang, source: str) -> pd.DataFrame:
    out = pd.DataFrame({"src": src, "tgt": tgt, "src_lang": src_lang, "tgt_lang": tgt_lang, "source": source})
    out["src"], out["tgt"] = out["src"].astype(str).str.strip(), out["tgt"].astype(str).str.strip()
    return out[(out["src"].str.len() > 5) & (out["tgt"].str.len() > 5)][COLS]


# ── Transformations (testables sans réseau) ───────────────────────────

def shape_bia_moore(df: pd.DataFrame) -> pd.DataFrame:
    n0 = len(df)
    df = df[df["segment_type"].isin(["sentence", "passage", "turn"])]  # ni mots isolés ni définitions
    if not KEEP_RELIGIOUS:
        df = df[df["domain"] != "religious"]  # le religieux domine déjà le corpus (moorefr)
    df = df[~df["source"].astype(str).str.lower().str.contains("smol")]  # déjà présent ; français = MT de l'anglais
    for side, want in (("source_lid_label", "fra"), ("target_lid_label", "mos")):
        lab = df[side].astype("string")
        df = df[lab.isna() | lab.str.startswith("und") | lab.str.startswith(want)]
    logger.info(f"  bia_moore : {n0:,} -> {len(df):,} (domaines : {df['domain'].value_counts().to_dict()})")
    return _out(df, df["src"], df["trg"], "francais", "moore", "bia_moore")


def shape_nllb(df: pd.DataFrame, config: str, max_rows: int = NLLB_MAX_PER_CONFIG) -> pd.DataFrame:
    n0 = len(df)
    urls = df.get("src_url", pd.Series("", index=df.index)).astype(str) + df.get("trg_url", pd.Series("", index=df.index)).astype(str)
    df = df[~urls.str.contains("jw.org", regex=False)]  # religieux, droits d'auteur, déjà couvert
    df = df[df["length_ratio"] >= NLLB_MIN_RATIO]
    if config.endswith("fuv"):
        df = df[df["trg_lid"].isin(NLLB_FULA)]
    # le score n'est comparable qu'au sein d'une config : on garde les meilleures paires
    df = df.sort_values("score", ascending=False).head(max_rows)
    s, t = config.split("-")
    logger.info(f"  nllb {config} : {n0:,} -> {len(df):,}")
    return _out(df, df["src_text"], df["trg_text"], _lang(s), _lang(t), "nllb_mined")


def shape_oza(df: pd.DataFrame) -> pd.DataFrame:
    n0 = len(df)
    sub = df["subset"].astype(str).str.lower() + "|" + df["dataset_source"].astype(str).str.lower()
    human = sub.str.contains("transcription|corbama") & ~sub.str.contains("synthetic|gec|dictionary")
    df = df[human]
    logger.info(f"  oza_bambara : {n0:,} -> {len(df):,} (sous-ensembles : {df['subset'].value_counts().to_dict()})")
    sl = df["source_lang"].map(_lang)
    fr_first = sl == "francais"
    src = df["source_text"].where(fr_first, df["target_text"])
    tgt = df["target_text"].where(fr_first, df["source_text"])
    ok = (sl == "francais") | (df["target_lang"].map(_lang) == "francais")
    return _out(df[ok], src[ok], tgt[ok], "francais", "bambara", "oza_bambara")


# ── Téléchargement (accès sur demande) ────────────────────────────────

def _hf(repo: str, filename: str):
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo, filename, repo_type="dataset")


def _guard(name: str, fn):
    try:
        return fn()
    except Exception as e:
        msg = str(e)
        hint = " : demande l'accès sur la page du dataset avec le compte du HF_TOKEN" if "403" in msg or "restricted" in msg.lower() else ""
        logger.warning(f"  ⚠️  {name} ignoré{hint} ({msg[:120]})")
        return pd.DataFrame(columns=COLS)


def download_bia_moore() -> pd.DataFrame:
    df = pd.read_parquet(_hf("burkimbia/moore-parallel-corpus", "data/train-00000-of-00001.parquet"))
    return shape_bia_moore(df)


def download_nllb() -> pd.DataFrame:
    parts = []
    for cfg in ("fra-mos", "fra-dyu", "fra-bam", "fra-fuv", "eng-mos", "eng-dyu", "eng-bam", "eng-fuv"):
        df = pd.read_parquet(_hf("burkimbia/nllb-burkina-parallel", f"data/{cfg}/train-00000-of-00001.parquet"))
        parts.append(shape_nllb(df, cfg))
    return pd.concat(parts, ignore_index=True)


def download_oza_bambara() -> pd.DataFrame:
    # train-00000-of-00002 et train-00001-of-00002 : environ 380 Mo au total
    parts = []
    for name in ("train-00000-of-00002.parquet", "train-00001-of-00002.parquet"):
        parts.append(shape_oza(pd.read_parquet(_hf("oza75/bambara-mt", f"bm-fr-full/{name}"))))
    return pd.concat(parts, ignore_index=True)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        from huggingface_hub import login
        login(token=token, add_to_git_credential=False)
    frames = [_guard(n, f) for n, f in (("bia_moore", download_bia_moore),
                                        ("nllb_mined", download_nllb),
                                        ("oza_bambara", download_oza_bambara))]
    df = pd.concat(frames, ignore_index=True).drop_duplicates(["src", "tgt"])
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_FILE, index=False, encoding="utf-8")
    logger.info(f"{len(df):,} paires -> {OUT_FILE}\n{df.groupby(['source', 'tgt_lang']).size().to_string()}")
    logger.info("Relance ensuite : python collect/clean_corpus.py")


if __name__ == "__main__":
    main()
