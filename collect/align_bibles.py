# collect/align_bibles.py — Nouveau Testament gourmantchéma aligné au français, verset par verset
"""
Source : eBible.org (format VPL, un verset par ligne « LIV C:V texte »), traductions marquées
redistribuables. Chaque verset gourmantchéma est apparié à chaque version française libre.

  gux    Bible Gourma (gourmanchéma central, Burkina) — © 2020 Door43, NT seulement
  fuhbkf Fulfulde du Burkina (SIM International, © 2012) — NT, code fuh mais utilisé sous « fulfulde » (fuv_Latn)
  fra    frasbl (Sainte Bible libre pour le monde) — domaine public

Dagaare : aucune traduction libre sur eBible.org (seulement le « daga » de Papouasie, autre langue).
Le Nouveau Testament dagaare existe chez Bibles International / YouVersion, sous droits : à obtenir
auprès des éditeurs, pas de téléchargement automatique ici.

Sortie : data/processed/corpus_burkina_bible.csv (lu par clean_corpus.py)
Usage  : python collect/align_bibles.py
"""
import io
import re
import logging
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

OUT_FILE = Path("data/processed/corpus_burkina_bible.csv")
BASE_URL = "https://ebible.org/Scriptures/{id}_vpl.zip"
TARGETS = {"gourmantsema": ["gux"], "fulfulde": ["fuhbkf"]}  # langue -> ids eBible (guxg = dialecte du Togo, à ajouter à la main)
FRENCH = ["frasbl"]  # un seul français : sinon chaque verset cible compte deux fois
LINE = re.compile(r"^([A-Z0-9]{3}) (\d+:\d+)(?:[-,]\d+)*\s+(.+)$")


def parse_vpl(text: str) -> dict:
    """{'MAT 1:1': 'texte'} ; ignore les lignes vides ou mal formées."""
    verses = {}
    for line in text.splitlines():
        m = LINE.match(line.strip())
        if m and m.group(3).strip():
            verses[f"{m.group(1)} {m.group(2)}"] = m.group(3).strip()
    return verses


def fetch_vpl(bible_id: str) -> dict:
    # eBible refuse l'agent par défaut de urllib (403)
    req = urllib.request.Request(BASE_URL.format(id=bible_id), headers={"User-Agent": "Mozilla/5.0 (corpus-builder)"})
    with urllib.request.urlopen(req, timeout=120) as r:
        z = zipfile.ZipFile(io.BytesIO(r.read()))
    name = next(n for n in z.namelist() if n.endswith("_vpl.txt"))
    return parse_vpl(z.read(name).decode("utf-8"))


def align(fr: dict, tgt: dict, tgt_lang: str, source: str) -> pd.DataFrame:
    keys = [k for k in tgt if k in fr]  # seuls les versets présents des deux côtés
    return pd.DataFrame({
        "src": [fr[k] for k in keys], "tgt": [tgt[k] for k in keys],
        "src_lang": "francais", "tgt_lang": tgt_lang, "source": source,
    })


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    french = {i: fetch_vpl(i) for i in FRENCH}
    parts = []
    for lang, ids in TARGETS.items():
        for bid in ids:
            tgt = fetch_vpl(bid)
            for fid, fr in french.items():
                df = align(fr, tgt, lang, f"ebible_{bid}")
                logger.info(f"{bid} x {fid} : {len(tgt):,} versets cibles -> {len(df):,} paires")
                parts.append(df)
    df = pd.concat(parts, ignore_index=True).drop_duplicates(["src", "tgt"])
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_FILE, index=False, encoding="utf-8")
    logger.info(f"{len(df):,} paires -> {OUT_FILE}")


if __name__ == "__main__":
    main()
