# training/score_pairs.py — filtre les paires par perte de décodage forcé (NLL moyenne par token)
"""
Score chaque paire (src -> tgt) avec le modèle de base NLLB (colonne nll). Le filtre lui-même
(retrait par direction des DROP_FRAC paires les moins plausibles) est appliqué par finetune_nllb.py
au train seulement, APRÈS le split test/validation : le test reste non filtré.

Usage (Kaggle ou local) :
    python -m training.score_pairs
    DROP_FRAC=0.07 NLLB_BASE_MODEL=facebook/nllb-200-distilled-1.3B python -m training.score_pairs

Sortie :
    data/processed/corpus_burkina_scored.csv  (toutes les lignes + colonne nll) -> CORPUS_FILE
"""
import os
import logging

import numpy as np
import pandas as pd
import torch

from training.finetune_nllb import BASE_MODEL, LANG_CODES, DEVICE, hf_login

logger = logging.getLogger(__name__)

IN_FILE = os.getenv("CORPUS_FILE", "data/processed/corpus_burkina_clean.csv")
SCORED = "data/processed/corpus_burkina_scored.csv"
DROP_FRAC = float(os.getenv("DROP_FRAC", "0.07"))
BATCH = int(os.getenv("SCORE_BATCH", "32"))
MAXLEN = 128


@torch.no_grad()
def score_group(tok, model, src, tgt, src_lang, tgt_lang):
    """NLL moyenne par token de tgt sachant src, pour une direction."""
    tok.src_lang, tok.tgt_lang = LANG_CODES[src_lang], LANG_CODES[tgt_lang]
    order = np.argsort([len(s) for s in src])  # batches de longueur homogène
    out = np.zeros(len(src), dtype=np.float32)
    for i in range(0, len(src), BATCH):
        idx = order[i:i + BATCH]
        enc = tok([src[j] for j in idx], text_target=[tgt[j] for j in idx],
                  max_length=MAXLEN, truncation=True, padding=True, return_tensors="pt").to(DEVICE)
        labels = enc["labels"].masked_fill(enc["labels"] == tok.pad_token_id, -100)
        logits = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], labels=labels).logits
        nll = torch.nn.functional.cross_entropy(
            logits.float().transpose(1, 2), labels, ignore_index=-100, reduction="none")
        n_tok = (labels != -100).sum(1).clamp(min=1)
        out[idx] = (nll.sum(1) / n_tok).cpu().numpy()
    return out


def main():
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    hf_login()
    df = pd.read_csv(IN_FILE, encoding="utf-8").dropna(subset=["src", "tgt"]).reset_index(drop=True)
    df["src"], df["tgt"] = df["src"].astype(str), df["tgt"].astype(str)

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32).to(DEVICE).eval()

    df["nll"] = np.nan
    for (sl, tl), g in df.groupby(["src_lang", "tgt_lang"]):
        if sl not in LANG_CODES or tl not in LANG_CODES:
            logger.warning(f"Direction ignorée (langue inconnue) : {sl}->{tl}")
            continue
        unk = tok.unk_token_id
        if unk in (tok.convert_tokens_to_ids(LANG_CODES[sl]), tok.convert_tokens_to_ids(LANG_CODES[tl])):
            logger.warning(f"Direction ignorée (jeton absent du modèle de base, notes vides) : {sl}->{tl}")
            continue
        logger.info(f"Score {sl}->{tl} : {len(g):,} paires")
        df.loc[g.index, "nll"] = score_group(tok, model, g["src"].tolist(), g["tgt"].tolist(), sl, tl)

    df.to_csv(SCORED, index=False, encoding="utf-8")
    logger.info(f"{len(df):,} lignes scorées : {SCORED}")
    logger.info("Lance l'entraînement avec CORPUS_FILE=%s : le split test/validation se fait AVANT le filtre, "
                "qui ne touche que le train (DROP_FRAC=%s).", SCORED, DROP_FRAC)
    logger.info("Relis un échantillon trié par nll décroissant avant de fixer DROP_FRAC.")


if __name__ == "__main__":
    main()
