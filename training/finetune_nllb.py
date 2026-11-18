# training/finetune_nllb.py — VERSION FINALE
"""
Fine-tuning de NLLB-200 distilled 600M sur les langues burkinabè.

Corrections finales :
  - Charge corpus_burkina_clean.csv (nettoyé)
  - Imports autonomes (sans dépendance à training.config)
  - Filtre automatique des données monolingues (bloom_mono)
  - dataloader_num_workers=0 (Windows compatible)
  - Login HF automatique au démarrage

Usage :
    python training/finetune_nllb.py

Prérequis :
    data/processed/corpus_burkina_clean.csv  <- généré par clean_corpus.py
"""

import os
import sys
import logging
import time
from pathlib import Path
import unicodedata

import torch
import pandas as pd

# ── Logging ───────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── Configuration (autonome) ──────────────────────────────────────────

BASE_MODEL  = os.getenv("NLLB_BASE_MODEL", "facebook/nllb-200-distilled-600M")
OUTPUT_DIR  = "models/nllb-burkina-v1"
LOG_DIR     = "logs/tensorboard"
# avec filtre qualité : CORPUS_FILE=data/processed/corpus_burkina_scored.csv (sortie de score_pairs.py)
CORPUS_FILE = os.getenv("CORPUS_FILE", "data/processed/corpus_burkina_clean.csv")
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

# Codes de langue à ajouter au tokenizer (absents de NLLB-200) ; "" pour désactiver
NEW_LANG_TOKENS = [t for t in os.getenv("NEW_LANG_TOKENS", "gux_Latn").split(",") if t]

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # lancé par `python training/finetune_nllb.py`
from translation.lang_codes import LANG_CODES  # noqa: E402

TRAIN_EPOCHS            = 5
BATCH_SIZE              = 4     # 8
GRADIENT_ACCUMULATION   = 8  # 2
LEARNING_RATE           = 3e-4  # Au lieu de 5e-5
WARMUP_STEPS            = 500
MAX_SOURCE_LENGTH       = 256
MAX_TARGET_LENGTH       = 256
EARLY_STOP_PATIENCE     = 10 # Au lieu de 3
SAVE_TOTAL_LIMIT        = 2

os.environ["TENSORBOARD_LOGGING_DIR"] = LOG_DIR
# ── Auth HuggingFace ──────────────────────────────────────────────────

def hf_login():
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        try:
            from huggingface_hub import login
            login(token=token, add_to_git_credential=False)
            logger.info("HuggingFace authentifié")
        except Exception as e:
            logger.warning(f"Login HF : {e}")
    else:
        logger.warning("HF_TOKEN manquant dans .env")

# ── Chargement modèle ─────────────────────────────────────────────────

def load_model():
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    from peft import get_peft_model, LoraConfig, TaskType

    logger.info(f"Chargement {BASE_MODEL} sur {DEVICE}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    # 1. Charger en float16 pour économiser la RAM/VRAM
    model = AutoModelForSeq2SeqLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    ).to(DEVICE)

    # 1b. Langues absentes de NLLB-200 : un jeton par code (ex. gux_Latn), dont seule la ligne
    # d'embedding est entraînée (trainable_token_indices) ; l'adaptateur sauvegardé la contient.
    missing = [t for t in NEW_LANG_TOKENS
               if tokenizer.convert_tokens_to_ids(t) in (tokenizer.unk_token_id, None)]
    new_ids = []
    if missing:
        tokenizer.add_tokens(missing, special_tokens=True)
        if len(tokenizer) > model.get_input_embeddings().num_embeddings:
            model.resize_token_embeddings(len(tokenizer))
        new_ids = [tokenizer.convert_tokens_to_ids(t) for t in missing]
        logger.info(f"Jetons de langue ajoutés : {dict(zip(missing, new_ids))}")

    # 2. Configuration LoRA
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "fc1", "fc2"],  # Cible l'attention
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
        trainable_token_indices={"embed_tokens": new_ids} if new_ids else None,
    )

    # 3. Application de LoRA au modèle
    model = get_peft_model(model, lora_config)

    # Affichera : "trainable params: ~4,000,000 || all params: 619,000,000 || trainable%: 0.6%"
    model.print_trainable_parameters()

    logger.info(f"Modèle préparé avec LoRA sur {DEVICE}")
    return tokenizer, model

# ── Chargement corpus ─────────────────────────────────────────────────

def remove_accents_and_special_chars(text: str) -> str:
    if not isinstance(text, str): return ""
    text_no_accents = "".join(
        c for c in unicodedata.normalize("NFD", text)
        if unicodedata.category(c) != "Mn"
    )
    replacements = {
        'ɛ': 'e', 'Ɛ': 'E', 'ɔ': 'o', 'Ɔ': 'O', 'ɓ': 'b', 'Ɓ': 'B',
        'ɗ': 'd', 'Ɗ': 'D', 'ŋ': 'n', 'Ŋ': 'N', 'ɲ': 'ny', 'Ɲ': 'Ny',
        'ʋ': 'v', 'Ʋ': 'V', 'ƴ': 'y', 'Ƴ': 'Y'
    }
    for special, standard in replacements.items():
        text_no_accents = text_no_accents.replace(special, standard)
    return text_no_accents


def load_corpus():
    if not Path(CORPUS_FILE).exists():
        logger.error(
            f"Corpus nettoyé introuvable : {CORPUS_FILE}\n"
            "Lance d'abord : python collect/clean_corpus.py"
        )
        sys.exit(1)

    df = pd.read_csv(CORPUS_FILE, encoding="utf-8")
    logger.info(f"Corpus brut chargé : {len(df):,} lignes")

    # 🛑 SÉCURITÉ ANTI-DATA LEAKAGE 🛑
    if "source" not in df.columns:
        logger.error(
            "ERREUR CRITIQUE : La colonne 'source' a disparu du corpus ! "
            "Ton script clean_corpus.py a dû la supprimer. "
            "Elle est indispensable pour isoler FLORES de l'entraînement."
        )
        sys.exit(1)

    # Nettoyage défensif
    df = df.dropna(subset=["src", "tgt"])
    df["src"] = df["src"].astype(str).str.strip()
    df["tgt"] = df["tgt"].astype(str).str.strip()
    df = df[df["src"].str.len() > 5]
    df = df[df["tgt"].str.len() > 5]

    # Validation = FLORES + 500 phrases bambara (absent de FLORES) ; test bambara écarté du train
    try:
        from training.data_prep import (split_corpus, split_real_test, temperature_sample,
                                        both_directions, quantile_filter)
    except ImportError:  # lancé par `python training/finetune_nllb.py`
        from data_prep import (split_corpus, split_real_test, temperature_sample,
                               both_directions, quantile_filter)
    # split sur le corpus NON filtré ; le filtre de qualité (colonne nll de score_pairs.py) ne vise que le train
    df_train, df_valid, df_test = split_corpus(df, holdout_lang="bambara")
    # test « usage réel » par langue (dioula, peul, mooré), retiré du train avant le filtre
    df_train, df_real = split_real_test(df_train, n=int(os.getenv("REAL_TEST_N", "150")))
    df_real.drop(columns=["nll"], errors="ignore").assign(verifie="").to_csv(
        "data/processed/test_reel.csv", index=False, encoding="utf-8")
    if "nll" in df_train.columns:
        n0 = len(df_train)
        df_train = quantile_filter(df_train, drop_frac=float(os.getenv("DROP_FRAC", "0.07")))
        logger.info(f"Filtre NLL (train seulement) : {n0:,} -> {len(df_train):,}")
        df_train, df_valid, df_test = (d.drop(columns="nll") for d in (df_train, df_valid, df_test))
    # validation dans les deux sens, 250 phrases max par direction
    df_valid = both_directions(df_valid, cap=int(os.getenv("VALID_PER_DIRECTION", "250")))
    logger.info(f"Validation : {len(df_valid):,} lignes\n"
                + df_valid.groupby(["src_lang", "tgt_lang"]).size().to_string())
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    df_test.to_csv("data/processed/test_bambara.csv", index=False, encoding="utf-8")

    # 1. SWAPPING (Bidirectionnel)
    logger.info("Inversion des paires (Apprentissage bidirectionnel)...")
    df_inverse = df_train.copy()
    df_inverse = df_inverse.rename(columns={"src": "tgt", "tgt": "src", "src_lang": "tgt_lang", "tgt_lang": "src_lang"})
    df_train = pd.concat([df_train, df_inverse], ignore_index=True)

    # 2. BRUIT (Robustesse - 15% des données)
    logger.info("Injection de bruit (tolérance aux fautes de frappe)...")
    df_noise = df_train.sample(frac=0.15, random_state=42).copy()
    df_noise["src"] = df_noise["src"].apply(remove_accents_and_special_chars)
    original_src = df_train.loc[df_noise.index, "src"]
    df_noise = df_noise[df_noise["src"] != original_src]
    df_train = pd.concat([df_train, df_noise], ignore_index=True)

    # 3. Échantillonnage à température (TEMPERATURE=1 -> proportions brutes)
    df_train = temperature_sample(df_train, T=float(os.getenv("SAMPLING_TEMPERATURE", "5")))

    return df_train, df_valid

# ── Tokenisation ──────────────────────────────────────────────────────

def drop_unsupported_langs(df, tokenizer, name):
    """Retire les lignes dont une langue n'a pas de jeton dans le tokenizer (ex. gux_Latn, dga_Latn :
    absents de NLLB-200). Sinon le code de langue devient <unk> et pollue l'entraînement."""
    unk = tokenizer.unk_token_id
    bad = {l for l in set(df["src_lang"]) | set(df["tgt_lang"])
           if tokenizer.convert_tokens_to_ids(LANG_CODES.get(l, "")) in (unk, None)}
    if bad:
        n0 = len(df)
        df = df[~df["src_lang"].isin(bad) & ~df["tgt_lang"].isin(bad)]
        logger.warning(f"[{name}] langues sans jeton NLLB, lignes retirées : {sorted(bad)} ({n0 - len(df):,} lignes). "
                       "Ajoute le jeton au tokenizer pour les entraîner.")
    return df


def make_tokenize_fn(tokenizer):
    def tokenize(examples):
        input_ids = []
        attention_masks = []
        labels = []

        # On s'assure d'avoir les listes de langues (fallback si la colonne n'existe pas)
        src_langs = examples.get("src_lang", ["moore"] * len(examples["src"]))
        tgt_langs = examples.get("tgt_lang", ["francais"] * len(examples["tgt"]))

        for i in range(len(examples["src"])):
            # 1. On configure le tokenizer avec les VRAIES langues de CETTE phrase
            tokenizer.src_lang = LANG_CODES.get(src_langs[i], "mos_Latn")
            tokenizer.tgt_lang = LANG_CODES.get(tgt_langs[i], "fra_Latn")

            # 2. Tokenisation de la phrase source
            inp = tokenizer(
                examples["src"][i],
                max_length=MAX_SOURCE_LENGTH,
                truncation=True
            )

            # 3. Tokenisation de la traduction (qui inclura désormais le bon token de langue cible)
            lab = tokenizer(
                text_target=examples["tgt"][i],
                max_length=MAX_TARGET_LENGTH,
                truncation=True
            )

            input_ids.append(inp["input_ids"])
            attention_masks.append(inp["attention_mask"])
            labels.append(lab["input_ids"])

        return {
            "input_ids": input_ids,
            "attention_mask": attention_masks,
            "labels": labels
        }

    return tokenize

def build_datasets(tokenizer, df_train, df_valid):
    from datasets import Dataset, DatasetDict

    logger.info("Tokenisation du corpus...")
    t0 = time.time()

    fn = make_tokenize_fn(tokenizer)

    train_ds = Dataset.from_pandas(df_train.reset_index(drop=True))
    valid_ds = Dataset.from_pandas(df_valid.reset_index(drop=True))

    tokenized = DatasetDict({
        "train": train_ds.map(
            fn, batched=True,
            remove_columns=train_ds.column_names,
            desc="Train",
        ),
        "validation": valid_ds.map(
            fn, batched=True,
            remove_columns=valid_ds.column_names,
            desc="Validation",
        ),
    })

    logger.info(f"Tokenisation terminée en {round(time.time()-t0)}s")
    return tokenized

# ── Métriques ─────────────────────────────────────────────────────────

def make_compute_metrics(tokenizer):
    import evaluate
    import numpy as np
    bleu = evaluate.load("sacrebleu")
    chrf = evaluate.load("chrf")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred.predictions, eval_pred.label_ids
        inputs = eval_pred.inputs  # requiert include_for_metrics=["inputs"]

        # Parfois, les prédictions sont retournées sous forme de tuple, on prend le premier élément
        if isinstance(predictions, tuple):
            predictions = predictions[0]

        # 1. Remplacer les -100 par le pad_token_id dans les PRÉDICTIONS
        predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

        # 2. Remplacer les -100 par le pad_token_id dans les LABELS
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Format attendu par sacrebleu : liste de listes pour les références
        decoded_labels = [[l] for l in decoded_labels]

        result = bleu.compute(predictions=decoded_preds, references=decoded_labels)

        metrics = {
            "bleu": round(result["score"], 2),
            "bleu_1": round(result["precisions"][0], 2),
            "bleu_2": round(result["precisions"][1], 2),
        }

        # chrF++ par direction : le 1er token de l'entrée = langue source, celui du label = langue cible
        src_codes = tokenizer.convert_ids_to_tokens(np.asarray(inputs)[:, 0].astype(int).tolist())
        tgt_codes = tokenizer.convert_ids_to_tokens(np.asarray(labels)[:, 0].astype(int).tolist())
        groups = {}
        for i, key in enumerate(zip(src_codes, tgt_codes)):
            groups.setdefault(key, []).append(i)
        scores = []
        for (s, t), idx in sorted(groups.items()):
            score = chrf.compute(
                predictions=[decoded_preds[i] for i in idx],
                references=[decoded_labels[i] for i in idx],
                word_order=2,
            )["score"]
            metrics[f"chrf_{s[:3]}_{t[:3]}"] = round(score, 2)
            scores.append(score)
        # chrf_macro = moyenne des directions ; chrf_min = la direction la plus faible (détecte un recul isolé)
        metrics["chrf_macro"] = round(sum(scores) / len(scores), 2)
        metrics["chrf_min"] = round(min(scores), 2)
        return metrics

    return compute_metrics

# ── Entraînement ──────────────────────────────────────────────────────

def make_lang_trainer_class():
    """Trainer dont la génération de validation démarre sur la langue cible de chaque exemple."""
    from transformers import Seq2SeqTrainer

    class LangPrefixTrainer(Seq2SeqTrainer):
        def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None, **gen_kwargs):
            labels = inputs.get("labels")
            if labels is not None and not prediction_loss_only:
                # labels[:, 0] = code de langue cible (le tokenizer le place en tête) ; le décodeur
                # démarre sur </s> puis ce code, ce qui remplace un forced_bos_token_id unique.
                start = torch.full_like(labels[:, :1], model.config.decoder_start_token_id)
                gen_kwargs["decoder_input_ids"] = torch.cat([start, labels[:, :1]], dim=1)
            return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys, **gen_kwargs)

    return LangPrefixTrainer


def train(tokenizer, model, tokenized_datasets):
    from transformers import (
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        DataCollatorForSeq2Seq,
        EarlyStoppingCallback,
    )

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    Path(LOG_DIR).mkdir(parents=True, exist_ok=True)

    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,

        # Epochs et batch
        num_train_epochs=TRAIN_EPOCHS,
        max_steps=int(os.getenv("MAX_STEPS", "-1")),  # essai rapide : MAX_STEPS=60 EVAL_STEPS=30
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,  # 2
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},

        # Optimiseur
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        weight_decay=0.05,  # Au lieu de 0.01
        lr_scheduler_type="cosine",

        # Précision
        fp16=DEVICE == "cuda",
        bf16=False,

        # Évaluation et sauvegarde
        eval_strategy="steps",
        eval_steps=int(os.getenv("EVAL_STEPS", "250")),    # Évalue le modèle tous les 1000 batchs
        save_strategy="steps",
        save_steps=int(os.getenv("EVAL_STEPS", "250")),    # Sauvegarde un checkpoint en même temps
        load_best_model_at_end=True,
        # chrf_macro (défaut) ou chrf_min (protège la langue la plus faible) via BEST_METRIC
        metric_for_best_model=os.getenv("BEST_METRIC", "chrf_macro"),
        greater_is_better=True,
        include_for_metrics=["inputs"],

        # Génération
        predict_with_generate=True,
        generation_max_length=MAX_TARGET_LENGTH,

        # Logs
        # logging_dir=LOG_DIR,
        logging_steps=50,
        report_to="tensorboard",

        # Divers
        save_total_limit=SAVE_TOTAL_LIMIT,
        push_to_hub=False,
        # ✅ 0 sur Windows pour éviter les erreurs multiprocessing
        dataloader_num_workers=0,
    )


    trainer = make_lang_trainer_class()(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        processing_class=tokenizer,  # au lieu de tokenizer=tokenizer
        data_collator=DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=-100,
            pad_to_multiple_of=8 if DEVICE == "cuda" else None,
        ),
        compute_metrics=make_compute_metrics(tokenizer),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOP_PATIENCE)],
    )

    logger.info("Démarrage du fine-tuning...")
    logger.info(f"  Modèle  : {BASE_MODEL}")
    logger.info(f"  Device  : {DEVICE}")
    logger.info(f"  Train   : {len(tokenized_datasets['train']):,}")
    logger.info(f"  Valid   : {len(tokenized_datasets['validation']):,}")
    logger.info(f"  Epochs  : {TRAIN_EPOCHS}")
    logger.info(f"  Batch   : {BATCH_SIZE}")
    logger.info(f"  Output  : {OUTPUT_DIR}")
    logger.info(f"  TBoard  : tensorboard --logdir {LOG_DIR}")

    t0 = time.time()
    trainer.train()
    duration = round((time.time() - t0) / 60, 1)
    logger.info(f"Fine-tuning terminé en {duration} minutes")

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    logger.info(f"Modèle sauvegardé : {OUTPUT_DIR}")

    return trainer

# ── Main ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    hf_login()
    torch.cuda.empty_cache()

    tokenizer, model   = load_model()
    df_train, df_valid = load_corpus()
    df_train = drop_unsupported_langs(df_train, tokenizer, "train")
    df_valid = drop_unsupported_langs(df_valid, tokenizer, "valid")
    tokenized          = build_datasets(tokenizer, df_train, df_valid)
    trainer            = train(tokenizer, model, tokenized)

    logger.info(
        f"\n{'='*50}\n"
        f"Fine-tuning termine !\n"
        f"Modele : {OUTPUT_DIR}\n\n"
        f"Integrer dans FasoConnect :\n"
        f"  translation/nllb_engine.py :\n"
        f"  MODEL_ID = './{OUTPUT_DIR}'\n"
        f"  Puis : docker compose up --build -d\n"
        f"{'='*50}"
    )