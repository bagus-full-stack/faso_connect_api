# translation/nllb_engine.py — VERSION PRODUCTION (LORA FUSIONNÉ)
import os
import threading
import torch
import logging
from pathlib import Path
from typing import Optional
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel

from translation.lang_codes import BURKINA_LANG_CODES  # noqa: F401 (réexporté pour l'API)


class NLLBTranslator:
    """
    Traducteur multilingue Meta NLLB-200 distilled 600M fine-tuné.
    Chargé une seule fois au démarrage via lifespan FastAPI.
    """

    # Doit être le MÊME modèle que celui utilisé pour entraîner l'adaptateur LoRA
    BASE_MODEL_ID = os.getenv("NLLB_BASE_MODEL", "facebook/nllb-200-distilled-600M")
    ADAPTER_MODEL_ID = os.getenv(
        "NLLB_ADAPTER_PATH",
        str(Path(__file__).resolve().parent.parent / "models" / "nllb-burkina-v1"),
    )

    def __init__(self, device: Optional[str] = None):
        # tokenizer.src_lang est un état partagé : on sérialise les inférences
        self._lock = threading.Lock()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Chargement de NLLB de base sur {self.device}...")

        # 1. Chargement du tokenizer depuis l'adaptateur local
        self.tokenizer = AutoTokenizer.from_pretrained(self.ADAPTER_MODEL_ID)

        # 2. Chargement du modèle de base
        base_model = AutoModelForSeq2SeqLM.from_pretrained(
            self.BASE_MODEL_ID,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            low_cpu_mem_usage=True,
        )

        # 2b. Jetons de langue ajoutés à l'entraînement (ex. gux_Latn) : ils sont dans le tokenizer de l'adaptateur
        if len(self.tokenizer) > base_model.get_input_embeddings().num_embeddings:
            base_model.resize_token_embeddings(len(self.tokenizer))

        # 3. Application de l'adaptateur LoRA
        logging.info(f"Application des poids LoRA depuis {self.ADAPTER_MODEL_ID}...")
        peft_model = PeftModel.from_pretrained(base_model, self.ADAPTER_MODEL_ID)

        # 4. Fusion des poids pour maximiser la vitesse d'inférence en production
        self.model = peft_model.merge_and_unload().to(self.device)
        self.model.eval()

        # Identifie l'adaptateur : entre dans la clé de cache, pour ne pas servir d'anciennes traductions
        adapter_file = Path(self.ADAPTER_MODEL_ID) / "adapter_model.safetensors"
        self.version = str(int(adapter_file.stat().st_mtime)) if adapter_file.exists() else "base"

        logging.info(f"✅ NLLB-200 Fine-tuné fusionné et prêt pour la production sur {self.device}")

    def _resolve_lang_id(self, lang_code: str) -> int:
        """Résout l'ID du token langue — compatible toutes versions transformers."""
        token_id = self.tokenizer.convert_tokens_to_ids(lang_code)
        if token_id == self.tokenizer.unk_token_id:
            raise ValueError(
                f"Code langue non reconnu : '{lang_code}'. "
                f"Disponibles : {list(BURKINA_LANG_CODES.values())}"
            )
        return token_id

    def translate(
            self,
            text: str,
            src_lang: str,
            tgt_lang: str,
            max_length: int = 512,
            num_beams: int = 5,
    ) -> str:
        with self._lock:
            src_code = BURKINA_LANG_CODES.get(src_lang, src_lang)
            tgt_code = BURKINA_LANG_CODES.get(tgt_lang, tgt_lang)

            self.tokenizer.src_lang = src_code
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(self.device)

            tgt_lang_id = self._resolve_lang_id(tgt_code)

            with torch.no_grad():
                output_tokens = self.model.generate(
                    **inputs,
                    forced_bos_token_id=tgt_lang_id,
                    max_length=max_length,
                    num_beams=num_beams,
                    early_stopping=True,
                )

            return self.tokenizer.decode(output_tokens[0], skip_special_tokens=True)

