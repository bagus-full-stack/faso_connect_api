# tts/mms_engine.py — VERSION FINALE
import torch
import numpy as np
import io
import base64
import logging
import os
import subprocess
import tempfile
import threading
import scipy.io.wavfile as wavfile
from dataclasses import dataclass
from transformers import VitsModel, AutoTokenizer

MMS_TTS_MODELS = {
    "moore":        "facebook/mms-tts-mos",
    "dioula":       "facebook/mms-tts-dyu",
    "bambara":      "facebook/mms-tts-bam",
    "fulfulde":     "facebook/mms-tts-ful",
    "gourmantsema": "facebook/mms-tts-gux",
    "dagaare":      "facebook/mms-tts-dga",
    "francais":     "facebook/mms-tts-fra",
    "anglais":      "facebook/mms-tts-eng",
}


@dataclass
class TTSResult:
    audio_b64:        str
    sample_rate:      int
    duration_seconds: float


def _time_stretch(pcm: np.ndarray, sample_rate: int, speed: float) -> np.ndarray:
    """Change la vitesse sans changer la hauteur (ffmpeg atempo, plage 0.5-2.0)."""
    try:
        with tempfile.TemporaryDirectory() as tmp:
            src, dst = os.path.join(tmp, "in.wav"), os.path.join(tmp, "out.wav")
            wavfile.write(src, sample_rate, pcm)
            subprocess.run(
                ["ffmpeg", "-v", "error", "-y", "-i", src, "-filter:a", f"atempo={speed}", dst],
                check=True, capture_output=True, timeout=30,
            )
            return wavfile.read(dst)[1]
    except (FileNotFoundError, subprocess.SubprocessError) as e:
        # shortcut: sans ffmpeg, rééchantillonnage simple (la hauteur change)
        logging.warning(f"ffmpeg indisponible ({e}) — repli sur resample")
        import scipy.signal as signal
        return signal.resample(pcm, int(len(pcm) / speed)).astype(np.int16)


class MMSTTSEngine:
    """
    Moteur TTS Meta MMS.
    Modèles chargés à la demande et mis en cache mémoire.
    L'audio n'est jamais stocké sur disque — base64 uniquement.
    """

    def __init__(self):
        self._cache: dict = {}
        self._load_lock = threading.Lock()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"MMS-TTS initialisé sur {self.device}")

    def _load_model(self, lang: str):
        with self._load_lock:
            return self._load_model_locked(lang)

    def _load_model_locked(self, lang: str):
        if lang not in self._cache:
            model_id = MMS_TTS_MODELS.get(lang)
            if not model_id:
                raise ValueError(
                    f"Langue TTS non supportée : '{lang}'. "
                    f"Disponibles : {list(MMS_TTS_MODELS.keys())}"
                )
            logging.info(f"Chargement MMS-TTS '{lang}'...")
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = VitsModel.from_pretrained(model_id).to(self.device)
            model.eval()
            self._cache[lang] = (tokenizer, model)
            logging.info(f"✅ MMS-TTS '{lang}' prêt")
        return self._cache[lang]

    def synthesize(self, text: str, lang: str, speed: float = 1.0) -> TTSResult:
        tokenizer, model = self._load_model(lang)

        inputs = tokenizer(text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            output = model(**inputs)

        waveform    = output.waveform.squeeze().cpu().numpy()
        sample_rate = model.config.sampling_rate

        pcm = (np.clip(waveform, -1.0, 1.0) * 32767).astype(np.int16)
        if speed != 1.0:
            pcm = _time_stretch(pcm, sample_rate, speed)

        duration = round(len(pcm) / sample_rate, 2)

        buffer = io.BytesIO()
        wavfile.write(buffer, sample_rate, pcm)
        audio_b64 = base64.b64encode(buffer.getvalue()).decode()

        return TTSResult(
            audio_b64=audio_b64,
            sample_rate=sample_rate,
            duration_seconds=duration,
        )

    def preload(self, langs: list[str]):
        """Préchauffage des modèles TTS au démarrage."""
        for lang in langs:
            try:
                self._load_model(lang)
            except ValueError as e:
                logging.warning(f"Préchargement ignoré : {e}")