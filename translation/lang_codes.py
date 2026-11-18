# translation/lang_codes.py — source unique des codes de langue NLLB (API, entraînement, évaluation)
BURKINA_LANG_CODES = {
    "moore": "mos_Latn",
    "dioula": "dyu_Latn",
    "bambara": "bam_Latn",
    "fulfulde": "fuv_Latn",
    "gourmantsema": "gux_Latn",  # absent de NLLB-200 : jeton ajouté à l'entraînement (NEW_LANG_TOKENS)
    "dagaare": "dga_Latn",       # idem, pas encore entraîné
    "francais": "fra_Latn",
    "anglais": "eng_Latn",
}

# + codes ISO courts, pour les scripts de données
LANG_CODES = {
    **BURKINA_LANG_CODES,
    "mos": "mos_Latn", "dyu": "dyu_Latn", "bam": "bam_Latn", "fuv": "fuv_Latn",
    "gux": "gux_Latn", "dga": "dga_Latn", "fra": "fra_Latn", "eng": "eng_Latn",
    "ff": "fuv_Latn",
}
