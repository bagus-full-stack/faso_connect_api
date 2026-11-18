# Doit s'exécuter avant tout import de api.main (config lue à l'import).
import os

os.environ["APP_ENV"] = "test"
os.environ["API_KEY"] = "test-key"
os.environ["ALLOWED_ORIGIN"] = "http://localhost:3000"
