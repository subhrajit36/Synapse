"""Suite-wide setup. Imported by pytest before any test module imports synapse."""

import os

# Tests must not read the developer's `.env`: real NEO4J_* or GEMINI_API_KEY
# values would turn "unconfigured" tests into live network calls.
os.environ["SYNAPSE_NO_DOTENV"] = "1"
