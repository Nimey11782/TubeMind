import os

from dotenv import load_dotenv

load_dotenv()

_enabled = os.getenv(
    "ENABLE_LANGSMITH_TRACING",
    "false",
).strip().lower() in {"1", "true", "yes"}
_value = str(_enabled).lower()
os.environ["LANGCHAIN_TRACING_V2"] = _value
os.environ["LANGSMITH_TRACING"] = _value
TRACING_ENABLED = _enabled
