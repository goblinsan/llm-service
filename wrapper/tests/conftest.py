"""Set environment variables before the wrapper module is imported by any test."""
import os
import sys
import tempfile
from pathlib import Path

# Ensure the wrapper package root is on sys.path so that `import main` works
# regardless of where pytest is invoked from.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# These must be set before wrapper.main is first imported because it reads
# them at module level.
_MODELS_TMP = tempfile.mkdtemp(prefix="llm_test_models_")

os.environ.setdefault("MODELS_DIR", _MODELS_TMP)
os.environ.setdefault("MODEL_PATH", os.path.join(_MODELS_TMP, "default.gguf"))
os.environ.setdefault("ADMIN_TOKEN", "test-admin-token")
os.environ.setdefault("SKIP_LLAMA_STARTUP", "1")
