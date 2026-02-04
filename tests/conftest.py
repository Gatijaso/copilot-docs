import os
import sys
from pathlib import Path

# Put project root (parent of `src`) on sys.path so `import src.*` works.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Make tests run with predictable CWD
os.chdir(str(ROOT))
