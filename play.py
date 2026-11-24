"""Shortcut: `uv run play.py` == `uv run main.py play` (watch the best agent)."""
import sys
sys.argv.insert(1, "play")
from main import main
main()
