#!/usr/bin/env python3
"""
Generate forecast files for multiple horizons and scenarios.
Run this after training models.
"""

from pathlib import Path
import sys

# Allow `python src/generate_forecasts.py` from `heat-risk-pk` (not only `python -m ...`).
_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from src.config import SEQUENCE_CHECKPOINT_NAME
from src.forecast import main

if __name__ == "__main__":
    print("=" * 60)
    print("GENERATING FORECASTS (GRU + Attention)")
    print("=" * 60)
    print(f"\nUses `models/{SEQUENCE_CHECKPOINT_NAME}` (GRU) via `src/forecast_lstm.py`.")
    print("\nThis will create forecast files for:")
    print("  - Horizons: 6m, 12m, 24m, …, 72m (to 2030)")
    print("  - Scenarios: baseline, +1°C, +2°C")
    print("\nForecasts will be saved to: outputs/forecasts/")
    print("\n" + "=" * 60)
    
    main()
    
    print("\n" + "=" * 60)
    print("✅ FORECAST GENERATION COMPLETE")
    print("=" * 60)
