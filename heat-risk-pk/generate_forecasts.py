#!/usr/bin/env python3
"""
Generate forecast files for multiple horizons and scenarios.
Run this after training models.
"""

from src.forecast import main

if __name__ == "__main__":
    print("=" * 60)
    print("GENERATING FORECASTS")
    print("=" * 60)
    print("\nThis will create forecast files for:")
    print("  - Horizons: 6m, 12m, 24m, 72m (to 2030)")
    print("  - Scenarios: baseline, +1°C, +2°C")
    print("\nForecasts will be saved to: outputs/forecasts/")
    print("\n" + "=" * 60)
    
    main()
    
    print("\n" + "=" * 60)
    print("✅ FORECAST GENERATION COMPLETE")
    print("=" * 60)
