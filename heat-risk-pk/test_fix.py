#!/usr/bin/env python3
"""
Test script to verify the extend_forecast_to_year fix
"""
import pandas as pd
import numpy as np

def extend_forecast_to_year(df_in, target_end_year=2030):
    """
    Extends forecast table to target_end_year by repeating last 12-month pattern per city.
    This is a PoC projection layer to support long-horizon dashboards (2026–2030).
    
    IMPORTANT: Only extends if the forecast horizon is >= 12 months.
    For shorter forecasts (6m), returns data as-is without artificial extension.
    """
    df = df_in.copy().sort_values(["city", "year", "month"])
    
    # Determine forecast horizon by checking a sample city
    if not df.empty:
        sample_city = df["city"].iloc[0]
        sample_data = df[df["city"] == sample_city]
        forecast_horizon = len(sample_data)
        
        # If forecast horizon < 12 months, don't extend beyond what we have
        if forecast_horizon < 12:
            # Still add date column for consistency
            if "date" not in df.columns:
                df["date"] = pd.to_datetime(df["year"].astype(str) + "-" + df["month"].astype(str).str.zfill(2) + "-01")
            return df
    
    if int(df["year"].max()) >= target_end_year:
        if "date" not in df.columns:
            df["date"] = pd.to_datetime(df["year"].astype(str) + "-" + df["month"].astype(str).str.zfill(2) + "-01")
        return df

    out_chunks = [df]
    for city in df["city"].unique():
        d = df[df["city"] == city].copy().sort_values(["year", "month"])
        if d.empty:
            continue

        pattern = d.tail(12).copy()
        last_y = int(d.iloc[-1]["year"])
        last_m = int(d.iloc[-1]["month"])

        y, m = last_y, last_m
        while y < target_end_year or (y == target_end_year and m < 12):
            m += 1
            if m == 13:
                y += 1
                m = 1
            src = pattern[pattern["month"] == m]
            if src.empty:
                src = pattern.tail(1)
            row = src.iloc[0].to_dict()
            row["year"] = y
            row["month"] = m
            out_chunks.append(pd.DataFrame([row]))

    df_ext = pd.concat(out_chunks, ignore_index=True)
    df_ext["date"] = pd.to_datetime(df_ext["year"].astype(str) + "-" + df_ext["month"].astype(str).str.zfill(2) + "-01")
    return df_ext

# Load forecasts
df6 = pd.read_csv('outputs/forecasts/forecast_6m_baseline.csv')
df12 = pd.read_csv('outputs/forecasts/forecast_12m_baseline.csv')

print("=" * 70)
print("TESTING FIX: 6-MONTH FORECAST SHOULD NOT BE EXTENDED BEYOND 6 MONTHS")
print("=" * 70)

df6_extended = extend_forecast_to_year(df6, 2030)
multan6 = df6_extended[df6_extended['city'] == 'Multan'].sort_values(['year', 'month'])

print(f"\n✓ 6-month forecast for Multan:")
print(f"  - Rows in original 6m data: {len(df6[df6['city'] == 'Multan'])}")
print(f"  - Rows after extension: {len(multan6)}")
print(f"  - Extreme months (pred_risk=3): {(multan6['pred_risk'] == 3).sum()}")
print(f"\n  Data:")
print(multan6[['year', 'month', 'pred_risk']].to_string(index=False))

print("\n" + "=" * 70)
print("TESTING: 12-MONTH FORECAST SHOULD STILL BE EXTENDED TO 2030")
print("=" * 70)

df12_extended = extend_forecast_to_year(df12, 2030)
multan12 = df12_extended[df12_extended['city'] == 'Multan'].sort_values(['year', 'month'])

print(f"\n✓ 12-month forecast for Multan:")
print(f"  - Rows in original 12m data: {len(df12[df12['city'] == 'Multan'])}")
print(f"  - Rows after extension to 2030: {len(multan12)}")
print(f"  - Total extreme months: {(multan12['pred_risk'] == 3).sum()}")

multan12_first = multan12[multan12['year'] == 2024]
print(f"  - Extreme months in 2024: {(multan12_first['pred_risk'] == 3).sum()}")
print(f"\n  First 12 months (2024):")
print(multan12_first[['year', 'month', 'pred_risk']].to_string(index=False))

print("\n" + "=" * 70)
print("VERIFICATION: CONSISTENCY CHECK")
print("=" * 70)

# Both should have same prediction for overlapping months (Jan-Jun 2024)
df6_2024 = multan6[multan6['year'] == 2024]
df12_2024 = multan12[multan12['year'] == 2024].head(6)

comparison = pd.DataFrame({
    '6m': df6_2024['pred_risk'].values,
    '12m': df12_2024['pred_risk'].values
})

print("\n✓ 6-month vs 12-month predictions for overlapping months (Jan-Jun 2024):")
print(comparison.to_string())
print(f"\n✓ Match: {(comparison['6m'] == comparison['12m']).all()}")

# Check specific issue
extreme_may_jun_6m = len(df6_2024[(df6_2024['month'].isin([5, 6])) & (df6_2024['pred_risk'] == 3)])
extreme_may_jun_12m = len(df12_2024[(df12_2024['month'].isin([5, 6])) & (df12_2024['pred_risk'] == 3)])

print(f"\nExtreme months in May-June 2024:")
print(f"  - 6m forecast: {extreme_may_jun_6m} (should be 2)")
print(f"  - 12m forecast: {extreme_may_jun_12m} (should be 2)")

if extreme_may_jun_6m == 2 and extreme_may_jun_12m == 2:
    print("\n✅ FIX SUCCESSFUL: The inconsistency has been resolved!")
else:
    print("\n❌ Issue still exists")
