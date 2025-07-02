import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import io

# Helper Functions (same as before, but adapted for Streamlit input)
@st.cache_data
def read_excel_data(uploaded_file):
    if uploaded_file is not None:
        try:
            df = pd.read_excel(uploaded_file)
            excel_data = {
                "Stockpile1": df.iloc[:, [0, 1]].dropna(),
                "Stockpile2": df.iloc[:, [0, 2]].dropna(),
                "Stockpile3": df.iloc[:, [0, 3]].dropna(),
            }
            return excel_data
        except Exception as e:
            st.error(f"Failed to load Excel: {e}")
            return None
    return None

def blend_percentages(blend_ratio, dfs):
    base_array = dfs[0].iloc[:, 1].values.astype(float)
    blended = np.zeros_like(base_array, dtype=float)
    for i, r in enumerate(blend_ratio):
        blended += r * dfs[i].iloc[:, 1].values.astype(float)
    return blended

def calculate_retained(blended_passing):
    retained = np.zeros_like(blended_passing)
    retained[0] = 100 - blended_passing[0]
    for i in range(1, len(blended_passing)):
        retained[i] = blended_passing[i-1] - blended_passing[i]
    retained = np.clip(retained, 0, None)
    return retained

def max_density_line(sieve_sizes):
    return 100 * (sieve_sizes / max(sieve_sizes)) ** 0.45

def check_CA_ratio(blended_passing, sieve_sizes, sieve_value=2.36):
    idx = np.where(np.isclose(sieve_sizes, sieve_value))[0][0]
    ca_ratio = ((100 - blended_passing[idx]) / 100)*1.4
    return 0.6 <= ca_ratio <= 0.8, ca_ratio

def check_FA_CA_balance(blended_passing, sieve_sizes, Gmb, Gsb):
    if Gmb >= Gsb:
        return False, -999

    voids_percentage = (1 - Gmb / Gsb) * 100 * 10
    idx_max = np.where(np.isclose(sieve_sizes, 2.36))[0][0]
    idx_min = np.where(np.isclose(sieve_sizes, 0.075))[0][0]
    percent_fine_agg = (blended_passing[idx_min] - blended_passing[idx_max])
    percent_fine_agg = abs(percent_fine_agg)
    ca_fa_ratio = (percent_fine_agg / voids_percentage)*1.4 if voids_percentage != 0 else 0
    return 0.35 <= ca_fa_ratio <= 1.0, ca_fa_ratio


def check_filler_constraint(blended_passing, sieve_sizes, Veb):
    idx = np.where(np.isclose(sieve_sizes, 0.075))[0][0]
    filler_percent = blended_passing[idx]
    filler_binder_ratio = (filler_percent / Veb)*1.4
    return 0.2 <= filler_binder_ratio <= 0.5, filler_binder_ratio

def combined_objective(blend_ratio, dfs, sieve_sizes, binder_specific_gravity, absorbed_binder, Gsb, Gmb, binder_percent_user):
    blended = blend_percentages(blend_ratio, dfs)
    md_line = max_density_line(sieve_sizes)
    mse = np.sum((blended - md_line) ** 2)

    retained = calculate_retained(blended)
    Gi = np.array([2.65] * len(sieve_sizes))
    effective_binder = binder_percent_user - absorbed_binder
    if effective_binder <= 0:
        return 1e9
    Veb = (effective_binder * 10) / binder_specific_gravity
    denom = np.sum(6 * retained / (Gi * sieve_sizes))
    if denom <= 0:
        return 1e9

    _, ca_val = check_CA_ratio(blended, sieve_sizes)
    _, faca_val = check_FA_CA_balance(blended, sieve_sizes, Gmb, Gsb)
    _, filler_val = check_filler_constraint(blended, sieve_sizes, Veb)

    penalty = 0
    if ca_val < 0.6:
        penalty += (0.6 - ca_val) ** 2 * 10000
    elif ca_val > 0.8:
        penalty += (ca_val - 0.8) ** 2 * 10000

    if faca_val < 0.35:
        penalty += (0.35 - faca_val) ** 2 * 10000
    elif faca_val > 1.0:
        penalty += (faca_val - 1.0) ** 2 * 10000

    if filler_val < 0.6:
        penalty += (0.6 - filler_val) ** 2 * 10000
    elif filler_val > 1.2:
        penalty += (filler_val - 1.2) ** 2 * 10000

    return mse + penalty

# Streamlit App Layout
st.title("Performance-Based Bituminous Mix Design")

uploaded_file = st.file_uploader("Upload Excel with Sieve Data", type=["xlsx", "xls"])

excel_data = read_excel_data(uploaded_file)

st.header("Material Properties")

col1, col2 = st.columns(2)

with col1:
    binder_specific_gravity = st.number_input("Binder Specific Gravity:", value=1.03)
    absorbed_binder = st.number_input("Absorbed Binder (% of 1000g):", value=0.5)
    Gsb = st.number_input("Aggregate Bulk Specific Gravity:", value=2.65)

with col2:
    Gmb = st.number_input("Bulk Specific Gravity of coarse aggregate:", value=2.55)
    asphalt_film_thickness = st.number_input("Target Asphalt Film Thickness (micron):", value=8.0)
    binder_percent_user = st.number_input("Binder Percentage (%):", value=5.0)

if st.button("Run Blending Optimization"):
    if excel_data is None:
        st.error("Please upload the Excel file first.")
    elif None in [binder_specific_gravity, absorbed_binder, Gsb, Gmb, asphalt_film_thickness, binder_percent_user]:
        st.error("Please set all material properties before optimizing.")
    else:
        dfs = [excel_data["Stockpile1"], excel_data["Stockpile2"], excel_data["Stockpile3"]]
        sieve_sizes = dfs[0].iloc[:, 0].values.astype(float)

        init_ratio = np.array([0.33, 0.33, 0.34])
        bounds = [(0.1, 0.8)] * 3
        cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

        try:
            res = minimize(combined_objective, init_ratio, args=(dfs, sieve_sizes, binder_specific_gravity, absorbed_binder, Gsb, Gmb, binder_percent_user), bounds=bounds, constraints=cons)

            if res.success:
                optimized_ratio = res.x
                blended = blend_percentages(optimized_ratio, dfs)
                retained = calculate_retained(blended)

                Gi = np.array([2.65] * len(sieve_sizes))
                effective_binder = binder_percent_user - absorbed_binder
                Veb = (effective_binder * 10) / binder_specific_gravity
                denom = np.sum(6 * retained / (Gi * sieve_sizes))
                if denom > 0:
                  AFT_microns = ((Veb / denom) * 1000) / 4
                else:
                  AFT_microns = float('inf') # Handle division by zero

                ca_ok, ca_val = check_CA_ratio(blended, sieve_sizes)
                faca_ok, CA_FA_ratio = check_FA_CA_balance(blended, sieve_sizes, Gmb, Gsb)
                filler_ok, filler_ratio = check_filler_constraint(blended, sieve_sizes, Veb)

                st.subheader("Optimization Results")
                st.write("Optimized Blending Ratio:")
                st.write(f"  Stockpile 1: {optimized_ratio[0]:.2f}")
                st.write(f"  Stockpile 2: {optimized_ratio[1]:.2f}")
                st.write(f"  Stockpile 3: {optimized_ratio[2]:.2f}")

                st.write("\nSieve Size (mm)\tBlended Passing (%)")
                for s, p in zip(sieve_sizes, blended):
                    st.write(f"{s:.2f}\t\t{p:.2f}")

                st.subheader("Final Binder Content Result:")
                st.write(f"Binder %: {binder_percent_user:.2f}")
                st.write(f"AFT (micron): {AFT_microns:.2f}")
                st.write(f"CA ratio: {ca_val:.4f} {'✅' if ca_ok else '❌'}")
                st.write(f"CA/FA Ratio: {CA_FA_ratio:.4f} {'✅' if faca_ok else '❌'}")
                st.write(f"Filler/Binder Ratio: {filler_ratio:.4f} {'✅' if filler_ok else '❌'}")

            else:
                st.error("Optimization failed.")
                st.write(f"Optimization result: {res}")

        except Exception as e:
            st.error(f"An error occurred during optimization: {e}")
