import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import io

# Page configuration
st.set_page_config(
    page_title="Bituminous Mix Design Optimizer",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
.header {
    font-size: 36px !important;
    font-weight: bold;
    color: #2c3e50;
    text-align: center;
    padding: 20px;
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    border-radius: 10px;
    margin-bottom: 30px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}

.subheader {
    font-size: 24px !important;
    font-weight: bold;
    color: #3498db;
    border-bottom: 2px solid #3498db;
    padding-bottom: 10px;
    margin-top: 25px;
}

.stButton>button {
    background: linear-gradient(135deg, #3498db 0%, #2c3e50 100%);
    color: white;
    font-weight: bold;
    border: none;
    border-radius: 5px;
    padding: 10px 20px;
    transition: all 0.3s ease;
}

.stButton>button:hover {
    transform: scale(1.05);
    box-shadow: 0 4px 8px rgba(0,0,0,0.2);
}

.info-box {
    background-color: #e3f2fd;
    border-left: 4px solid #3498db;
    padding: 15px;
    border-radius: 0 5px 5px 0;
    margin: 20px 0;
}

.success-box {
    background-color: #e8f5e9;
    border-left: 4px solid #4caf50;
    padding: 15px;
    border-radius: 0 5px 5px 0;
    margin: 20px 0;
}

.warning-box {
    background-color: #fff3e0;
    border-left: 4px solid #ff9800;
    padding: 15px;
    border-radius: 0 5px 5px 0;
    margin: 20px 0;
}

.card {
    background: white;
    border-radius: 10px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    padding: 20px;
    margin-bottom: 20px;
}

.data-table {
    width: 100%;
    border-collapse: collapse;
    margin: 20px 0;
}

.data-table th {
    background-color: #3498db;
    color: white;
    text-align: left;
    padding: 12px;
}

.data-table tr:nth-child(even) {
    background-color: #f2f2f2;
}

.data-table tr:hover {
    background-color: #e3f2fd;
}

.data-table td {
    padding: 12px;
    border-bottom: 1px solid #ddd;
}
</style>
""", unsafe_allow_html=True)

# Helper Functions
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
    ca_ratio = (100 - blended_passing[idx]) / 100
    return 0.6 <= ca_ratio <= 0.8, ca_ratio

def check_FA_CA_balance(blended_passing, sieve_sizes, Gmb, Gsb):
    if Gmb >= Gsb:
        return False, -999

    voids_percentage = (1 - Gmb / Gsb) * 100 * 10
    idx_max = np.where(np.isclose(sieve_sizes, 2.36))[0][0]
    idx_min = np.where(np.isclose(sieve_sizes, 0.075))[0][0]
    percent_fine_agg = (blended_passing[idx_min] - blended_passing[idx_max])
    percent_fine_agg = abs(percent_fine_agg)
    ca_fa_ratio = percent_fine_agg / voids_percentage if voids_percentage != 0 else 0
    return 0.35 <= ca_fa_ratio <= 1.0, ca_fa_ratio

def check_filler_constraint(blended_passing, sieve_sizes, Veb):
    idx = np.where(np.isclose(sieve_sizes, 0.075))[0][0]
    filler_percent = blended_passing[idx]
    filler_binder_ratio = (filler_percent / Veb) * 2
    return 0.6 <= filler_binder_ratio <= 1.2, filler_binder_ratio

def combined_objective(blend_ratio, dfs, sieve_sizes, binder_specific_gravity, binder_percent_user, Gmb, Gsb):
    blended = blend_percentages(blend_ratio, dfs)
    md_line = max_density_line(sieve_sizes)
    mse = np.sum((blended - md_line) ** 2)

    retained = calculate_retained(blended)
    Gi = np.array([2.65] * len(sieve_sizes))
    Veb = (binder_percent_user * 10) / binder_specific_gravity
    denom = np.sum(6 * retained / (Gi * sieve_sizes))
    if denom <= 0:
        return 1e9
    AFT_mm = Veb / denom
    AFT_microns = (AFT_mm * 1000) / 4

    ca_ok, _ = check_CA_ratio(blended, sieve_sizes)
    faca_ok, _ = check_FA_CA_balance(blended, sieve_sizes, Gmb, Gsb)
    filler_ok, _ = check_filler_constraint(blended, sieve_sizes, Veb)

    penalty = 0
    if not ca_ok:
        penalty += 1e6
    if not faca_ok:
        penalty += 1e6
    if not filler_ok:
        penalty += 1e6

    return mse + penalty

# Main App
def main():
    # Header
    st.markdown('<div class="header">Bituminous Mix Design Optimization</div>', unsafe_allow_html=True)
    
    # Info box
    with st.expander("About this tool"):
        st.info("""
        This application optimizes bituminous mix designs using performance-based criteria. 
        It calculates the optimal blend ratios for three stockpiles to achieve target properties including:
        - Asphalt film thickness
        - Coarse aggregate ratio
        - Fine to coarse aggregate balance
        - Filler to binder ratio
        
        Upload an Excel file with sieve analysis data and set material properties to get started.
        """)
    
    # Initialize session state
    if 'excel_data' not in st.session_state:
        st.session_state.excel_data = None
    
    if 'results' not in st.session_state:
        st.session_state.results = None
    
    # Layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="subheader">Upload Sieve Analysis Data</div>', unsafe_allow_html=True)
        
        # File upload
        uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx", "xls"], 
                                         help="Excel file should have three columns: Sieve Size (mm), Stockpile1 (%), Stockpile2 (%), Stockpile3 (%)")
        
        if uploaded_file is not None:
            try:
                df = pd.read_excel(uploaded_file)
                st.session_state.excel_data = {
                    "Stockpile1": df.iloc[:, [0, 1]].dropna(),
                    "Stockpile2": df.iloc[:, [0, 2]].dropna(),
                    "Stockpile3": df.iloc[:, [0, 3]].dropna(),
                }
                
                # Show preview
                st.success("File uploaded successfully!")
                st.write("Data Preview:")
                st.dataframe(df.head(10))
                
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
    
    with col2:
        st.markdown('<div class="subheader">Material Properties</div>', unsafe_allow_html=True)
        
        # Material properties form
        with st.form("material_properties"):
            binder_specific_gravity = st.number_input("Binder Specific Gravity", min_value=0.8, max_value=2.0, value=1.02, step=0.01,
                                                     help="Specific gravity of the asphalt binder")
            
            absorbed_binder = st.number_input("Absorbed Binder (% of 1000g)", min_value=0.0, max_value=10.0, value=0.8, step=0.1,
                                            help="Percentage of binder absorbed by the aggregate")
            
            Gsb = st.number_input("Bulk Specific Gravity (Aggregate)", min_value=2.0, max_value=3.0, value=2.65, step=0.01,
                                 help="Bulk specific gravity of the aggregate")
            
            Gmb = st.number_input("Bulk Density (Aggregate)", min_value=1.0, max_value=3.0, value=2.45, step=0.01,
                                 help="Bulk density of the aggregate mixture")
            
            asphalt_film_thickness = st.number_input("Target Asphalt Film Thickness (micron)", min_value=1.0, max_value=20.0, value=8.0, step=0.1,
                                                    help="Desired asphalt film thickness around aggregate particles")
            
            binder_percent_user = st.number_input("Binder Percentage (%)", min_value=1.0, max_value=10.0, value=5.0, step=0.1,
                                                 help="Percentage of binder in the mix")
            
            submitted = st.form_submit_button("Set Properties & Run Optimization")
    
    # Run optimization when form is submitted
    if submitted:
        if st.session_state.excel_data is None:
            st.error("Please upload an Excel file first.")
            return
        
        try:
            dfs = [st.session_state.excel_data["Stockpile1"], 
                  st.session_state.excel_data["Stockpile2"], 
                  st.session_state.excel_data["Stockpile3"]]
            
            sieve_sizes = dfs[0].iloc[:, 0].values.astype(float)
            
            init_ratio = np.array([0.33, 0.33, 0.34])
            bounds = [(0.1, 0.8)] * 3
            cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
            
            res = minimize(combined_objective, init_ratio, 
                          args=(dfs, sieve_sizes, binder_specific_gravity, binder_percent_user, Gmb, Gsb), 
                          bounds=bounds, constraints=cons)
            
            if res.success:
                optimized_ratio = res.x
                blended = blend_percentages(optimized_ratio, dfs)
                retained = calculate_retained(blended)
                
                Gi = np.array([2.65] * len(sieve_sizes))
                Veb = (binder_percent_user * 10) / binder_specific_gravity
                denom = np.sum(6 * retained / (Gi * sieve_sizes))
                
                if denom > 0:
                    AFT_mm = Veb / denom
                    AFT_microns = (AFT_mm * 1000) / 4
                else:
                    AFT_microns = 0
                
                ca_ok, ca_val = check_CA_ratio(blended, sieve_sizes)
                faca_ok, CA_FA_ratio = check_FA_CA_balance(blended, sieve_sizes, Gmb, Gsb)
                filler_ok, filler_ratio = check_filler_constraint(blended, sieve_sizes, Veb)
                
                # Save results to session state
                st.session_state.results = {
                    'ratios': optimized_ratio,
                    'sieve_sizes': sieve_sizes,
                    'blended': blended,
                    'binder_percent': binder_percent_user,
                    'AFT_microns': AFT_microns,
                    'ca_val': ca_val,
                    'CA_FA_ratio': CA_FA_ratio,
                    'filler_ratio': filler_ratio,
                    'ca_ok': ca_ok,
                    'faca_ok': faca_ok,
                    'filler_ok': filler_ok
                }
                
                st.success("Optimization completed successfully!")
                
            else:
                st.error("Optimization failed. Please check your input data and try again.")
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    
    # Display results if available
    if st.session_state.results:
        results = st.session_state.results
        
        st.markdown('<div class="subheader">Optimization Results</div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Blending Ratios**")
            st.metric("Stockpile 1", f"{results['ratios'][0]*100:.2f}%")
            st.metric("Stockpile 2", f"{results['ratios'][1]*100:.2f}%")
            st.metric("Stockpile 3", f"{results['ratios'][2]*100:.2f}%")
        
        with col2:
            st.markdown("**Key Parameters**")
            st.metric("Asphalt Film Thickness", f"{results['AFT_microns']:.2f} microns", 
                     help="Thickness of asphalt film coating aggregate particles")
            st.metric("Coarse Aggregate Ratio", f"{results['ca_val']:.3f}", 
                     "✅ Within range" if results['ca_ok'] else "⚠️ Outside range",
                     help="CA ratio should be between 0.6-0.8")
        
        with col3:
            st.markdown("**Mix Ratios**")
            st.metric("Fine to Coarse Aggregate Ratio", f"{results['CA_FA_ratio']:.3f}", 
                     "✅ Within range" if results['faca_ok'] else "⚠️ Outside range",
                     help="CA/FA ratio should be between 0.35-1.0")
            st.metric("Filler to Binder Ratio", f"{results['filler_ratio']:.3f}", 
                     "✅ Within range" if results['filler_ok'] else "⚠️ Outside range",
                     help="Filler/Binder ratio should be between 0.6-1.2")
        
        # Gradation data
        st.markdown("**Aggregate Gradation**")
        
        # Create a DataFrame for the results
        gradation_data = pd.DataFrame({
            'Sieve Size (mm)': results['sieve_sizes'],
            'Passing (%)': results['blended'],
            'Max Density Line': max_density_line(results['sieve_sizes'])
        })
        
        # Add retained calculations
        retained = calculate_retained(results['blended'])
        gradation_data['Retained (%)'] = retained
        
        # Display the table
        st.dataframe(gradation_data.style.format({
            'Sieve Size (mm)': '{:.2f}',
            'Passing (%)': '{:.2f}',
            'Max Density Line': '{:.2f}',
            'Retained (%)': '{:.2f}'
        }).applymap(lambda x: 'background-color: #e8f5e9' if x > 90 else '', 
                    subset=['Passing (%)']), height=400)
        
        # Download results
        csv = gradation_data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Gradation Data as CSV",
            data=csv,
            file_name="bituminous_mix_design_results.csv",
            mime="text/csv"
        )
        
        # Visual representation
        st.markdown("**Gradation Curve**")
        gradation_chart_data = gradation_data.melt(id_vars='Sieve Size (mm)', 
                                                 value_vars=['Passing (%)', 'Max Density Line'], 
                                                 var_name='Type', value_name='Percentage')
        
        st.line_chart(gradation_chart_data, x='Sieve Size (mm)', y='Percentage', color='Type',
                     use_container_width=True)

# Run the app
if __name__ == "__main__":
    main()
