import numpy as np
import joblib
import streamlit as st
import matplotlib.pyplot as plt

# Load the pre-trained model
model = joblib.load("XGBoost_best_of_10.joblib")

def predict_hc(mt_content, liquid_limit, plastic_limit, specific_gravity, 
               initial_wc, temperature, dry_density_values):
    warnings = []
    predictions = []
    
    # Process dry density values
    try:
        density_values = [float(x.strip()) for x in dry_density_values.split(',')]
        if len(density_values) < 4 or len(density_values) > 8:
            st.error("Please enter 4 to 8 values separated by commas.")
            return None, None
    except ValueError:
        st.error("Error in dry density input. Ensure values are numeric and comma-separated.")
        return None, None
    
    # Define trained ranges
    trained_ranges = {
        "Montmorillonite Content": (33.60, 92.00),
        "Liquid Limit": (50.00, 500.00),
        "Plastic Limit": (20.00, 200.00),
        "Specific Gravity": (2.65, 2.82),
        "Initial Water Content": (5.40, 21.10),
        "Temperature": (20.00, 40.00),
        "Dry Density": (1.30, 2.05),
    }
    
    # Check input values against trained ranges
    input_values = {
        "Montmorillonite Content": mt_content,
        "Liquid Limit": liquid_limit,
        "Plastic Limit": plastic_limit,
        "Specific Gravity": specific_gravity,
        "Initial Water Content": initial_wc,
        "Temperature": temperature,
    }
    
    for feature, value in input_values.items():
        min_val, max_val = trained_ranges[feature]
        if value < min_val or value > max_val:
            warnings.append(f"⚠️ {feature} is outside the trained range ({min_val}-{max_val}).")
    
    # Make predictions
    fixed_features = np.array([
        mt_content, liquid_limit, plastic_limit, specific_gravity, initial_wc, temperature
    ])
    
    for density in density_values:
        if not (trained_ranges["Dry Density"][0] <= density <= trained_ranges["Dry Density"][1]):
            warnings.append(f"⚠️ Dry Density {density} is outside the trained range.")
        input_array = np.append(fixed_features, density).reshape(1, -1)
        predictions.append(model.predict(input_array)[0])
    
    return predictions, density_values, warnings

# Streamlit UI
st.title("Saturated Hydraulic Conductivity Prediction Model")
st.markdown("Predict saturated hydraulic conductivity (HC) based on soil properties.")

# Input fields
mt_content = st.number_input("Montmorillonite Content [%]", 33.60, 92.00, 60.00, step=0.1)
liquid_limit = st.number_input("Liquid Limit [%]", 50.00, 500.00, 150.00, step=1)
plastic_limit = st.number_input("Plastic Limit [%]", 20.00, 200.00, 50.00, step=1)
specific_gravity = st.number_input("Specific Gravity", 2.65, 2.82, 2.70, step=0.01)
initial_wc = st.number_input("Initial Water Content [%]", 5.40, 21.10, 10.00, step=0.1)
temperature = st.number_input("Temperature [°C]", 20.00, 40.00, 25.00, step=0.1)
dry_density_values = st.text_input("Dry Density values [g/cm³] (4-8 values, comma separated)", "1.3, 1.5, 1.7, 1.9")

if st.button("Predict HC"):
    predictions, density_values, warnings = predict_hc(
        mt_content, liquid_limit, plastic_limit, specific_gravity, initial_wc, temperature, dry_density_values
    )
    
    if predictions:
        st.subheader("Prediction Results:")
        for d, p in zip(density_values, predictions):
            st.write(f"At ρ = {d:.2f} g/cm³: HC = {p:.4f} [-]")
        
        # Plot results
        fig, ax = plt.subplots()
        ax.plot(density_values, predictions, marker="o", linestyle="-", color="blue")
        ax.set_xlabel("Dry Density [g/cm³]")
        ax.set_ylabel("Saturated Hydraulic Conductivity (HC) [-]")
        ax.grid(True, linestyle="--", alpha=0.5)
        st.pyplot(fig)
    
    if warnings:
        st.warning("\n".join(warnings))

st.markdown("""  
**Developed by:**  
Muntasir Shehab*, Reza Taherdangkoo and Christoph Butscher  

**Institution:**  
TU Bergakademie Freiberg, Institute of Geotechnics  
Gustav-Zeuner-Str. 1, Freiberg, 09599, Germany  
""")
