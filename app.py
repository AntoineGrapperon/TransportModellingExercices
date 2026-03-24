import streamlit as st
import pandas as pd
from src.modelling.trip_generation import calculate_regression_trips

def main():
    st.set_page_config(page_title="Transport Modelling Exercises", layout="wide")
    
    st.sidebar.title("4-Step Demand Model")
    module = st.sidebar.selectbox(
        "Select a Module",
        ["Introduction", "1. Trip Generation", "2. Trip Distribution", "3. Modal Split", "4. Traffic Assignment"]
    )
    
    if module == "Introduction":
        st.header("Transport Demand Modelling: The 4-Step Model")
        st.markdown("""
        ### Educational Overview
        The 4-step model is the traditional approach to urban transportation planning.
        1. **Trip Generation**: How many trips start and end in each zone?
        2. **Trip Distribution**: Where do these trips go?
        3. **Modal Split**: What transport modes are used?
        4. **Traffic Assignment**: What routes are taken?
        """)
        
    elif module == "1. Trip Generation":
        st.header("Step 1: Trip Generation")
        
        with st.expander("Theoretical Concept: Regression Analysis"):
            st.write("""
            Trip generation models estimate the number of trips originating (Productions) 
            or terminating (Attractions) in a zone based on zonal characteristics.
            
            **Linear Regression Model:**
            $$T_i = a + b_1 P_i + b_2 E_i + ...$$
            Where:
            - $T_i$ = Trips in zone $i$
            - $a$ = Constant/Intercept
            - $P_i$ = Population in zone $i$
            - $E_i$ = Employment in zone $i$
            """)

        st.subheader("Exercise: Calculate Zonal Trips")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Zonal Inputs")
            pop = st.number_input("Population (Zone i)", min_value=0, value=1500, step=100)
            emp = st.number_input("Employment (Zone i)", min_value=0, value=400, step=50)
            
        with col2:
            st.markdown("### Model Parameters")
            intercept = st.number_input("Intercept (a)", value=100.0)
            p_coeff = st.number_input("Pop Coefficient (b1)", value=0.45)
            e_coeff = st.number_input("Emp Coefficient (b2)", value=1.2)
            
        coeffs = {
            "intercept": intercept,
            "pop_coeff": p_coeff,
            "emp_coeff": e_coeff
        }
        
        trips = calculate_regression_trips(pop, emp, coeffs)
        
        st.divider()
        st.metric("Estimated Total Trips (T_i)", f"{trips:,.0f}")
        
        # Interactive chart
        data = pd.DataFrame({
            "Variable": ["Population Component", "Employment Component", "Intercept"],
            "Value": [pop * p_coeff, emp * e_coeff, intercept]
        })
        st.bar_chart(data, x="Variable", y="Value")

if __name__ == "__main__":
    main()
