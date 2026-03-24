import streamlit as st
import pandas as pd
from src.modelling.trip_generation import calculate_regression_trips, cross_classification_trips, get_sample_trip_rates

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
        
        method = st.radio("Choose Method", ["Regression Analysis", "Category Analysis (Cross-Classification)"])
        
        if method == "Regression Analysis":
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
            
            data = pd.DataFrame({
                "Variable": ["Population Component", "Employment Component", "Intercept"],
                "Value": [pop * p_coeff, emp * e_coeff, intercept]
            })
            st.bar_chart(data, x="Variable", y="Value")
            
        else:
            with st.expander("Theoretical Concept: Category Analysis"):
                st.write("""
                Category Analysis (or Cross-Classification) groups households into categories 
                based on variables like household size, income, or car ownership.
                
                Total trips are calculated by multiplying the number of households in each 
                category by their observed average trip rate.
                
                $$T = \sum_{h \in H} N_h \cdot R_h$$
                Where $N_h$ is the number of households in category $h$, and $R_h$ is the trip rate for that category.
                """)

            st.subheader("Exercise: Household Trip Estimation")
            
            # Display Trip Rate Table
            st.markdown("### 1. Trip Rate Table (Avg Trips/HH)")
            rates_df = get_sample_trip_rates()
            edited_rates = st.data_editor(rates_df)
            
            # Input Household Counts
            st.markdown("### 2. Number of Households per Category")
            hh_counts = {}
            cols = st.columns(len(rates_df.columns))
            
            for i, car_cat in enumerate(rates_df.columns):
                with cols[i]:
                    st.write(f"**{car_cat}**")
                    for size_cat in rates_df.index:
                        key = f"{size_cat}_{car_cat}"
                        val = st.number_input(f"{size_cat}", min_value=0, value=10, key=key)
                        hh_counts[key] = val
            
            # Prepare rates for calculation
            flat_rates = {}
            for car_cat in edited_rates.columns:
                for size_cat in edited_rates.index:
                    flat_rates[f"{size_cat}_{car_cat}"] = edited_rates.loc[size_cat, car_cat]
            
            total_trips = cross_classification_trips(hh_counts, flat_rates)
            
            st.divider()
            st.metric("Total Trips Generated", f"{total_trips:,.0f}")
            
            # Summary visualization
            res_list = []
            for k, v in hh_counts.items():
                size, car = k.split("_")
                res_list.append({"Size": size, "Cars": car, "Trips": v * flat_rates[k]})
            
            res_df = pd.DataFrame(res_list)
            st.vega_lite_chart(res_df, {
                'mark': {'type': 'bar', 'tooltip': True},
                'encoding': {
                    'x': {'field': 'Size', 'type': 'nominal'},
                    'y': {'field': 'Trips', 'type': 'quantitative', 'aggregate': 'sum'},
                    'color': {'field': 'Cars', 'type': 'nominal'}
                }
            }, use_container_width=True)

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
