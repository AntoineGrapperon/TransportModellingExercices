import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from src.modelling.trip_generation import calculate_regression_trips, cross_classification_trips, get_sample_trip_rates
from src.modelling.trip_distribution import gravity_model, furness_balancing
from src.modelling.modal_split import calculate_utilities, multinomial_logit
from src.modelling.traffic_assignment import bpr_function, solve_2path_equilibrium

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

    elif module == "2. Trip Distribution":
        st.header("Step 2: Trip Distribution")
        
        with st.expander("Theoretical Concept: The Gravity Model"):
            st.write("""
            Trip distribution links the origins (Productions) to destinations (Attractions).
            The **Gravity Model** is based on Newton's law of gravitation:
            
            $$T_{ij} = P_i \frac{A_j F_{ij}}{\sum_j A_j F_{ij}}$$
            
            Where:
            - $T_{ij}$ = Trips between zone $i$ and zone $j$
            - $P_i$ = Trips produced in zone $i$
            - $A_j$ = Trips attracted to zone $j$
            - $F_{ij}$ = Friction factor (typically $\exp(-\beta C_{ij})$)
            """)

        st.subheader("Exercise: Gravity Model Calibration")
        
        num_zones = st.slider("Number of Zones", 2, 5, 3)
        zone_names = [f"Zone {i+1}" for i in range(num_zones)]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Productions (P_i)")
            p_data = pd.DataFrame({"Zone": zone_names, "Trips": [1000] * num_zones})
            edited_p = st.data_editor(p_data, key="p_editor", hide_index=True)
            
        with col2:
            st.markdown("### Attractions (A_j)")
            a_data = pd.DataFrame({"Zone": zone_names, "Trips": [1000] * num_zones})
            edited_a = st.data_editor(a_data, key="a_editor", hide_index=True)
            
        st.markdown("### Cost Matrix (C_ij)")
        cost_df = pd.DataFrame(
            np.eye(num_zones) * 5 + 10, # 5 for intra-zonal, 10 for inter-zonal
            columns=zone_names,
            index=zone_names
        )
        edited_costs = st.data_editor(cost_df)
        
        beta = st.slider("Beta (Friction Sensitivity)", 0.0, 1.0, 0.1, step=0.01)
        
        # Calculate Distributed Matrix
        p_array = edited_p["Trips"].values
        a_array = edited_a["Trips"].values
        cost_matrix = edited_costs.values
        
        od_matrix = gravity_model(p_array, a_array, cost_matrix, beta)
        
        # Display Results
        st.divider()
        st.subheader("Resulting O-D Matrix (Trip Matrix)")
        od_df = pd.DataFrame(od_matrix, columns=zone_names, index=zone_names)
        st.dataframe(od_df.style.format("{:.0f}"))
        
        # Heatmap
        fig = px.imshow(
            od_matrix, 
            labels=dict(x="Destination", y="Origin", color="Trips"),
            x=zone_names, 
            y=zone_names,
            text_auto=".0f",
            aspect="auto",
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig, use_container_width=True)

    elif module == "3. Modal Split":
        st.header("Step 3: Modal Split")
        
        with st.expander("Theoretical Concept: Discrete Choice & Utility"):
            st.write("""
            Modal Split determines the proportion of travelers choosing each transport mode.
            The **Multinomial Logit (MNL)** model is widely used:
            
            $$P_m = \\frac{\\exp(V_m)}{\\sum_{n} \\exp(V_n)}$$
            
            Where $V_m$ is the utility of mode $m$:
            $$V_m = ASC_m + \\beta_{time} \\cdot T_m + \\beta_{cost} \\cdot C_m$$
            
            - $ASC_m$: Alternative Specific Constant
            - $\\beta_{time}, \\beta_{cost}$: Sensitivity coefficients (typically negative)
            """)

        st.subheader("Exercise: Utility & Choice Probabilities")
        
        col1, col2 = st.columns([2, 1])
        
        with col2:
            st.markdown("### Global Sensitivity")
            b_time = st.slider("Beta Time (Travel Time sensitivity)", -0.10, 0.0, -0.02, step=0.005)
            b_cost = st.slider("Beta Cost (Travel Cost sensitivity)", -0.50, 0.0, -0.05, step=0.01)
            coeffs = {"beta_time": b_time, "beta_cost": b_cost}
            
            if b_cost != 0:
                vot = b_time / b_cost
                st.info(f"Implied Value of Time (VoT): **{abs(vot * 60):.2f} $/hr**")

        with col1:
            st.markdown("### Mode Characteristics")
            modes_df = pd.DataFrame([
                {"Mode": "Car", "ASC": 0.0, "Time (min)": 20, "Cost ($)": 5.0},
                {"Mode": "Bus", "ASC": -0.5, "Time (min)": 35, "Cost ($)": 2.0},
                {"Mode": "Train", "ASC": -0.2, "Time (min)": 25, "Cost ($)": 3.5}
            ])
            edited_modes = st.data_editor(modes_df, key="modes_editor", hide_index=True)
            
        # Preparation for calculation
        modes_list = []
        for _, row in edited_modes.iterrows():
            modes_list.append({
                "asc": row["ASC"],
                "time": row["Time (min)"],
                "cost": row["Cost ($)"]
            })
            
        utilities = calculate_utilities(modes_list, coeffs)
        probs = multinomial_logit(utilities)
        
        # Results
        st.divider()
        res_df = edited_modes.copy()
        res_df["Utility"] = utilities
        res_df["Probability"] = probs
        res_df["Share (%)"] = probs * 100
        
        st.subheader("Results: Mode Shares")
        cols = st.columns(len(res_df))
        for i, row in res_df.iterrows():
            with cols[i]:
                st.metric(row["Mode"], f"{row['Share (%)']:.1f}%")
        
        st.dataframe(res_df.style.format({"Utility": "{:.2f}", "Probability": "{:.4f}", "Share (%)": "{:.1f}%"}))
        
        # Visualizations
        fig_pie = px.pie(res_df, values='Probability', names='Mode', title='Mode Share Distribution', hole=.3)
        st.plotly_chart(fig_pie, use_container_width=True)

    elif module == "4. Traffic Assignment":
        st.header("Step 4: Traffic Assignment")
        
        with st.expander("Theoretical Concept: Link Performance & Equilibrium"):
            st.write("""
            Traffic Assignment predicts which routes travelers will take on the network.
            
            **Link Performance (BPR Function):**
            Travel time increases as the flow ($V$) approaches capacity ($C$):
            $$T = T_0 \\cdot (1 + \\alpha (V/C)^\\beta)$$
            
            **User Equilibrium (Wardrop's 1st Principle):**
            "The journey times in all routes actually used are equal and less than those which would be experienced by a single vehicle on any unused route."
            """)

        st.subheader("Exercise: 2-Path User Equilibrium")
        
        st.markdown("### Scenario: Origin O to Destination D")
        demand = st.number_input("Total Demand (Veh/hr)", min_value=0, value=2500, step=500)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Path 1 (Highway)")
            t0_1 = st.number_input("Free Flow Time 1 (min)", value=10.0, key="t01")
            cap_1 = st.number_input("Capacity 1 (Veh/hr)", value=1800.0, key="cap1")
            
        with col2:
            st.markdown("### Path 2 (Local Road)")
            t0_2 = st.number_input("Free Flow Time 2 (min)", value=15.0, key="t02")
            cap_2 = st.number_input("Capacity 2 (Veh/hr)", value=1200.0, key="cap2")
            
        params1 = {"t0": t0_1, "cap": cap_1, "alpha": 0.15, "beta": 4.0}
        params2 = {"t0": t0_2, "cap": cap_2, "alpha": 0.15, "beta": 4.0}
        
        # Calculate Equilibrium
        f1, f2, tt = solve_2path_equilibrium(demand, params1, params2)
        
        st.divider()
        st.subheader("Equilibrium Results")
        
        met1, met2, met3 = st.columns(3)
        met1.metric("Flow Path 1", f"{f1:,.0f} veh/hr")
        met2.metric("Flow Path 2", f"{f2:,.0f} veh/hr")
        met3.metric("Travel Time", f"{tt:.2f} min")
        
        # Visualization: Sensitivity of Travel Time to Flow
        flow_range = np.linspace(0, demand, 100)
        t1_vals = [bpr_function(t0_1, cap_1, f, 0.15, 4.0) for f in flow_range]
        t2_vals = [bpr_function(t0_2, cap_2, demand - f, 0.15, 4.0) for f in flow_range]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=flow_range, y=t1_vals, name="Path 1 Time", line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=flow_range, y=t2_vals, name="Path 2 Time", line=dict(color='red')))
        
        # Mark Equilibrium
        fig.add_trace(go.Scatter(
            x=[f1], y=[tt], 
            mode='markers+text', 
            name="Equilibrium",
            text=["User Equilibrium"],
            textposition="top center",
            marker=dict(color='green', size=12, symbol='star')
        ))
        
        fig.update_layout(
            title="User Equilibrium Search",
            xaxis_title="Flow on Path 1 (veh/hr)",
            yaxis_title="Travel Time (min)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
