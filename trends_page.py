import streamlit as st
import pandas as pd
import numpy as np # Needed for numerical operations like log10
import matplotlib.pyplot as plt # For custom plots if needed, though st.charts are preferred
import seaborn as sns # For styling if custom plots are used, not strictly for st.charts

def trends_page():
    # Navigation bar removed as requested. Navigation will be handled by the main app.

    st.title("Trends Analysis")
    st.write("Analyze trends in your latest processed aquatic dataset. Get deep intelligence on chlorophyll levels, water quality, and potential algal bloom indicators.")

    # Get the latest transformed dataset from session state
    df = st.session_state.get('main_df', None)
    
    if df is None or df.empty:
        st.warning("No processed data found. Please upload or process data in the Home page first.")
        st.info("Ensure you have successfully uploaded a file (CSV, JSON, XLSX, NC, HDF) or fetched satellite data, and that Florida chlorophyll data was extracted.")
        return

    st.subheader("Current Data Overview")
    st.dataframe(df.head())
    st.write(f"Total data points for analysis: {len(df):,}")

    # --- Identify Time Column ---
    time_col = None
    # Prioritize 'time' or 'date' in column names, then check dtypes
    possible_time_cols = [col for col in df.columns if 'time' in str(col).lower() or 'date' in str(col).lower()]
    
    for col in possible_time_cols:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            time_col = col
            break
        else: # Try converting if not already datetime
            try:
                # Use a copy to avoid SettingWithCopyWarning if original df is a slice
                temp_series = pd.to_datetime(df[col], errors='coerce')
                if not temp_series.isnull().all(): # Check if conversion was successful for at least some values
                    df[col] = temp_series # Update the column in the DataFrame
                    time_col = col
                    break
            except Exception:
                pass # Conversion failed, try next column

    if time_col:
        st.success(f"Time column identified: '{time_col}'.")
        # Drop rows where time_col became NaT after conversion
        df = df.dropna(subset=[time_col]).copy()
        if df.empty:
            st.warning("No valid time-based data points after cleaning. Time-based trends cannot be shown.")
            time_col = None # Reset time_col if no valid time data
    else:
        st.info("No suitable time column detected in the data. Time-based trend analysis will be limited or unavailable.")

    # Identify key environmental variables and numerical columns
    # These columns are expected from `process_satellite_data_to_florida_df`
    env_vars = ['Chlorophyll_mg_m3', 'Chlorophyll_log10', 'Distance_from_Shore_km']
    # Filter to only include those actually present and numeric
    env_vars = [c for c in env_vars if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    
    numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
    # Remove env_vars from numerical_cols if they are already explicitly handled
    numerical_cols = [c for c in numerical_cols if c not in env_vars]

    trend_type = st.radio("Select trend analysis type:", ["Time-based trends", "Non-time-based trends", "Categorical Distributions"])

    # --- Time-based Trends ---
    if trend_type == "Time-based trends":
        if time_col:
            st.subheader(f"Time-based Trends")
            st.info(f"Analyzing trends over time using the '{time_col}' column.")

            var_options = env_vars + numerical_cols # Combine for selection
            if not var_options:
                st.info("No suitable numerical variables found for time-based analysis.")
            else:
                selected_var = st.selectbox("Select variable for time trend analysis:", var_options, key="time_trend_var_select")
                
                # Ensure the selected variable is numeric
                if not pd.api.types.is_numeric_dtype(df[selected_var]):
                    st.warning(f"'{selected_var}' is not a numerical column. Please select a numerical variable for time trend analysis.")
                else:
                    freq = st.selectbox("Aggregate by:", ["Day", "Week", "Month", "Year"], key="time_trend_freq_select")
                    freq_map = {"Day": 'D', "Week": 'W', "Month": 'M', "Year": 'Y'}
                    
                    df_time = df[[time_col, selected_var]].dropna()
                    if not df_time.empty:
                        df_time = df_time.sort_values(time_col)
                        # Resample and calculate mean
                        df_time_resampled = df_time.set_index(time_col).resample(freq_map[freq])[selected_var].mean().reset_index()
                        
                        if not df_time_resampled.empty:
                            st.line_chart(df_time_resampled, x=time_col, y=selected_var)
                            st.write(f"Summary statistics for {selected_var} aggregated by {freq}:")
                            st.dataframe(df_time_resampled.describe())
                        else:
                            st.info(f"No data points after aggregating by {freq} for {selected_var}.")
                    else:
                        st.info(f"No valid data points for '{selected_var}' and '{time_col}' to perform time-based analysis.")
        else:
            st.info("No time column detected or valid after cleaning. Time-based trends cannot be displayed.")

    # --- Non-Time-Based Trends (Correlations) ---
    elif trend_type == "Non-time-based trends":
        st.subheader("Non-Time-Based Trends (Correlations)")
        st.info("Explore relationships between two numerical variables.")
        
        var_options = env_vars + numerical_cols
        if len(var_options) < 2:
            st.info("Need at least two numerical variables for non-time-based trend analysis.")
        else:
            # Ensure selected variables are distinct
            x_var = st.selectbox("Select X variable:", var_options, index=0, key="nontime_x_select")
            y_var = st.selectbox("Select Y variable:", var_options, index=1 if len(var_options) > 1 else 0, key="nontime_y_select")

            if x_var == y_var:
                st.warning("Please select two different variables for correlation analysis.")
            elif not pd.api.types.is_numeric_dtype(df[x_var]) or not pd.api.types.is_numeric_dtype(df[y_var]):
                st.warning("Both selected variables must be numerical.")
            else:
                df_scatter = df[[x_var, y_var]].dropna()
                if not df_scatter.empty:
                    st.scatter_chart(df_scatter, x=x_var, y=y_var)
                    correlation = df_scatter[x_var].corr(df_scatter[y_var])
                    st.write(f"Correlation between **{x_var}** and **{y_var}**: **{correlation:.3f}**")
                    st.write("Summary statistics:")
                    st.dataframe(df_scatter.describe())
                else:
                    st.info(f"No valid data points for '{x_var}' and '{y_var}' to perform scatter plot and correlation analysis.")

    # --- Categorical Distributions (New Section) ---
    elif trend_type == "Categorical Distributions":
        st.subheader("Categorical Distributions and Insights")
        st.info("Analyze how chlorophyll levels vary across predefined categories like Water Body, Florida Region, and Trophic Level.")

        # Trophic Level Distribution
        if 'Trophic_Level' in df.columns and not df['Trophic_Level'].isnull().all():
            st.markdown("#### Trophic Level Distribution")
            trophic_counts = df['Trophic_Level'].value_counts().sort_index()
            if not trophic_counts.empty:
                st.bar_chart(trophic_counts)
                st.write("Counts by Trophic Level:")
                st.dataframe(trophic_counts.to_frame(name='Count'))
                st.write("Mean Chlorophyll by Trophic Level:")
                st.dataframe(df.groupby('Trophic_Level')['Chlorophyll_mg_m3'].mean().to_frame(name='Mean Chlorophyll (mg/m³)'))
            else:
                st.info("No valid 'Trophic_Level' data found.")
        else:
            st.info("Trophic Level column not found or is empty. Please ensure data processing was successful.")

        # Chlorophyll by Water Body
        if 'Water_Body' in df.columns and not df['Water_Body'].isnull().all():
            st.markdown("#### Chlorophyll by Water Body")
            water_body_stats = df.groupby('Water_Body')['Chlorophyll_mg_m3'].agg(['count', 'mean', 'median', 'std']).round(4)
            if not water_body_stats.empty:
                st.bar_chart(water_body_stats['mean'])
                st.write("Chlorophyll Statistics by Water Body:")
                st.dataframe(water_body_stats)
            else:
                st.info("No valid 'Water_Body' data found.")
        else:
            st.info("Water Body column not found or is empty. Please ensure data processing was successful.")

        # Chlorophyll by Florida Region
        if 'Florida_Region' in df.columns and not df['Florida_Region'].isnull().all():
            st.markdown("#### Chlorophyll by Florida Region")
            region_stats = df.groupby('Florida_Region')['Chlorophyll_mg_m3'].agg(['count', 'mean', 'std']).round(4)
            region_stats = region_stats.sort_values('mean', ascending=False)
            if not region_stats.empty:
                st.bar_chart(region_stats['mean'])
                st.write("Chlorophyll Statistics by Florida Region:")
                st.dataframe(region_stats)
            else:
                st.info("No valid 'Florida_Region' data found.")
        else:
            st.info("Florida Region column not found or is empty. Please ensure data processing was successful.")

        # Chlorophyll by Shore Zone
        if 'Shore_Zone' in df.columns and not df['Shore_Zone'].isnull().all():
            st.markdown("#### Chlorophyll by Shore Zone")
            shore_stats = df.groupby('Shore_Zone')['Chlorophyll_mg_m3'].agg(['count', 'mean', 'std']).round(4)
            if not shore_stats.empty:
                st.bar_chart(shore_stats['mean'])
                st.write("Chlorophyll Statistics by Shore Zone:")
                st.dataframe(shore_stats)
            else:
                st.info("No valid 'Shore_Zone' data found.")
        else:
            st.info("Shore Zone column not found or is empty. Please ensure data processing was successful.")

        # Red Tide Risk Areas (Specific to Gulf of Mexico)
        st.markdown("#### Red Tide Risk Areas (Gulf of Mexico Focus)")
        red_tide_threshold = st.number_input("Red Tide Chlorophyll Threshold (mg/m³):", value=1.0, min_value=0.0, step=0.1)
        
        gulf_data = df[df['Water_Body'] == 'Gulf_of_Mexico'].copy()
        if not gulf_data.empty:
            high_chlor_for_red_tide = gulf_data[gulf_data['Chlorophyll_mg_m3'] > red_tide_threshold]
            st.write(f"Gulf of Mexico data points analyzed: {len(gulf_data):,}")
            st.write(f"Points above red tide threshold (>{red_tide_threshold} mg/m³): {len(high_chlor_for_red_tide):,}")

            if len(high_chlor_for_red_tide) > 0:
                st.warning(f"⚠️ Potential red tide areas detected based on this threshold!")
                st.write(f"Highest concentration in these areas: {high_chlor_for_red_tide['Chlorophyll_mg_m3'].max():.2f} mg/m³")
                most_affected_region = high_chlor_for_red_tide['Florida_Region'].mode()
                st.write(f"Most affected region(s): {', '.join(most_affected_region.tolist()) if not most_affected_region.empty else 'N/A'}")
                st.dataframe(high_chlor_for_red_tide[['Latitude', 'Longitude', 'Chlorophyll_mg_m3', 'Florida_Region']].nlargest(10, 'Chlorophyll_mg_m3'))
            else:
                st.success(f"✅ No red tide conditions detected (all values < {red_tide_threshold} mg/m³ in Gulf of Mexico for this sample).")
        else:
            st.info("No Gulf of Mexico data available for red tide analysis in this dataset.")