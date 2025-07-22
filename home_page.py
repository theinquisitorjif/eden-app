# home page of the streamlit application for EDEN AI
# Take time based data and/or satellite data based on what the user selects
# The data can be uploaded in either csv or json or xlsx or nc
# For satellite data pulling if requested by the user then get ready to pull the data from the API
import streamlit as st
import pandas as pd
import json
import matplotlib.pyplot as plt
import xarray as xr # Import xarray for NetCDF files
import earthaccess # For fetching satellite data from Earthdata
import os # For temporary file handling and path manipulation
import tempfile # For temporary file handling
from datetime import date, timedelta
import shutil # For removing directories
import numpy as np # Needed for numerical operations in data processing
import glob # For potential future use, not directly in Streamlit file uploader context now
import seaborn as sns # Used for plotting in the template, but we will use Streamlit's built-in plotting or leave it to other pages


# --- IMPORTANT: Ensure these libraries are installed ---
# You need to run these commands in your terminal where your Streamlit environment is active:
# pip install xarray
# pip install netCDF4
# pip install h5netcdf # Still useful for NetCDF4, which is HDF5-based
# pip install pyhdf   # CRITICAL for reading HDF4 files (e.g., many MODIS products)
# pip install pandas
# pip install matplotlib
# pip install earthaccess
# pip install seaborn # Required for seaborn plotting which is included in the template analysis part

# --- Data Processing Function (from your template, adapted for Streamlit) ---
def process_satellite_data_to_florida_df(file_path):
    """
    Reads a satellite data file (.nc, .hdf, .he5), extracts chlorophyll-a,
    flattens it, filters for the Florida region, and performs feature engineering.
    Returns a Pandas DataFrame or None on failure.
    """
    st.info(f"Attempting to process satellite file: {os.path.basename(file_path)}")
    df = None # Initialize df to None

    try:
        # Open the dataset using xarray
        ds = xr.open_dataset(file_path)

        # Identify the chlorophyll variable, checking for common naming conventions
        possible_chlor_names = ['chlor_a', 'chlorophyll_a', 'CHL', 'chl_a', 'chlor_oci', 'Rrs_488'] # Added Rrs_488 as a common variable
        chlor_var = None
        for name in possible_chlor_names:
            if name in ds.data_vars:
                chlor_var = name
                break

        # Exit if no chlorophyll variable is found
        if chlor_var is None:
            st.warning(f"Could not find a common chlorophyll variable in {os.path.basename(file_path)}. Available variables: {list(ds.data_vars.keys())}")
            ds.close()
            return None

        st.info(f"Using chlorophyll variable: '{chlor_var}' from {os.path.basename(file_path)}")

        # Get the chlorophyll data and coordinate variables (latitude, longitude)
        chlor_data = ds[chlor_var]

        # Robustly get latitude and longitude
        lat = None
        lon = None
        for dim_name in ['lat', 'latitude', 'YDim', 'y']: # Common latitude dimension names
            if dim_name in ds.coords:
                lat = ds.coords[dim_name]
                break
            elif dim_name in ds.data_vars: # Sometimes coords are data_vars
                lat = ds.data_vars[dim_name]
                break

        for dim_name in ['lon', 'longitude', 'XDim', 'x']: # Common longitude dimension names
            if dim_name in ds.coords:
                lon = ds.coords[dim_name]
                break
            elif dim_name in ds.data_vars: # Sometimes coords are data_vars
                lon = ds.data_vars[dim_name]
                break

        if lat is None or lon is None:
            st.error(f"Could not find valid latitude ('lat', 'latitude', 'YDim', 'y') or longitude ('lon', 'longitude', 'XDim', 'x') coordinates in {os.path.basename(file_path)}.")
            ds.close()
            return None
        
        # Handle time dimension: select the first time slice if present
        # This makes sure we are dealing with a 2D (lat, lon) array for flattening
        if 'time' in chlor_data.dims:
            st.info(f"Selecting first time slice from variable '{chlor_var}'.")
            chlor_data = chlor_data.isel(time=0)
        elif 'Time' in chlor_data.dims: # Some datasets use 'Time'
            st.info(f"Selecting first time slice from variable '{chlor_var}'.")
            chlor_data = chlor_data.isel(Time=0)


        st.info("Converting data to table format (Pandas DataFrame)...")

        # Get coordinate and chlorophyll values as NumPy arrays
        # Ensure lat, lon are aligned with chlor_data dimensions if they are not simple 1D
        if len(lat.shape) == 1 and len(lon.shape) == 1:
            # Typical case where lat and lon are 1D coordinate arrays
            lat_vals = lat.values
            lon_vals = lon.values
            lon_grid, lat_grid = np.meshgrid(lon_vals, lat_vals)
        elif len(lat.shape) == 2 and len(lon.shape) == 2 and lat.shape == chlor_data.shape:
            # Case where lat/lon are 2D arrays, directly matching the data shape (e.g., some imagery products)
            lat_grid = lat.values
            lon_grid = lon.values
        else:
            # Fallback for complex cases, attempt to get coords from chlor_data itself if possible
            # This handles cases where lat/lon are not explicit variables but derived from dims
            try:
                # This assumes chlor_data has 'lat' and 'lon' dimensions
                # If these are just dimension names, not data variables, xarray handles it.
                # If they are data variables, need to access them explicitly.
                if 'latitude' in ds.data_vars and 'longitude' in ds.data_vars:
                    lat_grid = ds['latitude'].values
                    lon_grid = ds['longitude'].values
                    if len(lat_grid.shape) == 1: # If they are 1D, make a grid
                         lon_grid, lat_grid = np.meshgrid(lon_grid, lat_grid)
                else: # Default to the simpler method if no specific 2D lat/lon data vars are found
                    lat_vals = chlor_data[lat.name].values # Use the name of the latitude coord
                    lon_vals = chlor_data[lon.name].values # Use the name of the longitude coord
                    lon_grid, lat_grid = np.meshgrid(lon_vals, lat_vals)

            except Exception as e:
                st.error(f"Complex coordinate handling failed for {os.path.basename(file_path)}: {e}")
                ds.close()
                return None


        chlor_vals = chlor_data.values

        # Ensure all arrays are flattened correctly
        lat_flat = lat_grid.flatten()
        lon_flat = lon_grid.flatten()
        chlor_flat = chlor_vals.flatten()

        # Create a boolean mask to remove invalid data points (NaNs, negative chlorophyll)
        # Also ensure lat/lon are valid, as some datasets might have invalid values
        valid_mask = (~np.isnan(chlor_flat)) & (chlor_flat >= 0) & (~np.isnan(lat_flat)) & (~np.isnan(lon_flat)) & \
                     (lat_flat >= -90) & (lat_flat <= 90) & (lon_flat >= -180) & (lon_flat <= 180)


        # Create the initial DataFrame with only valid data
        df = pd.DataFrame({
            'Latitude': lat_flat[valid_mask],
            'Longitude': lon_flat[valid_mask],
            'Chlorophyll_mg_m3': chlor_flat[valid_mask]
        })
        
        # Close the xarray dataset as we have extracted necessary data
        ds.close()

        if df.empty:
            st.warning(f"No valid data points found in {os.path.basename(file_path)} after initial processing and masking.")
            return None


        st.info(f"Original valid data points from {os.path.basename(file_path)}: {len(df):,}")

        # --- Florida Region Filtering ---
        # Define the bounding box for the Florida region (including surrounding waters)
        florida_bounds = {
            'lat_min': 24.0,   # Southern tip of Florida Keys
            'lat_max': 31.0,   # Northern Florida/Georgia border
            'lon_min': -87.5,  # Western Gulf of Mexico (Pensacola area)
            'lon_max': -79.5   # Eastern Atlantic (Miami/Keys area)
        }

        # Apply the geographic filter to the DataFrame
        florida_mask = (
            (df['Latitude'] >= florida_bounds['lat_min']) &
            (df['Latitude'] <= florida_bounds['lat_max']) &
            (df['Longitude'] >= florida_bounds['lon_min']) &
            (df['Longitude'] <= florida_bounds['lon_max']))
        df = df[florida_mask].copy() # Use .copy() to avoid SettingWithCopyWarning

        st.info(f"Data points in Florida region: {len(df):,}")

        # Check if any data remains after filtering
        if len(df) == 0:
            st.warning(f"No data found in Florida region for {os.path.basename(file_path)} after filtering. The satellite data may not cover this specific area, or the bounds might need adjustment.")
            return None

        st.success(f"Successfully filtered {os.path.basename(file_path)} to Florida waters.")

        # --- Feature Engineering / Derived Columns ---

        # Add a log10 transformed chlorophyll column for better distribution analysis
        df['Chlorophyll_log10'] = np.log10(df['Chlorophyll_mg_m3'].replace(0, np.nan).fillna(method='ffill').fillna(method='bfill')) # Handle log of zero/negative, then interpolate and backfill

        # Categorize chlorophyll levels into trophic states
        df['Trophic_Level'] = pd.cut(df['Chlorophyll_mg_m3'],
                                     bins=[0, 0.1, 0.3, 1.0, 3.0, float('inf')],
                                     labels=['Oligotrophic', 'Ultra-oligotrophic', 'Mesotrophic', 'Eutrophic', 'Hypertrophic'],
                                     right=False,
                                     include_lowest=True) # Ensure 0 is included if it somehow passes filter


        # Classify water bodies as Gulf of Mexico or Atlantic Ocean based on longitude
        df['Water_Body'] = 'Unknown'
        gulf_mask = df['Longitude'] < -82.5
        df.loc[gulf_mask, 'Water_Body'] = 'Gulf_of_Mexico'
        atlantic_mask = df['Longitude'] >= -82.5
        df.loc[atlantic_mask, 'Water_Body'] = 'Atlantic_Ocean'

        # Further classify into more specific Florida regions
        df['Florida_Region'] = 'Other'
        # Panhandle/Big Bend (Gulf side, northern)
        panhandle_mask = (df['Longitude'] < -82.5) & (df['Latitude'] > 29.0)
        df.loc[panhandle_mask, 'Florida_Region'] = 'Panhandle_Gulf'
        # Nature Coast/Tampa Bay area (Gulf side, central)
        nature_coast_mask = (df['Longitude'] < -82.5) & (df['Latitude'] >= 27.5) & (df['Latitude'] <= 29.0)
        df.loc[nature_coast_mask, 'Florida_Region'] = 'Nature_Coast_Gulf'
        # Southwest Florida (Gulf side, southern)
        southwest_mask = (df['Longitude'] < -82.5) & (df['Latitude'] < 27.5)
        df.loc[southwest_mask, 'Florida_Region'] = 'Southwest_Gulf'
        # Northeast Florida (Atlantic side, northern)
        northeast_mask = (df['Longitude'] >= -82.5) & (df['Latitude'] > 28.5)
        df.loc[northeast_mask, 'Florida_Region'] = 'Northeast_Atlantic'
        # Central East Coast (Atlantic side, central)
        central_east_mask = (df['Longitude'] >= -82.5) & (df['Latitude'] >= 26.0) & (df['Latitude'] <= 28.5)
        df.loc[central_east_mask, 'Florida_Region'] = 'Central_East_Atlantic'
        # Southeast Florida/Keys (Atlantic side, southern)
        southeast_mask = (df['Longitude'] >= -82.5) & (df['Latitude'] < 26.0)
        df.loc[southeast_mask, 'Florida_Region'] = 'Southeast_Keys_Atlantic'

        # Approximate distance from a central Florida coastline point (simplified calculation)
        df['Distance_from_Shore_km'] = np.sqrt(
            ((df['Latitude'] - 27.5) * 111)**2 + # 1 degree lat ~ 111 km
            ((df['Longitude'] + 82.5) * 111 * np.cos(np.radians(df['Latitude'])))**2 # 1 degree lon varies with latitude
        )
        # Categorize data points based on approximate distance from shore
        df['Shore_Zone'] = pd.cut(df['Distance_from_Shore_km'],
                                 bins=[0, 20, 50, 200, float('inf')],
                                 labels=['Nearshore_0-20km', 'Coastal_20-50km', 'Shelf_50-200km', 'Offshore_200km+'],
                                 right=False,
                                 include_lowest=True)

        return df

    except MemoryError as me:
        st.error(f"Memory Error processing {os.path.basename(file_path)}: {me}. This file is too large to process in memory.")
        return None
    except Exception as e:
        st.error(f"Failed to process satellite file {os.path.basename(file_path)}: {e}")
        return None
    finally:
        # Ensure dataset is closed if it was opened
        if 'ds' in locals() and ds is not None:
            ds.close()


def home_page():
    st.set_page_config(page_title="EDEN AI Home", page_icon=":earth_africa:", layout="wide")
    # Universal CSS for black text readability
    st.markdown("""
        <style>
        body, .stApp, .stMarkdown, .stText, .stHeader, .stSubheader, .stTitle, .stCaption, .stDataFrame, .stTable, .stAlert, .stRadio, .stSelectbox, .stTextInput, .stNumberInput {
            color: #111 !important;
        }
        /* Force all labels, radio/select options, and expander headers to black for visibility */
        label, .stRadio>label, .stSelectbox>label, .stRadio div[role="radiogroup"] label, .stSelectbox div[role="listbox"] span, .streamlit-expanderHeader {
            color: #111 !important;
        }
        /* Also force text inside input fields and buttons to black */
        input, textarea {
            color: #111 !important;
        }
        /* Restore original color for file uploader button */
        .stFileUploader > label, .stButton > button {
            color: initial !important;
        }
        </style>
    """, unsafe_allow_html=True)
    # Navigation bar removed as requested

    # --- Custom CSS for Aquatic Theme ---
    st.markdown(
        """
        <style>
        /* General background and text color */
        .stApp {
            background-color: #e0f2f7; /* Light blue/cyan for a fresh, aquatic feel */
            color: #1a1a1a; /* Dark text for readability */
        }

        /* Titles and headers */
        h1, h2, h3, h4, h5, h6 {
            color: #004d40; /* Dark teal/green for a professional, environmental feel */
        }

        /* Streamlit widgets - buttons, selectboxes, radio buttons */
        .stButton>button {
            background-color: #00796b; /* Greenish-blue for buttons */
            color: white;
            border-radius: 5px;
            border: 1px solid #004d40;
        }
        .stButton>button:hover {
            background-color: #004d40; /* Darker on hover */
            color: white;
        }

        .stRadio>label, .stSelectbox>label {
            color: #004d40; /* Match label color to headers */
            font-weight: bold;
        }
        .stRadio div[role="radiogroup"] label {
            color: #1a1a1a; /* Ensure radio options are readable */
        }

        /* File uploader and text input */
        .stFileUploader label {
            color: #004d40;
            font-weight: bold;
        }
        .stTextInput>div>div>input, .stNumberInput>div>div>input {
            background-color: #f0f8ff; /* Very light blue for input fields */
            color: #1a1a1a;
            border: 1px solid #00796b;
            border-radius: 5px;
        }
        
        /* Information and success messages */
        .stAlert {
            background-color: #b2dfdb; /* Lighter teal for info/success */
            color: #004d40;
            border-radius: 5px;
        }
        .stAlert.stSuccess {
            background-color: #c8e6c9; /* Light green for success */
            color: #2e7d32;
        }
        .stAlert.stWarning {
            background-color: #ffe0b2; /* Light orange for warning */
            color: #ef6c00;
        }
        .stAlert.stError {
            background-color: #ffcdd2; /* Light red for error */
            color: #c62828;
        }

        /* Dataframes */
        .stDataFrame {
            background-color: #ffffff; /* White background for dataframes */
            color: #1a1a1a;
            border-radius: 5px;
            box-shadow: 2px 2px 5px rgba(0,0,0,0.1); /* Subtle shadow */
        }
        .css-1r6dm7m { /* Target for dataframe header cells - may vary with Streamlit versions */
            background-color: #e0f2f7 !important; /* Lighter blue header */
        }
        .css-vk32pt { /* Target for dataframe header text - may vary with Streamlit versions */
             color: #004d40 !important; /* Dark teal header text */
             font-weight: bold;
        }

        /* Expander (if used) */
        .streamlit-expanderHeader {
            background-color: #e0f2f7;
            color: #004d40;
            border-radius: 5px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    with st.expander("Quick Start Guide", expanded=True):
        st.markdown("""
        1. **Upload Data:** Use the sidebar to upload your aquatic dataset (CSV, JSON, XLSX, NC, HDF) or import satellite data via Earthdata API.
        2. **Preview & Clean:** View, rename, and clean your data directly in the app.
        3. **Visualize:** Explore geospatial maps and scatter plots of your data.
        4. **Analyze:** Use the Predict page to train models and forecast marine events.
        5. **Download:** Export processed data and reports for further use.

        ---
        **We value your feedback!** Please email your suggestions or issues to [jarellfelix@alarextech.com](mailto:jarellfelix@alarextech.com).
        """)
    st.title("Welcome to EDEN AI - Aquatic Environmental Intelligence")
    st.write("This application provides comprehensive intelligence for aquatic environments, combining time-series and satellite data.")

    st.write("You can upload your data or select options to proceed.")

    data_option = st.radio(
        "How would you like to import your data?",
        ("Upload file (CSV, JSON, XLSX, NC, HDF)", "Import satellite data via Earthdata API")
    )

    # Initialize st.session_state['main_df'] if it doesn't exist
    if 'main_df' not in st.session_state:
        st.session_state['main_df'] = None 
    # Also initialize a temporary storage for xarray dataset if .nc is uploaded
    if 'uploaded_nc_ds' not in st.session_state:
        st.session_state['uploaded_nc_ds'] = None

    if data_option == "Upload file (CSV, JSON, XLSX, NC, HDF)":
        uploaded_file = st.file_uploader("Upload your data file (CSV, JSON, XLSX, NC, HDF)", type=["csv", "json", "xlsx", "nc", "hdf", "he5"])
        if uploaded_file is not None:
            st.success("File uploaded successfully!")
            st.session_state['main_df'] = None # Clear previous df on new upload
            st.session_state['uploaded_nc_ds'] = None # Clear previous xarray dataset

            # Determine file type and read data
            if uploaded_file.name.endswith('.csv'):
                st.session_state['main_df'] = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.json'):
                data = json.load(uploaded_file)
                st.session_state['main_df'] = pd.DataFrame(data)
            elif uploaded_file.name.endswith('.xlsx'):
                st.session_state['main_df'] = pd.read_excel(uploaded_file)
            elif uploaded_file.name.endswith('.nc') or uploaded_file.name.endswith('.hdf') or uploaded_file.name.endswith('.he5'):
                # Handle .nc, .hdf, .he5 files using the enhanced processing function
                temp_path = None
                try:
                    # Save the uploaded file to a temporary location
                    suffix = os.path.splitext(uploaded_file.name)[1]
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        temp_path = tmp_file.name
                    
                    st.info(f"Processing '{os.path.basename(uploaded_file.name)}' for Florida chlorophyll data...")
                    
                    processed_df = process_satellite_data_to_florida_df(temp_path)
                    
                    if processed_df is not None and not processed_df.empty:
                        st.session_state['main_df'] = processed_df
                        st.success(f"Successfully processed {os.path.basename(uploaded_file.name)} and extracted Florida chlorophyll data.")
                        st.dataframe(processed_df.head()) # Show preview

                        # Offer download of the processed Florida data
                        base_filename = os.path.basename(uploaded_file.name).split('.')[0]
                        processed_csv = processed_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label=f"Download Processed Florida Chlorophyll Data as CSV (from {base_filename})",
                            data=processed_csv,
                            file_name=f"florida_chlorophyll_data_{base_filename}.csv",
                            mime="text/csv",
                            key=f"download_processed_florida_csv_{base_filename}"
                        )
                        processed_json = processed_df.to_json(orient='records', indent=4).encode('utf-8')
                        st.download_button(
                            label=f"Download Processed Florida Chlorophyll Data as JSON (from {base_filename})",
                            data=processed_json,
                            file_name=f"florida_chlorophyll_data_{base_filename}.json",
                            mime="application/json",
                            key=f"download_processed_florida_json_{base_filename}"
                        )

                    else:
                        st.warning(f"No usable Florida chlorophyll data could be extracted from {os.path.basename(uploaded_file.name)}.")
                        st.session_state['main_df'] = None # Ensure main_df is clear on failure

                    # Offer to display full dataset (if it's an .nc and not too big) OR just let them download original
                    if uploaded_file.name.endswith('.nc'):
                        try:
                            ds_nc_uploaded = xr.open_dataset(temp_path)
                            st.session_state['uploaded_nc_ds'] = ds_nc_uploaded
                            st.subheader("Original NetCDF Dataset Overview (Full Structure)")
                            st.write(ds_nc_uploaded)
                            
                            st.markdown("---")
                            st.subheader("Convert Entire NetCDF Dataset to Pandas DataFrame (Optional - Memory Intensive)")
                            st.warning("**Warning:** Converting the *entire* NetCDF file (all variables and dimensions) to a flat Pandas DataFrame can be **very memory and CPU intensive** for large datasets. Click this button **only if you have sufficient RAM** and explicitly need the full dataset for comprehensive tabular analysis.")
                            
                            if st.button("Convert Full Dataset to DataFrame for Analysis", key="convert_full_ds_uploaded"):
                                with st.spinner("Converting entire NetCDF dataset to DataFrame... This might take a while."):
                                    try:
                                        full_dataset_df = st.session_state['uploaded_nc_ds'].to_dataframe().reset_index()
                                        st.success("Full dataset conversion complete! The DataFrame is now available for analysis.")
                                        st.write("Preview of full dataset converted to DataFrame:")
                                        st.dataframe(full_dataset_df.head())

                                        full_dataset_csv = full_dataset_df.to_csv(index=False).encode('utf-8')
                                        st.download_button(
                                            label="Download Full Dataset as CSV",
                                            data=full_dataset_csv,
                                            file_name="full_dataset_from_nc.csv",
                                            mime="text/csv",
                                            key="download_full_ds_uploaded"
                                        )
                                        # Only update main_df if user explicitly asks for full conversion
                                        st.session_state['main_df'] = full_dataset_df 
                                    except MemoryError as me:
                                        st.error(f"Memory Error: Could not convert the full dataset to DataFrame. Your system ran out of memory. Error: {me}")
                                        st.info("Consider using smaller files or extracting only specific variables if this problem persists.")
                                    except Exception as e:
                                        st.error(f"Error during full dataset conversion: {e}")
                                # Close the full dataset if it was opened for a temporary display
                                if 'ds_nc_uploaded' in locals() and ds_nc_uploaded is not None:
                                    ds_nc_uploaded.close()
                            else:
                                st.info("The full dataset DataFrame is not yet available for general map/scatter plots unless you click the button above.")

                        except Exception as e:
                            st.error(f"Error opening uploaded NetCDF file for overview: {e}")
                            if 'ds_nc_uploaded' in locals() and ds_nc_uploaded is not None:
                                ds_nc_uploaded.close()

                except Exception as e:
                    st.error(f"Error handling uploaded satellite file: {e}")
                    st.session_state['main_df'] = None
                finally:
                    if temp_path and os.path.exists(temp_path):
                        os.unlink(temp_path) # Clean up the temporary file

            else: # Fallback for unknown file types or issues
                st.error("Unsupported file type or error reading file.")
                st.session_state['main_df'] = None

    elif data_option == "Import satellite data via Earthdata API":
        st.info("This option allows you to fetch satellite data directly from NASA Earthdata.")
        st.warning("You need an Earthdata Login account for this. If you don't have one, register at urs.earthdata.nasa.gov")

        # --- Earthdata Login ---
        st.subheader("Earthdata Login")
        if 'auth_status' not in st.session_state:
            st.session_state.auth_status = False
        
        auth = None # Initialize auth outside the if/else for consistent scope

        if not st.session_state.auth_status:
            with st.expander("Enter Earthdata Credentials (Optional, for restricted datasets)"):
                earthdata_username = st.text_input("Earthdata Username", key="ed_user")
                earthdata_password = st.text_input("Earthdata Password", type="password", key="ed_pass")
                if st.button("Log In to Earthdata"):
                    if earthdata_username and earthdata_password:
                        try:
                            auth = earthaccess.login(strategy="reauth", persist=True,
                                                    username=earthdata_username, password=earthdata_password)
                            st.success("Authenticated with Earthdata. You might be prompted in your browser to authorize.")
                            st.session_state.auth_status = True
                        except Exception as e:
                            st.error(f"Earthdata login failed: {e}. Please check your credentials or .netrc file.")
                            st.session_state.auth_status = False
                    else:
                        st.warning("Please enter both username and password.")
                else:
                    # Try silent login if not explicitly logged in via button
                    try:
                        auth = earthaccess.login(strategy="environment") # Tries .netrc, env vars etc.
                        if auth: # Check if authentication was successful
                            st.info("Attempting silent authentication with existing Earthdata credentials.")
                            st.session_state.auth_status = True
                        else:
                            st.warning("Not authenticated. Some datasets may not be accessible. Enter credentials above or proceed for public data.")
                            st.session_state.auth_status = False 
                    except Exception:
                        st.warning("Not authenticated. Some datasets may not be accessible. Enter credentials above or proceed for public data.")
                        st.session_state.auth_status = False
        else:
            st.success("You are logged in to Earthdata.")
            try:
                auth = earthaccess.login(strategy="environment") # Re-get auth object for current session
            except Exception as e:
                st.error(f"Could not re-establish Earthdata session: {e}. You may need to log in again.")
                st.session_state.auth_status = False

        # --- Dynamic Dataset Discovery Section ---
        st.subheader("1. Find Satellite Datasets")
        dataset_keyword = st.text_input("Enter keywords to search for datasets (e.g., 'chlorophyll', 'sea surface temperature')", key="dataset_keyword")
        dataset_instrument = st.text_input("Enter instrument (e.g., 'MODIS', 'VIIRS', 'ICESAT-2')", key="dataset_instrument")

        if st.button("Search Available Datasets", key="search_datasets_button"):
            # Clear previous results and selection when a new search is initiated
            if 'found_datasets' in st.session_state:
                del st.session_state.found_datasets
            if 'selected_product_info' in st.session_state:
                del st.session_state.selected_product_info

            if not dataset_keyword and not dataset_instrument:
                st.warning("Please enter at least a keyword or an instrument to search for datasets.")
            else:
                with st.spinner("Searching Earthdata for datasets..."):
                    try:
                        search_params = {}
                        if dataset_keyword:
                            search_params['keyword'] = dataset_keyword
                        if dataset_instrument:
                            search_params['instrument'] = dataset_instrument
                        
                        found_datasets = earthaccess.search_datasets(**search_params)
                        
                        if found_datasets:
                            st.session_state.found_datasets = found_datasets
                            st.success(f"Found {len(found_datasets)} datasets matching your criteria.")
                        else:
                            st.warning("No datasets found for the specified criteria. Try different keywords or instruments.")
                            st.session_state.found_datasets = [] # Clear previous results
                            st.session_state.selected_product_info = None # Clear previous selection

                    except Exception as e:
                        st.error(f"Error searching for datasets: {e}")
                        st.session_state.found_datasets = []
                        st.session_state.selected_product_info = None

        if 'found_datasets' in st.session_state and st.session_state.found_datasets:
            st.subheader("2. Select a Dataset")
            
            # Prepare options for radio button, showing index, title, and short name
            dataset_options_display = [
                f"{i+1}: {ds['umm']['EntryTitle']} (Short Name: {ds['umm']['ShortName']})"
                for i, ds in enumerate(st.session_state.found_datasets)
            ]
            
            # Use st.radio for selection, default to the first one if not already selected
            if 'selected_dataset_radio_index' not in st.session_state:
                st.session_state.selected_dataset_radio_index = 0 # Default to first option

            selected_dataset_index = st.radio(
                "Choose a dataset from the search results:",
                options=range(len(dataset_options_display)),
                format_func=lambda x: dataset_options_display[x],
                key="selected_dataset_radio"
            )

            # Update session_state.selected_product_info whenever selection changes
            if selected_dataset_index is not None and selected_dataset_index < len(st.session_state.found_datasets):
                selected_dataset_info = st.session_state.found_datasets[selected_dataset_index]
                st.session_state.selected_product_info = {
                    'short_name': selected_dataset_info['umm']['ShortName'],
                    'concept_id': selected_dataset_info['meta']['concept-id'],
                    'entry_title': selected_dataset_info['umm']['EntryTitle']
                }
                st.info(f"You have selected: **{st.session_state.selected_product_info['entry_title']}** (Short Name: `{st.session_state.selected_product_info['short_name']}`)")
            elif selected_dataset_index is None: # If no option is selected (e.g. after search with no results)
                st.info("Please select a dataset to proceed.")
                st.session_state.selected_product_info = None # Ensure it's cleared if nothing is selected


        # --- Granule Search and Download Section (Conditional) ---
        if 'selected_product_info' in st.session_state and st.session_state.selected_product_info:
            st.subheader(f"3. Fetch Data for: {st.session_state.selected_product_info['entry_title']}")
            
            st.markdown("**Spatial and Temporal Filters:**")
            col1, col2 = st.columns(2)
            with col1:
                # Set default date to a recent month/year for demonstration
                # Use current date as reference for sensible defaults
                today = date.today()
                # Default start date ~1 month ago
                default_start_date = today - timedelta(days=30) 
                start_date_api = st.date_input("Start Date", value=default_start_date, key="start_date_api")
            with col2:
                # Default end date to yesterday, as 'today's' data might not be fully processed/available
                default_end_date = today - timedelta(days=1)
                end_date_api = st.date_input("End Date", value=default_end_date, key="end_date_api")
            
            st.write("Enter Bounding Box (W, S, E, N):")
            col_bbox1, col_bbox2, col_bbox3, col_bbox4 = st.columns(4)
            with col_bbox1:
                bbox_w = st.number_input("West Longitude", value=-87.5, format="%.2f", key="bbox_w") # Adjusted to match Florida bounds
            with col_bbox2:
                bbox_s = st.number_input("South Latitude", value=24.0, format="%.2f", key="bbox_s") # Adjusted to match Florida bounds
            with col_bbox3:
                bbox_e = st.number_input("East Longitude", value=-79.5, format="%.2f", key="bbox_e") # Adjusted to match Florida bounds
            with col_bbox4:
                bbox_n = st.number_input("North Latitude", value=31.0, format="%.2f", key="bbox_n") # Adjusted to match Florida bounds
            
            bounding_box = (bbox_w, bbox_s, bbox_e, bbox_n)

            # Add troubleshooting tips for the user
            st.info(
                """
                **Troubleshooting Tip:** If "No data found" appears:
                - **Adjust Dates:** Try a more recent date range, or a wider range (e.g., several months).
                - **Adjust Bounding Box:** Try a slightly larger or different oceanic area, or use the provided default Florida bounds.
                - **Verify Product Availability:** Use the official NASA Earthdata Search Portal
                  (search.earthdata.nasa.gov) to manually verify if data for your selected
                  product, dates, and location is available. This can help confirm if the issue
                  is with your parameters or data existence.
                """
            )

            if st.button("Fetch Satellite Data", key="fetch_granules_button_final"):
                if not st.session_state.auth_status:
                    st.warning("You are not authenticated with Earthdata Login. Some restricted datasets might not be accessible. Attempting to proceed with public access.")
                
                st.info(f"Searching for {st.session_state.selected_product_info['short_name']} data for {start_date_api} to {end_date_api} in bounding box {bounding_box}...")
                
                temp_download_dir = None # Initialize outside try block
                try:
                    # Search for data granules
                    results = earthaccess.search_data(
                        short_name=st.session_state.selected_product_info['short_name'],
                        temporal=(str(start_date_api), str(end_date_api)),
                        bounding_box=bounding_box,
                        count=10 # Limit results to avoid too many files for demonstration
                    )

                    if not results:
                        st.warning("No data found for the specified criteria. Try adjusting dates or bounding box as suggested above.")
                        st.session_state['main_df'] = None # Clear df if no data found
                    else:
                        st.success(f"Found {len(results)} data granules. Attempting to download...")
                        
                        # Create a temporary directory for downloads that will be cleaned up later
                        temp_download_dir = tempfile.mkdtemp()
                        
                        with st.spinner("Downloading datasets... This might take a moment depending on data size and connection."):
                            downloaded_files = earthaccess.download(results, local_path=temp_download_dir)
                            
                            processed_florida_dfs = [] # List to store processed DataFrames from all files
                            
                            st.subheader("Downloaded Satellite Files:")
                            if not downloaded_files:
                                st.info("No files were successfully downloaded.")
                                st.session_state['main_df'] = None # Clear df if no files downloaded
                            
                            for i, file_path in enumerate(downloaded_files):
                                file_name = os.path.basename(file_path)
                                # Ensure we have a valid result object for metadata
                                if i < len(results):
                                    original_url = results[i].data_links()[0] # Get the original Earthdata URL for the granule
                                else:
                                    original_url = "N/A (Could not retrieve original URL)"

                                st.markdown(f"**File {i+1}:** `{file_name}`")
                                st.markdown(f"Local Path: `{file_path}`")
                                st.markdown(f"Original Earthdata URL: [Link]({original_url})")

                                # Provide direct download button for the original downloaded file
                                try:
                                    with open(file_path, "rb") as f_orig:
                                        st.download_button(
                                            label=f"Download Original '{file_name}'",
                                            data=f_orig.read(),
                                            file_name=file_name,
                                            mime="application/octet-stream", # Generic binary MIME type
                                            key=f"download_original_{i}"
                                        )
                                except Exception as e_orig_download:
                                    st.warning(f"Could not prepare original file '{file_name}' for direct download: {e_orig_download}")

                                # Process the downloaded file using the dedicated function
                                processed_df_item = process_satellite_data_to_florida_df(file_path)
                                
                                if processed_df_item is not None and not processed_df_item.empty:
                                    processed_florida_dfs.append(processed_df_item)
                                    st.success(f"File {file_name}: Florida chlorophyll data successfully extracted and processed.")
                                    st.dataframe(processed_df_item.head()) # Show preview
                                    
                                    # Offer download for this specific processed file's data
                                    base_filename_item = os.path.basename(file_name).split('.')[0]
                                    processed_csv_item = processed_df_item.to_csv(index=False).encode('utf-8')
                                    st.download_button(
                                        label=f"Download Processed Florida Chlorophyll Data as CSV (from {base_filename_item})",
                                        data=processed_csv_item,
                                        file_name=f"florida_chlorophyll_data_{base_filename_item}.csv",
                                        mime="text/csv",
                                        key=f"download_processed_florida_csv_item_{i}"
                                    )
                                else:
                                    st.warning(f"File {file_name}: No usable Florida chlorophyll data extracted or processing failed.")
                                st.markdown("---") # Separator for clarity
                                
                            # Concatenate all successfully processed Florida DataFrames
                            if processed_florida_dfs:
                                try:
                                    st.session_state['main_df'] = pd.concat(processed_florida_dfs, ignore_index=True)
                                    st.success("All successfully extracted Florida chlorophyll datasets have been concatenated into a single DataFrame for further analysis.")
                                    st.info("The combined DataFrame for general analysis (map/scatter plots) now contains only Florida-specific chlorophyll data.")
                                    st.dataframe(st.session_state['main_df'].head()) # Show combined preview
                                except Exception as concat_err:
                                    st.warning(f"Could not concatenate all Florida chlorophyll DataFrames from fetched files: {concat_err}. No combined DataFrame for general visualization from fetched data.")
                                    st.session_state['main_df'] = None # Ensure df is None if concatenation fails
                            else:
                                st.info("No Florida chlorophyll data was successfully extracted and processed into DataFrames from the downloaded data for general visualization.")
                                st.session_state['main_df'] = None

                except Exception as e: # This try-except block covers earthaccess.search_data and the subsequent processing
                    st.error(f"An error occurred during satellite data retrieval or processing: {e}")
                    st.session_state['main_df'] = None
                finally:
                    # Clean up the temporary directory after processing/downloading
                    if temp_download_dir and os.path.exists(temp_download_dir):
                        shutil.rmtree(temp_download_dir)
                        st.info(f"Cleaned up temporary download directory: {temp_download_dir}")

        else:
            st.info("Please search for and select a dataset above to fetch satellite data granules.")

    # --- Common Data Processing and Visualization (applies to both upload and fetched data) ---
    current_df = st.session_state['main_df'] # Get the current DataFrame from session state

    # --- Column Renaming Feature ---
    if current_df is not None and not current_df.empty:
        st.markdown("---")
        st.subheader("Rename Columns in Your Dataset")
        col_names = list(current_df.columns)
        rename_dict = {}
        with st.expander("Rename columns (optional)"):
            for col in col_names:
                new_name = st.text_input(f"Rename '{col}'", value=col, key=f"rename_{col}")
                if new_name and new_name != col:
                    rename_dict[col] = new_name
            if st.button("Apply Column Renames", key="apply_renames"):
                current_df.rename(columns=rename_dict, inplace=True)
                st.session_state['main_df'] = current_df
                st.success("Column names updated!")
        st.subheader("Processed Data Preview (Florida Chlorophyll Data)")
        st.dataframe(current_df)

        # Add CSV Download Button for the main processed data (the combined DF from NCs or uploaded data)
        csv_data = current_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Current Processed Florida Chlorophyll Data as CSV",
            data=csv_data,
            file_name="eden_ai_processed_florida_chlorophyll_data.csv",
            mime="text/csv",
            key="download_main_processed_df"
        )
        json_data = current_df.to_json(orient='records', indent=4).encode('utf-8')
        st.download_button(
            label="Download Current Processed Florida Chlorophyll Data as JSON",
            data=json_data,
            file_name="eden_ai_processed_florida_chlorophyll_data.json",
            mime="application/json",
            key="download_main_processed_json"
        )


        st.subheader("Data Information")
        buffer = pd.io.common.StringIO()
        current_df.info(buf=buffer)
        s = buffer.getvalue()
        st.text(s)

        st.subheader("Data Description")
        st.dataframe(current_df.describe())

        # --- Map Visualization ---
        st.subheader("Geospatial Visualization: Map Plots")

        # The processed_satellite_data_to_florida_df function should ensure 'Latitude' and 'Longitude' are present
        # but double-check and provide fallback messages.
        if 'Latitude' in current_df.columns and 'Longitude' in current_df.columns:
            st.success(f"Displaying Florida chlorophyll data on a map.")
            df_for_map = current_df[['Latitude', 'Longitude']].copy()
            df_for_map.rename(columns={'Latitude': 'lat', 'Longitude': 'lon'}, inplace=True)
            df_for_map.dropna(subset=['lat', 'lon'], inplace=True)
            
            # Convert to standard Python float to avoid JSON serialization error
            df_for_map['lat'] = df_for_map['lat'].astype(float)
            df_for_map['lon'] = df_for_map['lon'].astype(float)
            
            if not df_for_map.empty:
                st.map(df_for_map)
            else:
                st.warning("No valid numerical latitude/longitude data points after cleaning for map display.")
        else:
            st.info("No suitable 'Latitude' and 'Longitude' columns detected in the data for map visualization. This should not happen if the satellite data processing was successful.")

        st.subheader("Data Visualization: Scatter Plots")

        # The processed data frame will have 'Chlorophyll_mg_m3', 'Latitude', 'Longitude', 'Distance_from_Shore_km'
        numerical_cols = current_df.select_dtypes(include=['number']).columns.tolist()
        
        if numerical_cols:
            st.write("Select two numerical columns for a scatter plot:")
            
            # Try to pre-select common axes for chlorophyll data
            default_x_index = numerical_cols.index('Distance_from_Shore_km') if 'Distance_from_Shore_km' in numerical_cols else 0
            default_y_index = numerical_cols.index('Chlorophyll_mg_m3') if 'Chlorophyll_mg_m3' in numerical_cols else (1 if len(numerical_cols) > 1 else 0)


            col1 = st.selectbox("Select X-axis", numerical_cols, index=default_x_index, key='scatter_x')
            col2 = st.selectbox("Select Y-axis", numerical_cols, index=default_y_index, key='scatter_y')

            if col1 and col2:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.scatter(current_df[col1], current_df[col2], alpha=0.7, color='#3498db')
                ax.set_title(f"Scatter Plot of {col1} vs {col2}")
                ax.set_xlabel(col1)
                ax.set_ylabel(col2)
                st.pyplot(fig)
            else:
                st.info("Please select two columns to display a scatter plot.")

            st.subheader("Data Cleaning")
            if st.checkbox("Remove Null Values"):
                initial_rows = current_df.shape[0]
                df_cleaned = current_df.dropna()
                rows_removed = initial_rows - df_cleaned.shape[0]
                st.success(f"Removed {rows_removed} rows with null values.")
                st.write("Cleaned Data Preview:")
                st.dataframe(df_cleaned)
                st.session_state['main_df'] = df_cleaned # Update df to the cleaned version in session state
        else:
            st.warning("No numerical columns found in the dataset for scatter plotting.")

        st.write("You can now proceed with your analysis or visualization based on the (potentially cleaned) data.")
    else:
        st.info("Upload a file (CSV, JSON, XLSX for general data, or NC/HDF for satellite data) or fetch satellite data via Earthdata API to get started with analysis and visualization. For satellite files, we will extract and process Florida chlorophyll data.")


if __name__ == "__main__":
    home_page()
