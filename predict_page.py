# predict_page.py
import streamlit as st
import pandas as pd
import numpy as np
# You'll import your machine learning models here once they are built
# from models.your_red_tide_model import predict_red_tide
# from models.your_chlorophyll_model import predict_chlorophyll
# from models.your_karenia_model import predict_karenia_brevis

def predict_page():

    st.set_page_config(page_title="Predict Future Events", page_icon="🔮", layout="wide")
    st.title("🌊 Predict Future Marine Events")
    st.write("Utilize past data and current conditions to forecast red tide events, chlorophyll levels, and Karenia brevis cell counts.")
    st.markdown("---")

    # Access the loaded data from session state
    main_df = st.session_state.get('main_df', pd.DataFrame())

    if main_df.empty:
        st.warning("Please go to the 'Home' page and load data before proceeding with predictions.")
        return

    st.subheader("Select Prediction Target and Train Model")
    prediction_type = st.radio(
        "What would you like to predict?",
        ["Red Tide Events", "Chlorophyll Levels", "Karenia Brevis Cell Counts"],
        horizontal=True,
        key="prediction_type_selector"
    )
    st.markdown("---")

    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score

    # Feature selection with fuzzy keyword matching
    import difflib
    numeric_cols = main_df.select_dtypes(include=['number']).columns.tolist()
    st.write("Select input features for your model:")
    feature_options = [col for col in numeric_cols if col not in ["Red_Tide_Event", "Chlorophyll_mg_m3", "Karenia_Brevis_Cell_Count"]]

    # Keyword input for fuzzy matching (features)
    keyword_input = st.text_input("Enter keywords for features (comma separated, e.g. 'temp, salinity, chloro')")
    selected_features = []
    if keyword_input:
        keywords = [kw.strip().lower() for kw in keyword_input.split(",") if kw.strip()]
        for kw in keywords:
            matches = difflib.get_close_matches(kw, [col.lower() for col in feature_options], n=1, cutoff=0.6)
            if matches:
                orig_col = next((col for col in feature_options if col.lower() == matches[0]), None)
                if orig_col:
                    selected_features.append(orig_col)
    if not selected_features:
        default_features = [col for col in numeric_cols[:-1] if col in feature_options]
        selected_features = default_features
    features = st.multiselect("Features (X)", options=feature_options, default=selected_features)

    # Fuzzy keyword matching for target variable
    target_keywords_map = {
        "Red Tide Events": ["Red_Tide_Event", "red tide", "tide event", "tide"],
        "Chlorophyll Levels": ["Chlorophyll_mg_m3", "chlorophyll", "chloro", "chlorophyll-a"],
        "Karenia Brevis Cell Counts": ["Karenia_Brevis_Cell_Count", "karenia", "brevis", "cell count", "abundance", "cells/L", "Karenia brevis abundance (cells/L)"]
    }
    possible_targets = [col for col in numeric_cols]
    target_keyword_list = target_keywords_map[prediction_type]
    # Try to fuzzy match any of the keywords to the actual columns
    target = None
    for kw in target_keyword_list:
        matches = difflib.get_close_matches(kw.lower(), [col.lower() for col in possible_targets], n=1, cutoff=0.6)
        if matches:
            target = next((col for col in possible_targets if col.lower() == matches[0]), None)
            break
    if not target:
        st.warning(f"No matching target column found for '{prediction_type}'. Please check your upload and keywords.")
        return

    # Model selection
    model_type = st.selectbox("Choose Model Type", ["Random Forest Regressor", "Linear Regression"])
    test_size = st.slider("Test Set Size (%)", min_value=10, max_value=50, value=20, step=5) / 100.0

    if features:
        X = main_df[features]
        y = main_df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        if st.button("Train Model", key="train_model_button"):
            if model_type == "Random Forest Regressor":
                model = RandomForestRegressor(n_estimators=100, random_state=42)
            else:
                model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            st.success("Model trained successfully!")
            st.write(f"**Test MSE:** {mse:.4f}")
            st.write(f"**Test R² Score:** {r2:.4f}")
            st.write("Sample predictions:")
            st.dataframe(pd.DataFrame({"Actual": y_test.values, "Predicted": y_pred}).head())
            # Practical model projection visualization
            import matplotlib.pyplot as plt
            st.markdown("#### Model Projection: Actual vs Predicted")
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.scatter(y_test, y_pred, alpha=0.7, color='#00796b', label='Predicted')
            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Ideal')
            ax.set_xlabel('Actual Values')
            ax.set_ylabel('Predicted Values')
            ax.set_title('Model Projection: Actual vs Predicted')
            ax.legend()
            st.pyplot(fig)

            # Feature vs Target scatter plots
            st.markdown("#### Feature vs Target Visualizations")
            for feat in features:
                fig2, ax2 = plt.subplots(figsize=(7, 4))
                ax2.scatter(X_test[feat], y_test, alpha=0.6, color='#3498db', label='Actual')
                ax2.scatter(X_test[feat], y_pred, alpha=0.6, color='#e67e22', label='Predicted')
                ax2.set_xlabel(feat)
                ax2.set_ylabel(target)
                ax2.set_title(f"{feat} vs {target} (Actual & Predicted)")
                ax2.legend()
                st.pyplot(fig2)
            st.session_state['trained_model'] = model
            st.session_state['model_features'] = features
            st.session_state['model_type'] = model_type
            st.session_state['model_target'] = target

    # Prediction section
    if 'trained_model' in st.session_state and st.session_state['trained_model']:
        st.markdown("---")
        st.subheader(f"Make a Prediction for {prediction_type}")
        input_data = {}
        missing_feats = []
        for feat in st.session_state['model_features']:
            if feat in main_df.columns:
                val = st.number_input(f"{feat}", value=float(main_df[feat].mean()), key=f"predict_input_{feat}")
                input_data[feat] = val
            else:
                missing_feats.append(feat)
        if missing_feats:
            st.warning(f"The following selected features are not present in your uploaded data and will be skipped: {', '.join(missing_feats)}")
        if st.button("Predict with Trained Model", key="predict_with_model"):
            model = st.session_state['trained_model']
            X_new = pd.DataFrame([input_data])
            pred = model.predict(X_new)[0]
            st.success(f"Predicted {prediction_type}: {pred:.4f}")

        # --- Predict future values for uploaded data (if time-based) ---
        st.markdown("---")
        st.subheader(f"Batch Predict {prediction_type} for Uploaded Data")
        if st.button("Run Batch Prediction on Uploaded Data", key="batch_predict"):
            model = st.session_state['trained_model']
            features = st.session_state['model_features']
            if all([feat in main_df.columns for feat in features]):
                batch_X = main_df[features]
                batch_pred = model.predict(batch_X)
                result_df = main_df.copy()
                result_df[f"Predicted_{prediction_type.replace(' ', '_')}"] = batch_pred
                st.success(f"Batch prediction complete! Showing first 20 results:")
                st.dataframe(result_df.head(20))
                csv = result_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label=f"Download Batch Predictions as CSV",
                    data=csv,
                    file_name=f"batch_predictions_{prediction_type.replace(' ', '_').lower()}.csv",
                    mime="text/csv",
                    key="download_batch_predictions"
                )
            else:
                st.error("Not all selected features are present in the uploaded data.")

    st.markdown("---")
    st.write("### Model Training & Configuration")
    st.info("This page uses only your uploaded data for model training and prediction. No simulated values are used.")

    # --- PDF Report Generation ---
    st.markdown("---")
    st.header("Generate Full Analysis Report (PDF)")
    st.write("Draft a report summarizing trends, dataset info, and predictions. Download as PDF.")
    from fpdf import FPDF
    import io

    if st.button("Generate PDF Report", key="generate_pdf_report"):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="EDEN AI Aquatic Intelligence Report", ln=True, align='C')
        pdf.ln(10)

        # Dataset Info
        pdf.set_font("Arial", style='B', size=12)
        pdf.cell(0, 10, "Dataset Information:", ln=True)
        pdf.set_font("Arial", size=10)
        if not main_df.empty:
            pdf.multi_cell(0, 8, f"Rows: {main_df.shape[0]}, Columns: {main_df.shape[1]}")
            pdf.multi_cell(0, 8, f"Columns: {', '.join(main_df.columns)}")
        else:
            pdf.multi_cell(0, 8, "No dataset loaded.")
        pdf.ln(5)

        # Trends (basic stats)
        pdf.set_font("Arial", style='B', size=12)
        pdf.cell(0, 10, "Trends & Statistics:", ln=True)
        pdf.set_font("Arial", size=10)
        try:
            desc = main_df.describe().to_string()
            pdf.set_font("Arial", size=8)
            pdf.multi_cell(0, 6, desc)
        except Exception:
            pdf.multi_cell(0, 8, "No statistics available.")
        pdf.ln(5)

        # Model & Prediction Info
        pdf.set_font("Arial", style='B', size=12)
        pdf.cell(0, 10, "Model & Predictions:", ln=True)
        pdf.set_font("Arial", size=10)
        if 'trained_model' in st.session_state and st.session_state['trained_model']:
            pdf.multi_cell(0, 8, f"Model Type: {st.session_state['model_type']}")
            pdf.multi_cell(0, 8, f"Target: {st.session_state['model_target']}")
            pdf.multi_cell(0, 8, f"Features: {', '.join(st.session_state['model_features'])}")
            # Show last prediction if available
            if 'last_prediction' in st.session_state:
                pdf.multi_cell(0, 8, f"Last Prediction: {st.session_state['last_prediction']}")
        else:
            pdf.multi_cell(0, 8, "No model trained.")
        pdf.ln(5)

        # Batch predictions summary
        if 'batch_predictions' in st.session_state:
            pdf.set_font("Arial", style='B', size=12)
            pdf.cell(0, 10, "Batch Predictions (first 10):", ln=True)
            pdf.set_font("Arial", size=8)
            for i, val in enumerate(st.session_state['batch_predictions'][:10]):
                pdf.cell(0, 6, f"Row {i+1}: {val}", ln=True)

        # Save PDF to buffer
        pdf_bytes = pdf.output(dest='S').encode('latin-1')
        st.download_button(
            label="Download Full Report as PDF",
            data=pdf_bytes,
            file_name="eden_ai_report.pdf",
            mime="application/pdf",
            key="download_pdf_report"
        )

# This __name__ == "__main__" block is not strictly necessary when using st.navigation
# as app.py will manage execution, but it's harmless to keep for local testing.
if __name__ == "__main__":
    predict_page()