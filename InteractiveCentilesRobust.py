import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import norm, probplot
import plotly.graph_objects as go
import plotly.figure_factory as ff
import os

# --- Page Config ---
st.set_page_config(page_title="Centile Calculator", layout="wide")

# --- Helper Functions (Backend Logic) ---
def fit_best_combined_robust_model(data, use_site=False):
    """Fits the mean model using safe internal column names and calculates Pseudo-R2."""
    models = {}
    
    y_mean = np.mean(data['__Y__'])
    sst = np.sum((data['__Y__'] - y_mean)**2)
    
    site_term = " + C(__SITE__)" if use_site else ""
    
    formulas = [
        ('linear', f'__Y__ ~ __X__ + C(__S__){site_term}'),
        ('quadratic', f'__Y__ ~ __X__ + I(__X__**2) + C(__S__){site_term}'),
        ('cubic (s-shape)', f'__Y__ ~ __X__ + I(__X__**2) + I(__X__**3) + C(__S__){site_term}')
    ]

    for m_type, form in formulas:
        try:
            model = smf.rlm(form, data=data, M=sm.robust.norms.TukeyBiweight()).fit()
            y_pred = model.predict(data)
            ssr = np.sum((data['__Y__'] - y_pred)**2)
            r2 = 1 - (ssr / sst) if sst != 0 else 0
            models[m_type] = {'model': model, 'ssr': ssr, 'r2': r2}
        except Exception:
            continue

    if (data['__Y__'] > 0).all():
        try:
            form = f'np.log(__Y__) ~ __X__ + C(__S__){site_term}'
            model = smf.rlm(form, data=data, M=sm.robust.norms.TukeyBiweight()).fit()
            y_pred_exp = np.exp(model.predict(data))
            ssr_exp = np.sum((data['__Y__'] - y_pred_exp)**2)
            r2_exp = 1 - (ssr_exp / sst) if sst != 0 else 0
            models['exponential'] = {'model': model, 'ssr': ssr_exp, 'r2': r2_exp}
        except Exception:
            pass

    if not models:
        return None, None
    
    best_key = min(models, key=lambda k: models[k]['ssr'])
    best_model = {'type': best_key, 'model_obj': models[best_key]['model']}
    
    return best_model, models

def predict_values(model_dict, x_vals, sex_vals, use_site=False, site_vals=None):
    """Predicts values and forces output to a raw numpy array to prevent Pandas index alignment bugs."""
    model = model_dict['model_obj']
    m_type = model_dict['type']
    
    if np.isscalar(sex_vals):
        sex_vals = [sex_vals] * len(x_vals)
        
    pred_dict = {'__X__': x_vals, '__S__': sex_vals}
    
    if use_site:
        if np.isscalar(site_vals):
            site_vals = [site_vals] * len(x_vals)
        pred_dict['__SITE__'] = site_vals
        
    pred_df = pd.DataFrame(pred_dict)
    preds = model.predict(pred_df)
    
    if m_type == 'exponential':
        return np.exp(preds).values
    return preds.values

def calculate_models_and_z(df_input, use_site):
    """Runs the full mean and SD model fitting pipeline and calculates Z-scores."""
    df_work = df_input.copy()
    mean_model, all_models = fit_best_combined_robust_model(df_work, use_site)
    
    if not mean_model:
        return None, None, None, df_work
        
    actual_sites = df_work['__SITE__'].values if use_site else None
    mean_preds = predict_values(mean_model, df_work['__X__'].values, df_work['__S__'].values, use_site, actual_sites)
    df_work['resid_abs'] = np.abs(df_work['__Y__'] - mean_preds) * 1.25
    
    sd_model = None
    # Attempt 1: The Ideal Model
    try:
        site_term = " + C(__SITE__)" if use_site else ""
        sd_mod_obj = smf.rlm(f'resid_abs ~ __X__ + C(__S__){site_term}', data=df_work, M=sm.robust.norms.TukeyBiweight()).fit()
        sd_model = {'type': 'linear', 'model_obj': sd_mod_obj}
    except: pass
    
    # Fallback 1
    if not sd_model:
        try:
            sd_mod_obj = smf.rlm('resid_abs ~ __X__ + C(__S__)', data=df_work, M=sm.robust.norms.TukeyBiweight()).fit()
            sd_model = {'type': 'linear', 'model_obj': sd_mod_obj}
        except: pass

    # Fallback 2
    if not sd_model:
        try:
            sd_mod_obj = smf.rlm('resid_abs ~ __X__', data=df_work, M=sm.robust.norms.TukeyBiweight()).fit()
            sd_model = {'type': 'linear', 'model_obj': sd_mod_obj}
        except: pass

    # Fallback 3
    if not sd_model:
        try:
            sd_mod_obj = smf.ols('resid_abs ~ 1', data=df_work).fit()
            sd_model = {'type': 'linear', 'model_obj': sd_mod_obj}
        except: pass
        
    if sd_model:
        sigma_preds = predict_values(sd_model, df_work['__X__'].values, df_work['__S__'].values, use_site, actual_sites)
        sigma_preds[sigma_preds <= 0] = 1e-6 
        df_work['z_score'] = (df_work['__Y__'] - mean_preds) / sigma_preds
        
    return mean_model, sd_model, all_models, df_work

# --- Main App Interface ---
st.title("Centile Calculator")

# 1. Sidebar - Data Loading
st.sidebar.header("1. Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload Data File", type=["xlsx", "xls", "csv"])

if uploaded_file:
    try:
        if uploaded_file.name.lower().endswith('.csv'):
            df_raw = pd.read_csv(uploaded_file)
        else:
            xl = pd.ExcelFile(uploaded_file)
            sheet = st.sidebar.selectbox("Select Sheet", xl.sheet_names)
            df_raw = xl.parse(sheet)
            
        df_raw.replace(r'^\s*$', np.nan, regex=True, inplace=True)
        
        st.sidebar.header("2. Configuration")
        all_cols = df_raw.columns.tolist()
        
        default_age = all_cols.index('Age') if 'Age' in all_cols else 0
        default_sex = all_cols.index('Sex') if 'Sex' in all_cols else 0
        default_id = all_cols.index('ID') if 'ID' in all_cols else 0
        
        id_col = st.sidebar.selectbox("ID Column (for hover)", all_cols, index=default_id)
        age_col = st.sidebar.selectbox("Age Column", all_cols, index=default_age)
        sex_col = st.sidebar.selectbox("Sex Column", all_cols, index=default_sex)
        site_col = st.sidebar.selectbox("Site Column (Optional)", ["None"] + all_cols)
        
        img_col = st.sidebar.selectbox("Image Path Column (Optional)", ["None"] + all_cols)
        color_col = st.sidebar.selectbox("Color Points By (Optional)", ["None"] + all_cols)
        
        measure_cols = [c for c in all_cols if c not in [age_col, sex_col, id_col, site_col, img_col, color_col]]
        measure_col = st.sidebar.selectbox("Measurement Column", measure_cols)
        
        # Scaling Feature
        scale_data = st.sidebar.checkbox("Scale measurement by another variable?", value=False)
        scale_col = "None"
        if scale_data:
            scale_cols = [c for c in all_cols if c not in [age_col, sex_col, id_col, site_col, img_col, color_col, measure_col]]
            if scale_cols:
                scale_col = st.sidebar.selectbox("Scaling Column", scale_cols)
            else:
                st.sidebar.warning("No columns left to scale by.")
        
        connect_lines = st.sidebar.checkbox("Connect longitudinal data (same ID)", value=False)
        z_thresh = st.sidebar.number_input("Absolute Z-Score Threshold Shading (±)", min_value=0.0, max_value=5.0, value=2.0, step=0.1)
        
        # --- NEW: Advanced Filtering ---
        st.sidebar.header("3. Advanced Filtering")
        pre_filter = st.sidebar.checkbox("Pre-filter Extreme Outliers", value=False, help="Runs a 2-pass background fit to identify and permanently remove extreme outliers before drawing charts.")
        if pre_filter:
            outlier_thresh = st.sidebar.number_input("Drop points with absolute Z-Score >", min_value=2.0, max_value=10.0, value=4.0, step=0.1)

        # --- Age Bounds ---
        df_raw[age_col] = pd.to_numeric(df_raw[age_col], errors='coerce')
        data_min_age = df_raw[age_col].min()
        data_max_age = df_raw[age_col].max()
        
        if pd.isna(data_min_age): data_min_age = 0.0
        if pd.isna(data_max_age): data_max_age = 20.0
        
        floor_age = float(np.floor(data_min_age))
        ceil_age = float(np.ceil(data_max_age))
        if floor_age == ceil_age: ceil_age += 1.0

        min_age, max_age = st.sidebar.slider("Age Range", min_value=floor_age, max_value=ceil_age, value=(floor_age, ceil_age))
        
        # --- Data Scrubbing & Setup ---
        df_raw[measure_col] = pd.to_numeric(df_raw[measure_col], errors='coerce')
        
        cols_to_drop = [age_col, sex_col, measure_col]
        if scale_data and scale_col != "None":
            df_raw[scale_col] = pd.to_numeric(df_raw[scale_col], errors='coerce')
            cols_to_drop.append(scale_col)
        if site_col != "None":
            cols_to_drop.append(site_col)
            
        df = df_raw.dropna(subset=cols_to_drop).copy()
        
        if scale_data and scale_col != "None":
            df = df[df[scale_col] != 0] # Prevent division by zero
            df['__Y__'] = df[measure_col] / df[scale_col]
            display_measure = f"{measure_col} / {scale_col}"
        else:
            df['__Y__'] = df[measure_col]
            display_measure = measure_col
            
        df['__X__'] = df[age_col]
        
        def clean_categorical(val):
            if pd.isna(val): return 'NaN'
            if isinstance(val, float) and val.is_integer(): return str(int(val))
            elif isinstance(val, int): return str(val)
            return str(val).strip()

        df[sex_col] = df[sex_col].apply(clean_categorical)
        df['__S__'] = df[sex_col]
        
        use_site = (site_col != "None")
        if use_site:
            df[site_col] = df[site_col].apply(clean_categorical)
            df['__SITE__'] = df[site_col]
            df = df[~df['__SITE__'].isin(['nan', 'NaN', 'None', ''])]
            sites = sorted(df['__SITE__'].unique())
        
        df[id_col] = df[id_col].astype(str).str.strip()
        df[id_col] = df[id_col].str.replace(r'_(\d+[mM]|[tT]\d+)$', '', regex=True)
        
        df = df[~df['__S__'].isin(['nan', 'NaN', 'None', ''])]
        df = df[(df['__X__'] >= min_age) & (df['__X__'] <= max_age)]
        
        if img_col != "None": df[img_col] = df[img_col].fillna("").astype(str)
        sexes = sorted(df['__S__'].unique())

        if len(df) < 10:
            st.error("Not enough valid data points in selection after filtering out missing values.")
            st.stop()

        # Build Models with Optional 2-Pass Pre-filtering
        with st.spinner(f"Building models for {display_measure}..."):
            
            # --- TWO-PASS PRE-FILTERING ---
            if pre_filter:
                _, sd_pass1, _, df_pass1 = calculate_models_and_z(df, use_site)
                if sd_pass1 and 'z_score' in df_pass1.columns:
                    initial_len = len(df)
                    df = df_pass1[df_pass1['z_score'].abs() <= outlier_thresh].copy()
                    dropped = initial_len - len(df)
                    if dropped > 0:
                        st.sidebar.success(f"🧹 Pre-filter removed {dropped} extreme outliers!")
                    else:
                        st.sidebar.info("🧹 Pre-filter ran, but no outliers exceeded the threshold.")
                else:
                    st.sidebar.warning("Could not pre-filter (Background pass failed). Proceeding with raw data.")
            
            # --- FINAL MODEL FIT ---
            mean_model, sd_model, all_models, df = calculate_models_and_z(df, use_site)
            
            if mean_model:
                st.success(f"Models built successfully! (Selected Mean Model: {mean_model['type']})")
                
                st.write("#### Fitting Data Summary")
                summary_df = df.groupby('__S__').agg(
                    Total_Observations=('__Y__', 'count'),
                    Unique_Subjects=(id_col, 'nunique')
                ).rename_axis("Sex Category").reset_index()
                st.dataframe(summary_df, hide_index=True)
                st.divider()
            else:
                st.error("Could not fit mean model.")
                st.stop()

        # --- Main Layout ---
        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("3. Input New Patient")
            default_age_val = float((min_age + max_age) / 2)
            default_measure_val = float(df[measure_col].mean())
            if pd.isna(default_measure_val): default_measure_val = 100.0
            
            in_age = st.number_input("Age", min_value=float(min_age), max_value=float(max_age), value=default_age_val)
            in_val = st.number_input(f"Measurement Value ({measure_col})", value=default_measure_val)
            in_val_calc = in_val
            
            if scale_data and scale_col != "None":
                default_scale_val = float(df[scale_col].mean())
                if pd.isna(default_scale_val) or default_scale_val == 0: default_scale_val = 1.0
                
                in_scale = st.number_input(f"Scaling Value ({scale_col})", value=default_scale_val)
                if in_scale == 0:
                    st.warning("Scaling value cannot be zero.")
                    in_val_calc = 0
                else:
                    in_val_calc = in_val / in_scale
                st.caption(f"**Scaled Target ({display_measure}):** {in_val_calc:.4f}")
                
            in_sex = st.selectbox("Sex Category", sexes)
            in_site = st.selectbox("Site Category", sites) if use_site else None
            
            if st.button("Calculate", type="primary"):
                mu = predict_values(mean_model, [in_age], [in_sex], use_site, [in_site] if use_site else None)[0]
                if sd_model:
                    sigma = predict_values(sd_model, [in_age], [in_sex], use_site, [in_site] if use_site else None)[0]
                    if sigma <= 0: sigma = 1e-6
                    z = (in_val_calc - mu) / sigma
                    p = norm.cdf(z) * 100
                    st.metric("Z-Score", f"{z:.2f}")
                    st.metric("Percentile", f"{p:.1f}th")
                else:
                    st.error("No SD model available.")

        with col2:
            st.subheader("Reference Charts (Interactive)")
            st.caption("Click any data point to view its associated image and highlight their longitudinal trajectory.")
            if use_site:
                st.caption(f"*(Note: Points and background centile bands are filtered to the selected site: **{in_site}**)*")
            
            named_colors = {'Male': '#1f77b4', 'Female': '#d62728', 'M': '#1f77b4', 'F': '#d62728'}
            fallback_palette = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd', '#ff7f0e']
            cat_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
            
            # --- LOOP 1: Individual Sex Charts ---
            for i, sex in enumerate(sexes):
                fig = go.Figure()
                plot_key = f"plot_{sex}"
                base_color = named_colors.get(sex, fallback_palette[i % len(fallback_palette)])
                sub = df[df['__S__'] == sex]
                
                if use_site:
                    sub = sub[sub['__SITE__'] == in_site]
                    
                if sub.empty: continue
                
                selected_id = None
                if plot_key in st.session_state:
                    sel = st.session_state[plot_key].get("selection", {}).get("points", [])
                    if sel and "customdata" in sel[0]:
                        selected_id = sel[0]["customdata"][0]
                
                x_grid = np.linspace(min_age, max_age, 100)
                mu_vec = predict_values(mean_model, x_grid, sex, use_site, in_site)
                
                if sd_model:
                    sigma_vec = predict_values(sd_model, x_grid, sex, use_site, in_site)
                    sigma_vec[sigma_vec < 0] = 1e-6
                    
                    z_low = mu_vec - z_thresh * sigma_vec
                    z_high = mu_vec + z_thresh * sigma_vec
                    
                    fig.add_trace(go.Scatter(x=x_grid, y=z_low, mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'))
                    fig.add_trace(go.Scatter(x=x_grid, y=z_high, mode='lines', line=dict(width=0), fill='tonexty', fillcolor="rgba(128, 128, 128, 0.3)", showlegend=False, hoverinfo='skip'))

                    centiles = [3, 10, 25, 50, 75, 90, 97]
                    for p in centiles:
                        c_y = mu_vec + norm.ppf(p/100) * sigma_vec
                        fig.add_trace(go.Scatter(x=x_grid, y=c_y, mode='lines', line=dict(color=base_color, width=(3 if p == 50 else 1), dash=('dash' if p == 50 else 'solid')), name=f'{p}th Centile', showlegend=False, hoverinfo='skip'))

                # Longitudinal Lines
                if connect_lines:
                    sub_sorted = sub.sort_values(by=[id_col, '__X__'])
                    x_lines_norm, y_lines_norm = [], []
                    x_lines_high, y_lines_high = [], []
                    
                    for p_id, group in sub_sorted.groupby(id_col):
                        if len(group) > 1:
                            if p_id == selected_id:
                                x_lines_high.extend(group['__X__'].tolist() + [None])
                                y_lines_high.extend(group['__Y__'].tolist() + [None])
                            else:
                                x_lines_norm.extend(group['__X__'].tolist() + [None])
                                y_lines_norm.extend(group['__Y__'].tolist() + [None])
                                
                    if x_lines_norm:
                        fig.add_trace(go.Scatter(x=x_lines_norm, y=y_lines_norm, mode='lines', line=dict(color=base_color, width=1.5), opacity=0.3, showlegend=False, hoverinfo='skip'))
                    if x_lines_high:
                        fig.add_trace(go.Scatter(x=x_lines_high, y=y_lines_high, mode='lines', line=dict(color='black', width=3.0), opacity=1.0, showlegend=False, hoverinfo='skip'))

                # Scatter Points
                show_leg = False
                if color_col != "None":
                    is_numeric = pd.api.types.is_numeric_dtype(df[color_col])
                    if is_numeric:
                        c_img = sub[img_col] if img_col != "None" else [""] * len(sub)
                        customdata = np.stack((sub[id_col], c_img, sub[color_col]), axis=-1)
                        htemp = "<b>%{customdata[0]}</b><br>Age: %{x:.2f}<br>" + display_measure + ": %{y:.4f}<br>" + color_col + ": %{customdata[2]:.2f}<extra></extra>"
                        fig.add_trace(go.Scatter(
                            x=sub['__X__'], y=sub['__Y__'], mode='markers',
                            marker=dict(color=sub[color_col], colorscale='Viridis', showscale=True, colorbar=dict(title=color_col, thickness=12, len=0.6, yanchor="top", y=1), opacity=0.8, size=6, line=dict(width=0.5, color='white')),
                            customdata=customdata, name='Data', hovertemplate=htemp
                        ))
                    else:
                        show_leg = True
                        unique_cvals = df[color_col].fillna("Missing").astype(str).unique()
                        for j, c_val in enumerate(unique_cvals):
                            sub_c = sub[sub[color_col].fillna("Missing").astype(str) == c_val]
                            if sub_c.empty: continue
                            c_img = sub_c[img_col] if img_col != "None" else [""] * len(sub_c)
                            customdata = np.stack((sub_c[id_col], c_img, sub_c[color_col].fillna("Missing").astype(str)), axis=-1)
                            htemp = "<b>%{customdata[0]}</b><br>Age: %{x:.2f}<br>" + display_measure + ": %{y:.4f}<br>" + color_col + ": %{customdata[2]}<extra></extra>"
                            fig.add_trace(go.Scatter(
                                x=sub_c['__X__'], y=sub_c['__Y__'], mode='markers',
                                marker=dict(color=cat_palette[j % len(cat_palette)], opacity=0.8, size=6, line=dict(width=0.5, color='white')),
                                customdata=customdata, name=str(c_val), hovertemplate=htemp
                            ))
                else:
                    c_img = sub[img_col] if img_col != "None" else [""] * len(sub)
                    customdata = np.stack((sub[id_col], c_img), axis=-1)
                    htemp = "<b>%{customdata[0]}</b><br>Age: %{x:.2f}<br>" + display_measure + ": %{y:.4f}<extra></extra>"
                    fig.add_trace(go.Scatter(x=sub['__X__'], y=sub['__Y__'], mode='markers', marker=dict(color=base_color, opacity=0.8, size=6, line=dict(width=0.5, color='white')), customdata=customdata, name='Data', hovertemplate=htemp))

                # User Patient Point
                if 'in_sex' in locals() and in_sex == sex:
                    fig.add_trace(go.Scatter(x=[in_age], y=[in_val_calc], mode='markers', marker=dict(color='gold', size=15, symbol='x', line=dict(width=2, color='black')), name='New Patient', showlegend=show_leg, hovertemplate="Age: %{x}<br>Val: %{y:.4f}<extra></extra>"))

                fig.update_layout(title=f"Reference Intervals for Category: {sex}", xaxis_title=age_col, yaxis_title=display_measure, template="simple_white", height=450, margin=dict(l=20, r=20, t=40, b=20), showlegend=show_leg, clickmode='event+select')
                
                event = st.plotly_chart(fig, use_container_width=True, on_select="rerun", key=plot_key)
                
                if event and len(event.selection.points) > 0:
                    clicked_point = event.selection.points[0]
                    if "customdata" in clicked_point:
                        pt_id, pt_img = clicked_point["customdata"][0], clicked_point["customdata"][1]
                        if pt_img and pt_img not in ["", "None"]:
                            with st.expander(f"📷 Viewing Image for ID: {pt_id}", expanded=True):
                                if os.path.exists(pt_img): st.image(pt_img, use_container_width=True)
                                else: st.warning(f"Could not find an image at the path: `{pt_img}`")

            # --- LOOP 2: Overlay Chart (Lines Only, No Fill) ---
            st.divider()
            st.subheader("Overlay Chart (All Categories)")
            st.caption("Centile lines for direct comparison (no shading).")
            
            fig_overlay = go.Figure()
            
            for i, sex in enumerate(sexes):
                base_color = named_colors.get(sex, fallback_palette[i % len(fallback_palette)])
                
                x_grid = np.linspace(min_age, max_age, 100)
                mu_vec = predict_values(mean_model, x_grid, sex, use_site, in_site)
                
                if sd_model:
                    sigma_vec = predict_values(sd_model, x_grid, sex, use_site, in_site)
                    sigma_vec[sigma_vec < 0] = 1e-6
                    
                    centiles = [3, 10, 25, 50, 75, 90, 97]
                    for p in centiles:
                        c_y = mu_vec + norm.ppf(p/100) * sigma_vec
                        
                        show_leg_ov = True if p == 50 else False
                        name_str = f'{sex} (Median)' if p == 50 else f'{sex} {p}th'
                        
                        fig_overlay.add_trace(go.Scatter(
                            x=x_grid, y=c_y, mode='lines', 
                            line=dict(color=base_color, width=(3 if p == 50 else 1), dash=('dash' if p == 50 else 'solid')), 
                            name=name_str, showlegend=show_leg_ov, hoverinfo='skip'
                        ))
                else:
                    fig_overlay.add_trace(go.Scatter(
                        x=x_grid, y=mu_vec, mode='lines', 
                        line=dict(color=base_color, width=3, dash='dash'), 
                        name=f'{sex} (Mean)', showlegend=True, hoverinfo='skip'
                    ))

            if 'in_age' in locals() and 'in_val_calc' in locals():
                fig_overlay.add_trace(go.Scatter(
                    x=[in_age], y=[in_val_calc], mode='markers', 
                    marker=dict(color='gold', size=15, symbol='x', line=dict(width=2, color='black')), 
                    name='New Patient', showlegend=True, hovertemplate="Age: %{x}<br>Val: %{y:.4f}<extra></extra>"
                ))

            fig_overlay.update_layout(
                title=f"Overlay Comparison" + (f" (Site: {in_site})" if use_site else ""), 
                xaxis_title=age_col, 
                yaxis_title=display_measure, 
                template="simple_white", 
                height=450, 
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_overlay, use_container_width=True)

        # --- Diagnostics ---
        st.divider()
        st.subheader("Model Diagnostics")
        
        if 'z_score' in df.columns:
            tab1, tab2, tab3 = st.tabs(["Distribution", "Q-Q Plot (Normality)", "Bias (Z vs Age)"])
            
            with tab1:
                col_d1, col_d2 = st.columns([3, 1])
                with col_d1:
                    fig_hist = go.Figure()
                    fig_hist.add_trace(go.Histogram(x=df['z_score'], histnorm='probability density', marker_color='orange', opacity=0.6, name='Data Z-Scores'))
                    x_range = np.linspace(df['z_score'].min(), df['z_score'].max(), 100)
                    y_norm = norm.pdf(x_range, 0, 1)
                    fig_hist.add_trace(go.Scatter(x=x_range, y=y_norm, mode='lines', line=dict(color='red', width=3, dash='dash'), name='Standard Normal'))
                    fig_hist.update_layout(title="Z-Score Histogram", xaxis_title="Z-Score", yaxis_title="Density", template="simple_white", height=400)
                    st.plotly_chart(fig_hist, use_container_width=True)
                with col_d2:
                    st.write("#### Stats")
                    
                    st.write(f"**Mean:** {df['z_score'].mean():.3f} (Ideal: 0)")
                    st.write(f"**SD:** {df['z_score'].std():.3f} (Ideal: 1)")
                    
                    st.write("---")
                    
                    z_median = df['z_score'].median()
                    try:
                        z_mad = sm.robust.scale.mad(df['z_score']) 
                    except:
                        z_mad = np.median(np.abs(df['z_score'] - z_median)) * 1.4826
                        
                    st.write(f"**Robust Median:** {z_median:.3f} (Ideal: 0)")
                    st.write(f"**Robust MAD:** {z_mad:.3f} (Ideal: 1)")
                    
                    st.caption("A good fit matches the red bell curve.")
                    
                    st.write("---")
                    st.write("#### Model Comparison")
                    r2_data = [{"Model": k, "Pseudo-R²": v['r2'], "SSR": v['ssr']} for k, v in all_models.items()]
                    r2_df = pd.DataFrame(r2_data).sort_values(by="Pseudo-R²", ascending=False).reset_index(drop=True)
                    st.dataframe(r2_df.style.format({"Pseudo-R²": "{:.4f}", "SSR": "{:.2f}"}), use_container_width=True)
                    st.caption("Pseudo-R² reflects model fit quality (higher is better).")
            
            with tab2:
                osm, osr = probplot(df['z_score'], dist="norm")[0]
                fig_qq = go.Figure()
                fig_qq.add_trace(go.Scatter(x=osm, y=osr, mode='markers', marker=dict(color='purple', size=5), name='Z-Scores'))
                min_val, max_val = min(osm.min(), osr.min()), max(osm.max(), osr.max())
                fig_qq.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val], mode='lines', line=dict(color='red', dash='dash'), name='Perfect Normal'))
                fig_qq.update_layout(title="Normal Q-Q Plot", xaxis_title="Theoretical Quantiles", yaxis_title="Sample Z-Scores", template="simple_white", height=400, showlegend=False)
                st.plotly_chart(fig_qq, use_container_width=True)
            
            with tab3:
                fig_scat = go.Figure()
                for i, sex in enumerate(sexes):
                    sub = df[df['__S__'] == sex]
                    fig_scat.add_trace(go.Scatter(
                        x=sub['__X__'], y=sub['z_score'], mode='markers', 
                        marker=dict(color=fallback_palette[i % len(fallback_palette)], opacity=0.6),
                        name=str(sex), text=sub[id_col], hovertemplate="<b>%{text}</b><br>Age: %{x}<br>Z: %{y:.2f}"
                    ))
                fig_scat.add_shape(type="line", x0=df['__X__'].min(), y0=0, x1=df['__X__'].max(), y1=0, line=dict(color="red", width=2, dash="dash"))
                fig_scat.update_layout(title="Z-Scores vs. Age (Bias Check)", xaxis_title=age_col, yaxis_title="Z-Score", template="simple_white", height=400)
                st.plotly_chart(fig_scat, use_container_width=True)

    except Exception as e:
        st.error(f"Error processing file: {e}")

else:
    st.info("Please upload an Excel or CSV file to start.")