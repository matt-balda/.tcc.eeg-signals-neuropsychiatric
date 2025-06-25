import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

st.set_page_config(page_title="EEG Exploration", layout="wide")
st.title("🧠 Data Exploration - Park et al. (2021)")

@st.cache_data
def load_data():
    usecols = lambda col: col.startswith('AB') or col.startswith('COH') or col in ['age', 'education', 'IQ']
    return pd.read_csv('./data/raw/EEG.machinelearing_data_BRMH.csv', usecols=usecols)

@st.cache_data
def compute_corr(df_sel: pd.DataFrame):
    return df_sel.corr()

@st.cache_data
def scale_data(data: pd.DataFrame):
    scaler = StandardScaler()
    return scaler.fit_transform(data)

@st.cache_data
def compute_pca(data_scaled: np.ndarray, n_components: int):
    pca = PCA(n_components=n_components, svd_solver='randomized')
    components = pca.fit_transform(data_scaled)
    explained_variance = pca.explained_variance_ratio_.cumsum()
    return components, explained_variance

df = load_data()
st.success("✅ Data successfully loaded!")

quant_vars = ['age', 'education', 'IQ']
psd_cols = df.filter(regex=r'^AB').columns.tolist()
fc_cols = df.filter(regex=r'^COH').columns.tolist()
freq_bands = ['delta', 'theta', 'alpha', 'beta', 'highbeta', 'gamma']

# Sidebar
st.sidebar.subheader("🔢 Variable Selection")

select_all_quant = st.sidebar.checkbox("Select all quantitative variables")
quant_selected = st.sidebar.multiselect("🧮 Quantitative variables:", quant_vars, default=quant_vars if select_all_quant else [])

st.sidebar.markdown("#### 📊 PSD Bands")
select_all_psd = st.sidebar.checkbox("Select all PSD bands")
select_by_band_psd = st.sidebar.multiselect("Select specific bands (PSD):", freq_bands)

if select_all_psd:
    psd_selected = psd_cols
elif select_by_band_psd:
    psd_selected = [col for col in psd_cols if any(band in col.lower() for band in select_by_band_psd)]
else:
    psd_selected = st.sidebar.multiselect("Manual selection (PSD):", psd_cols)

st.sidebar.markdown("#### 🔗 Functional Connectivity (FC)")
select_all_fc = st.sidebar.checkbox("Select all FC connections")
select_by_band_fc = st.sidebar.multiselect("Select specific bands (FC):", freq_bands, key='fc_band')

if select_all_fc:
    fc_selected = fc_cols
elif select_by_band_fc:
    fc_selected = [col for col in fc_cols if any(band in col.lower() for band in select_by_band_fc)]
else:
    fc_selected = st.sidebar.multiselect("Manual selection (FC):", fc_cols, key='fc_manual')

color_palette = st.sidebar.selectbox("🎨 Color palette:", ['coolwarm', 'viridis', 'plasma', 'magma', 'rocket'])

selected_cols = quant_selected + psd_selected + fc_selected

if selected_cols:
    df_sel = df[selected_cols]

    st.subheader("📄 Preview of Selected Data")
    st.dataframe(df_sel.head())

    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Descriptive Stats", "🧪 Correlation", "📊 Univariate Plots", "🔍 PCA (Dim. Reduction)"
    ])

    # Descriptive Statistics
    with tab1:
        st.subheader("📈 Descriptive Statistics")
        st.dataframe(df_sel.describe().T)

    # Correlation Heatmap
    with tab2:
        st.subheader("🧪 Correlation Heatmap")

        # Reduce FC variables to 10% if all selected
        if fc_selected == fc_cols:
            st.info("🔎 Reducing FC variables to 10% to avoid slow rendering.")
            reduced_fc = fc_selected[:max(1, int(len(fc_selected) * 0.1))]
        else:
            reduced_fc = fc_selected

        corr_cols = quant_selected + psd_selected + reduced_fc
        df_corr = df[corr_cols]
        corr = compute_corr(df_corr)

        fig, ax = plt.subplots(figsize=(max(10, len(corr.columns)*0.5), max(8, len(corr.columns)*0.5)))
        cax = ax.matshow(corr.values, cmap=color_palette, vmin=-1, vmax=1)

        ax.set_xticks(range(len(corr.columns)))
        ax.set_yticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=90, fontsize=8)
        ax.set_yticklabels(corr.columns, fontsize=8)
        ax.set_title("Correlation Between Selected Variables", pad=20)
        fig.colorbar(cax, shrink=0.7)

        for i in range(len(corr.columns)):
            for j in range(len(corr.columns)):
                ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha='center', va='center', fontsize=6)

        st.pyplot(fig)

    # Univariate Visualizations
    with tab3:
        st.subheader("📊 Univariate Visualizations")

        mode = st.radio("Visualization mode:", ["Manual", "Show All"])

        if mode == "Manual":
            var_to_plot = st.selectbox("Select a variable to plot:", selected_cols)
            col1, col2 = st.columns(2)

            with col1:
                fig_hist, ax_hist = plt.subplots()
                sns.histplot(df[var_to_plot], kde=True, ax=ax_hist, color="skyblue")
                ax_hist.set_title(f"Histogram of {var_to_plot}")
                st.pyplot(fig_hist)

            with col2:
                fig_box, ax_box = plt.subplots()
                sns.boxplot(y=df[var_to_plot], ax=ax_box, color="lightcoral")
                ax_box.set_title(f"Boxplot of {var_to_plot}")
                st.pyplot(fig_box)

        else:  # Show All
            st.info(f"🔍 Rendering {len(selected_cols)} variables. This may take a few seconds...")
            for var in selected_cols:
                st.markdown(f"### 📌 Variable: `{var}`")
                col1, col2 = st.columns(2)

                with col1:
                    fig_hist, ax_hist = plt.subplots()
                    sns.histplot(df[var], kde=True, ax=ax_hist, color="skyblue")
                    ax_hist.set_title(f"Histogram of {var}")
                    st.pyplot(fig_hist)

                with col2:
                    fig_box, ax_box = plt.subplots()
                    sns.boxplot(y=df[var], ax=ax_box, color="lightcoral")
                    ax_box.set_title(f"Boxplot of {var}")
                    st.pyplot(fig_box)

                st.markdown("---")

    # PCA
    with tab4:
        combined_cols = psd_selected + fc_selected
        if len(combined_cols) >= 3:
            st.subheader("🔍 PCA for Dimensionality Reduction")
            max_comp = min(100, len(combined_cols), len(df))
            n_components = st.slider("Number of principal components:", 2, max_comp, 2)

            data_scaled = scale_data(df[combined_cols])
            components, explained = compute_pca(data_scaled, n_components)
            df_pca = pd.DataFrame(components, columns=[f'PC{i+1}' for i in range(n_components)])

            st.write("📉 Cumulative explained variance:", explained)

            col_x, col_y = st.columns(2)
            with col_x:
                pc_x = st.selectbox("X-axis component:", df_pca.columns.tolist(), index=0)
            with col_y:
                pc_y = st.selectbox("Y-axis component:", df_pca.columns.tolist(), index=1)

            fig_pca, ax_pca = plt.subplots()
            sns.scatterplot(x=pc_x, y=pc_y, data=df_pca, alpha=0.7, s=40)
            ax_pca.set_title(f"PCA - {pc_x} vs {pc_y}")
            st.pyplot(fig_pca)
        else:
            st.info("ℹ️ Select at least 3 PSD or FC variables to enable PCA.")

    # Download filtered data
    csv = df_sel.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download Filtered Data", data=csv, file_name='filtered_data.csv', mime='text/csv')

else:
    st.warning("⚠️ Please select at least one variable for visualization.")
