import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# ZMIANA: Usunięto RandomForest, dodano LinearRegression i GradientBoosting
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ---------------------------------------------------------
# Konfiguracja strony
# ---------------------------------------------------------
st.set_page_config(
    page_title="Wine Analytics Pro & Food Pairings",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title(" 🍷 Wine Analytics Pro & Food Pairings")
st.markdown(
    "Rozbudowana aplikacja do eksploracji jakości czerwonych win (Advanced Analytics) "
    "oraz inteligentnego parowania win z jedzeniem."
)

# ---------------------------------------------------------
# Funkcje wczytywania danych
# ---------------------------------------------------------
@st.cache_data
def load_wine_quality(path: str = "winequality-red.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

@st.cache_data
def load_wine_food_pairings(path: str = "wine_food_pairings.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

# Próba wczytania danych
wine_quality_df, pairings_df = None, None
wine_quality_error, pairings_error = None, None

try:
    wine_quality_df = load_wine_quality()
except Exception as e:
    wine_quality_error = str(e)

try:
    pairings_df = load_wine_food_pairings()
except Exception as e:
    pairings_error = str(e)

# ---------------------------------------------------------
# Sidebar – wybór modułu
# ---------------------------------------------------------
st.sidebar.header(" ⚙️  Ustawienia")
module = st.sidebar.radio(
    "Wybierz moduł:",
    options=["Analiza jakości wina", "Parowanie wina z jedzeniem"]
)

# =========================================================
# 1. ANALIZA JAKOŚCI WINA (winequality-red.csv)
# =========================================================
if module == "Analiza jakości wina":
    if wine_quality_df is None:
        st.error(f"Błąd wczytywania `winequality-red.csv`: {wine_quality_error}")
        st.stop()
    
    df = wine_quality_df.copy()

    # Użycie zakładek dla lepszej organizacji
    tab1, tab2, tab3 = st.tabs(["📊 Eksploracja Danych", "📈 Zaawansowane Wizualizacje", "🤖 Modelowanie ML"])

    # -----------------------------------------------------
    # TAB 1: Podstawowa Eksploracja
    # -----------------------------------------------------
    with tab1:
        st.subheader("Podstawowe statystyki i rozkłady")
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.write("Pierwsze wiersze datasetu:")
            st.dataframe(df.head(), use_container_width=True)
            st.write(f"Wymiary: {df.shape}")
        with col2:
            st.write("Statystyki opisowe:")
            st.dataframe(df.describe().T.style.format("{:.2f}"), use_container_width=True)

        st.markdown("---")
        
        st.markdown("### 🔥 Macierz Korelacji")
        corr = df.corr(numeric_only=True)
        fig_corr, ax_corr = plt.subplots(figsize=(10, 5))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax_corr, linewidths=0.5)
        st.pyplot(fig_corr)

    # -----------------------------------------------------
    # TAB 2: Zaawansowane Wizualizacje
    # -----------------------------------------------------
    with tab2:
        st.subheader("Zaawansowana Analityka Wizualna")

        # 1. Boxploty
        st.markdown("### 1. Analiza rozkładu cech (Boxplot)")
        st.info("Wykresy pudełkowe pozwalają zidentyfikować wartości odstające (outliers) w każdej klasie jakości.")
        feature_to_plot = st.selectbox("Wybierz cechę do analizy:", df.columns.drop('quality'))
        
        fig_box = px.box(df, x="quality", y=feature_to_plot, color="quality", 
                         title=f"Rozkład {feature_to_plot} w zależności od jakości",
                         color_discrete_sequence=px.colors.qualitative.Bold)
        st.plotly_chart(fig_box, use_container_width=True)

        col_adv1, col_adv2 = st.columns(2)

        # 2. Wykres Radarowy
        with col_adv1:
            st.markdown("### 2. Profil Wina: Dobre vs. Słabe (Radar Chart)")
            # Normalizacja danych
            df_norm = (df - df.min()) / (df.max() - df.min())
            df_norm['quality_label'] = df['quality'].apply(lambda x: 'Wysoka jakość (>6)' if x > 6 else 'Niska/Średnia (<=6)')
            
            radar_data = df_norm.groupby('quality_label').mean().reset_index()
            categories = list(df.columns.drop('quality'))
            
            fig_radar = go.Figure()
            
            for label in radar_data['quality_label'].unique():
                values = radar_data[radar_data['quality_label'] == label][categories].values.flatten().tolist()
                values += values[:1]
                cats = categories + [categories[0]]
                
                fig_radar.add_trace(go.Scatterpolar(
                    r=values,
                    theta=cats,
                    fill='toself',
                    name=label
                ))

            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                showlegend=True,
                title="Porównanie średnich znormalizowanych cech"
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        # 3. Wykres 3D
        with col_adv2:
            st.markdown("### 3. Interaktywna analiza 3D")
            st.write("Wybierz 3 wymiary, aby poszukać klastrów.")
            x_ax = st.selectbox("Oś X", df.columns, index=0)
            y_ax = st.selectbox("Oś Y", df.columns, index=1)
            z_ax = st.selectbox("Oś Z", df.columns, index=10)
            
            fig_3d = px.scatter_3d(
                df, x=x_ax, y=y_ax, z=z_ax,
                color='quality',
                opacity=0.7,
                color_continuous_scale='Viridis',
                title=f"Relacja 3D: {x_ax} vs {y_ax} vs {z_ax}"
            )
            st.plotly_chart(fig_3d, use_container_width=True)

    # -----------------------------------------------------
    # TAB 3: Modelowanie ML (BEZ RANDOM FOREST)
    # -----------------------------------------------------
    with tab3:
        st.subheader("🤖 Predykcja jakości wina")
        
        col_set1, col_set2 = st.columns(2)
        
        with col_set1:
            st.markdown("**Konfiguracja modelu**")
            # Wybór modelu: Gradient Boosting lub Regresja Liniowa
            model_type = st.selectbox(
                "Wybierz algorytm:",
                ["Gradient Boosting (Zaawansowany)", "Regresja Liniowa (Baseline)"]
            )
            test_size = st.slider("Zbiór testowy (%)", 10, 50, 20) / 100.0
            
        with col_set2:
            st.markdown("**Hiperparametry (tylko dla Gradient Boosting)**")
            if "Gradient Boosting" in model_type:
                n_estimators = st.slider("Liczba estymatorów", 50, 500, 200, step=50)
                learning_rate = st.slider("Learning rate", 0.01, 0.3, 0.1, step=0.01)
            else:
                st.info("Regresja liniowa nie wymaga dobierania hiperparametrów w tym widoku.")
                n_estimators = 100 # dummy value
                learning_rate = 0.1

        random_seed = 42

        # Przygotowanie danych
        X = df.drop("quality", axis=1)
        y = df["quality"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_seed)

        # Inicjalizacja modelu
        model = None
        if "Gradient Boosting" in model_type:
            model = GradientBoostingRegressor(
                n_estimators=n_estimators, 
                learning_rate=learning_rate, 
                random_state=random_seed
            )
        else:
            model = LinearRegression()

        if st.button("Trenuj model"):
            with st.spinner("Trenowanie modelu..."):
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                r2 = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))

            st.success(f"Model {model_type} wytrenowany pomyślnie!")
            
            m1, m2, m3 = st.columns(3)
            m1.metric("R² Score", f"{r2:.3f}", delta_color="normal")
            m2.metric("MAE", f"{mae:.3f}", delta_color="inverse")
            m3.metric("RMSE", f"{rmse:.3f}", delta_color="inverse")

            # Wykres: Rzeczywiste vs Przewidywane (BEZ trendline="ols", żeby nie wywołać błędu statsmodels)
            fig_res = px.scatter(
                x=y_test, y=y_pred, 
                labels={'x': 'Rzeczywista jakość', 'y': 'Przewidywana jakość'},
                title=f"Wydajność modelu: {model_type}",
                opacity=0.6
            )
            # Dodanie idealnej linii 1:1 dla odniesienia
            fig_res.add_shape(type="line",
                x0=y_test.min(), y0=y_test.min(), x1=y_test.max(), y1=y_test.max(),
                line=dict(color="Red", dash="dash")
            )
            st.plotly_chart(fig_res, use_container_width=True)

            # Feature Importance tylko dla Gradient Boosting
            if "Gradient Boosting" in model_type:
                importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=True)
                fig_imp = px.bar(importances, orientation='h', title="Ważność cech (Feature Importance)")
                st.plotly_chart(fig_imp, use_container_width=True)
            
            st.session_state['trained_model'] = model
            st.session_state['model_columns'] = X.columns

        st.markdown("---")
        st.markdown("### 🔮 Symulator jakości (Predykcja własna)")
        
        if 'trained_model' in st.session_state:
            with st.form("user_pred_form"):
                st.write("Dostosuj parametry chemiczne wina:")
                u_cols = st.columns(4)
                user_input = {}
                cols_list = list(st.session_state['model_columns'])
                
                for i, col_name in enumerate(cols_list):
                    col = u_cols[i % 4]
                    min_v, max_v = float(df[col_name].min()), float(df[col_name].max())
                    mean_v = float(df[col_name].mean())
                    user_input[col_name] = col.slider(col_name, min_v, max_v, mean_v)
                
                submitted = st.form_submit_button("Oceń wino")
                if submitted:
                    input_df = pd.DataFrame([user_input])
                    prediction = st.session_state['trained_model'].predict(input_df)[0]
                    
                    fig_gauge = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = prediction,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Przewidywana Jakość"},
                        gauge = {
                            'axis': {'range': [3, 9]},
                            'bar': {'color': "darkred"},
                            'steps': [
                                {'range': [3, 5], 'color': "#f4cccc"},
                                {'range': [5, 7], 'color': "#ea9999"},
                                {'range': [7, 9], 'color': "#e06666"}],
                        }
                    ))
                    st.plotly_chart(fig_gauge, use_container_width=True)
        else:
            st.warning("Najpierw wytrenuj model (kliknij przycisk powyżej), aby korzystać z symulatora.")

# =========================================================
# 2. PAROWANIE WINA Z JEDZENIEM (wine_food_pairings.csv)
# =========================================================
elif module == "Parowanie wina z jedzeniem":
    if pairings_df is None:
        st.error(f"Błąd wczytywania `wine_food_pairings.csv`: {pairings_error}")
        st.stop()
        
    dfp = pairings_df.copy()
    st.subheader(" 🍽️ Inteligentne Parowanie Wina")

    with st.expander("🔍 Filtry wyszukiwania", expanded=True):
        col_f1, col_f2, col_f3 = st.columns(3)
        wine_type_sel = col_f1.multiselect("Typ wina", options=sorted(dfp["wine_type"].unique()))
        cuisine_sel = col_f2.multiselect("Kuchnia", options=sorted(dfp["cuisine"].unique()))
        min_score = col_f3.slider("Min. ocena parowania", int(dfp["pairing_quality"].min()), int(dfp["pairing_quality"].max()), int(dfp["pairing_quality"].min()))

    filt = dfp.copy()
    if wine_type_sel:
        filt = filt[filt["wine_type"].isin(wine_type_sel)]
    if cuisine_sel:
        filt = filt[filt["cuisine"].isin(cuisine_sel)]
    filt = filt[filt["pairing_quality"] >= min_score]

    col_viz1, col_viz2 = st.columns([2, 1])
    
    with col_viz1:
        st.markdown(f"### Znaleziono **{filt.shape[0]}** rekomendacji")
        st.dataframe(
            filt[["food_item", "wine_type", "cuisine", "pairing_quality", "description"]]
            .sort_values(by="pairing_quality", ascending=False),
            height=300,
            use_container_width=True
        )
        
    with col_viz2:
        st.markdown("### Statystyki")
        if not filt.empty:
            fig_pie = px.pie(filt, names='wine_type', title='Rozkład typów wina w wynikach', hole=0.4)
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("Brak danych do wykresu.")

    st.markdown("---")
    
    st.markdown("### 🔎 Wyszukiwarka dań")
    search_term = st.text_input("Wpisz nazwę dania (np. 'Duck', 'Cheese'):", "")
    
    if search_term:
        rec = dfp[dfp["food_item"].str.contains(search_term, case=False, na=False)]
        if rec.empty:
            st.warning("Nie znaleziono takiego dania.")
        else:
            best_match = rec.sort_values(by="pairing_quality", ascending=False).iloc[0]
            st.success(f"Top rekomendacja dla '{search_term}':")
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Wino", best_match['wine_type'])
            c2.metric("Kategoria", best_match['wine_category'])
            c3.metric("Ocena", f"{best_match['pairing_quality']}/100")
            
            st.info(f"💡 **Dlaczego?** {best_match['description']}")
            
            with st.expander("Zobacz wszystkie wyniki dla tego zapytania"):
                st.dataframe(rec)
