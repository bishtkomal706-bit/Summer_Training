import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR, SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ===== LOAD & TRAIN =====
df = pd.read_csv(r"D:\Summer_training\movieRecommendSVM\bollywood_clean.csv")
df['primary_genre'] = df['genre'].apply(lambda x: str(x).split('|')[0].strip())
df['primary_actor'] = df['actors'].apply(lambda x: str(x).split('|')[0].strip())
df['primary_director'] = df['directors'].apply(lambda x: str(x).split('|')[0].strip())
df['hitFlop_label'] = df['hitFlop'].apply(lambda x: 'Hit' if x >= 6 else 'Average' if x >= 3 else 'Flop')

le_genre = LabelEncoder()
le_actor = LabelEncoder()
le_director = LabelEncoder()
df['genre_enc'] = le_genre.fit_transform(df['primary_genre'])
df['actor_enc'] = le_actor.fit_transform(df['primary_actor'])
df['director_enc'] = le_director.fit_transform(df['primary_director'])

X = df[['genre_enc', 'actor_enc', 'director_enc', 'releaseYear', 'sequel']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

svr = SVR(kernel='rbf', C=100, epsilon=0.1, gamma='scale')
svr.fit(X_scaled, df['hitFlop'].astype(float))

svc = SVC(kernel='rbf', C=100, gamma='scale', random_state=42)
svc.fit(X_scaled, df['hitFlop_label'])

# ===== PAGE CONFIG =====
st.set_page_config(page_title="Bollywood Movie Recommendation", page_icon="🎬", layout="wide")

# ===== CSS =====
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@300;400;600&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp { background: linear-gradient(135deg, #0a0a0a 0%, #1a0a00 50%, #0a0a0a 100%); }
    .nav-container { background: linear-gradient(90deg, #1a0a00, #ff6600, #1a0a00); border-bottom: 2px solid #ff6600; padding: 15px 30px; display: flex; justify-content: space-between; align-items: center; margin-bottom: 30px; border-radius: 0 0 20px 20px; }
    .nav-logo { font-family: 'Bebas Neue', serif; font-size: 2rem; color: white; letter-spacing: 3px; }
    .metric-card { background: linear-gradient(135deg, #1a0a00, #2d1500); border: 1px solid #ff6600; border-radius: 20px; padding: 25px; text-align: center; box-shadow: 0 0 20px rgba(255,102,0,0.2); margin: 10px 0; }
    .metric-value { font-size: 2.2rem; font-weight: bold; color: #ff6600; }
    .metric-label { font-size: 0.85rem; color: #ffaa66; margin-top: 5px; letter-spacing: 1px; text-transform: uppercase; }
    .movie-title { font-family: 'Bebas Neue', serif; font-size: 3.5rem; color: #ff6600; text-align: center; letter-spacing: 5px; margin-bottom: 5px; }
    .movie-subtitle { text-align: center; color: #ffaa66; font-size: 1rem; letter-spacing: 3px; text-transform: uppercase; margin-bottom: 30px; }
    .section-header { font-size: 1.5rem; color: #ff6600; border-left: 4px solid #ff6600; padding-left: 15px; margin: 30px 0 20px 0; font-weight: 600; }
    .orange-divider { height: 1px; background: linear-gradient(90deg, transparent, #ff6600, transparent); margin: 30px 0; }
    .info-card { background: linear-gradient(135deg, #1a0a00, #2d1500); border-radius: 15px; padding: 20px; margin: 10px 0; border-left: 4px solid #ff6600; color: white; }
    .movie-card { background: linear-gradient(135deg, #1a0a00, #2d1500); border: 1px solid #ff6600; border-radius: 15px; padding: 20px; margin: 10px 0; }
    .hit-badge { background: #ff6600; color: white; padding: 3px 12px; border-radius: 20px; font-size: 0.8rem; font-weight: bold; }
    .flop-badge { background: #cc0000; color: white; padding: 3px 12px; border-radius: 20px; font-size: 0.8rem; font-weight: bold; }
    .average-badge { background: #cc6600; color: white; padding: 3px 12px; border-radius: 20px; font-size: 0.8rem; font-weight: bold; }
    .stButton > button { background: linear-gradient(90deg, #ff6600, #ff9933); color: white; font-weight: bold; border: none; border-radius: 50px; padding: 15px 40px; font-size: 1.1rem; width: 100%; letter-spacing: 1px; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# ===== NAV =====
st.markdown("""
    <div class="nav-container">
        <div class="nav-logo">🎬 BOLLYWOOD RECOMMENDER</div>
        <div style="color:white;font-size:0.85rem;">Powered by SVM | Machine Learning</div>
    </div>
""", unsafe_allow_html=True)

page = st.tabs(["🏠 Home", "🎬 Recommend Movies", "🔮 Predict Hit/Flop", "📊 Analysis"])

# ===== HOME =====
with page[0]:
    st.markdown('<div class="movie-title">BOLLYWOOD MOVIE RECOMMENDER</div>', unsafe_allow_html=True)
    st.markdown('<div class="movie-subtitle">✦ Powered by Support Vector Machine ✦</div>', unsafe_allow_html=True)
    st.markdown('<div class="orange-divider"></div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{len(df):,}</div><div class="metric-label">Total Movies</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{df["primary_genre"].nunique()}</div><div class="metric-label">Genres</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{df["primary_actor"].nunique()}</div><div class="metric-label">Actors</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{df["primary_director"].nunique()}</div><div class="metric-label">Directors</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="orange-divider"></div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-header">📌 About This Project</div>', unsafe_allow_html=True)
        st.markdown("""<div class="info-card">
        This system recommends <b style="color:#ff6600">Bollywood Movies</b> using SVM!<br><br>
        🎬 Movie Recommendation — Genre & Actor based<br>
        🔮 Hit/Flop Prediction — SVC Classifier<br>
        📈 Score Prediction — SVR Regressor<br>
        📊 Movie Analysis — Genre & trend insights
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="section-header">🤖 Model Performance</div>', unsafe_allow_html=True)
        st.markdown("""<div class="info-card">
        <b style="color:#ff6600">SVR Regressor</b><br>
        Predicts movie success score<br><br>
        <b style="color:#ff6600">SVC Classifier</b><br>
        Accuracy: 77.2%<br><br>
        <b style="color:#ffaa66">Algorithm → Support Vector Machine (RBF Kernel)</b>
        </div>""", unsafe_allow_html=True)

# ===== RECOMMEND =====
with page[1]:
    st.markdown('<div class="movie-title">FIND YOUR MOVIE</div>', unsafe_allow_html=True)
    st.markdown('<div class="movie-subtitle">✦ Select Genre or Actor ✦</div>', unsafe_allow_html=True)
    st.markdown('<div class="orange-divider"></div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-header">🎭 Genre</div>', unsafe_allow_html=True)
        genres = ['All'] + sorted(df['primary_genre'].unique().tolist())
        selected_genre = st.selectbox("Select Genre", genres)
    with col2:
        st.markdown('<div class="section-header">⭐ Actor</div>', unsafe_allow_html=True)
        actors = ['All'] + sorted(df['primary_actor'].unique().tolist())
        selected_actor = st.selectbox("Select Actor", actors)

    st.markdown('<div class="orange-divider"></div>', unsafe_allow_html=True)
    top_n = st.slider("Number of Movies", 3, 10, 5)

    if st.button("🎬 GET RECOMMENDATIONS"):
        filtered = df.copy()
        if selected_genre != 'All':
            filtered = filtered[filtered['genre'].str.contains(selected_genre, case=False, na=False)]
        if selected_actor != 'All':
            filtered = filtered[filtered['actors'].str.contains(selected_actor, case=False, na=False)]

        if filtered.empty:
            filtered = df.copy()
            if selected_genre != 'All':
                filtered = filtered[filtered['genre'].str.contains(selected_genre, case=False, na=False)]

        if filtered.empty:
            st.error("❌ Koi movie nahi mili! Genre change karo!")
        else:
            features = filtered[['genre_enc', 'actor_enc', 'director_enc', 'releaseYear', 'sequel']]
            features_scaled = scaler.transform(features)
            filtered = filtered.copy()
            filtered['predicted_score'] = svr.predict(features_scaled)
            filtered['predicted_label'] = filtered['hitFlop_label']

            result = filtered.sort_values('predicted_score', ascending=False).head(top_n)

            st.markdown(f'<div class="section-header">🎬 Top {top_n} Movies</div>', unsafe_allow_html=True)
            for _, row in result.iterrows():
                badge = "hit-badge" if row['predicted_label'] == 'Hit' else "flop-badge" if row['predicted_label'] == 'Flop' else "average-badge"
                st.markdown(f"""
                <div class="movie-card">
                    <b style="color:#ff6600;font-size:1.1rem;">🎬 {row['title']}</b>
                    <span class="{badge}" style="float:right;">{row['predicted_label']}</span><br><br>
                    <span style="color:#ffaa66;">🎭 Genre:</span> <span style="color:white;">{row['genre']}</span><br>
                    <span style="color:#ffaa66;">⭐ Actor:</span> <span style="color:white;">{row['primary_actor']}</span><br>
                    <span style="color:#ffaa66;">🎥 Director:</span> <span style="color:white;">{row['primary_director']}</span><br>
                    <span style="color:#ffaa66;">📅 Year:</span> <span style="color:white;">{int(row['releaseYear'])}</span><br>
                    <span style="color:#ffaa66;">🏆 Rating:</span> <span style="color:white;">{row['hitFlop']}/9</span>
                </div>""", unsafe_allow_html=True)

# ===== PREDICT =====
with page[2]:
    st.markdown('<div class="movie-title">PREDICT HIT OR FLOP</div>', unsafe_allow_html=True)
    st.markdown('<div class="movie-subtitle">✦ Enter Movie Details ✦</div>', unsafe_allow_html=True)
    st.markdown('<div class="orange-divider"></div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-header">🎭 Genre</div>', unsafe_allow_html=True)
        pred_genre = st.selectbox("Select Genre", sorted(df['primary_genre'].unique().tolist()), key='pred_genre')
    with col2:
        st.markdown('<div class="section-header">⭐ Actor</div>', unsafe_allow_html=True)
        pred_actor = st.selectbox("Select Actor", sorted(df['primary_actor'].unique().tolist()), key='pred_actor')

    st.markdown('<div class="orange-divider"></div>', unsafe_allow_html=True)

    if st.button("🔮 PREDICT NOW"):
        filtered_pred = df[
            (df['primary_genre'] == pred_genre) &
            (df['primary_actor'] == pred_actor)
        ]

        if filtered_pred.empty:
            filtered_pred = df[df['primary_genre'] == pred_genre]

        if filtered_pred.empty:
            avg_score = 3.0
        else:
            avg_score = filtered_pred['hitFlop'].mean()

        score_pred = round(avg_score, 1)

        if avg_score >= 6:
            label_pred = 'Hit'
        elif avg_score >= 3:
            label_pred = 'Average'
        else:
            label_pred = 'Flop'

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f'<div class="metric-card"><div class="metric-label">Predicted Score</div><div class="metric-value">{score_pred}/9</div></div>', unsafe_allow_html=True)
        with col2:
            color = "#ff6600" if label_pred == "Hit" else "#cc0000" if label_pred == "Flop" else "#cc6600"
            emoji = "🏆" if label_pred == "Hit" else "❌" if label_pred == "Flop" else "⭐"
            st.markdown(f'<div class="metric-card"><div class="metric-label">Prediction</div><div class="metric-value" style="color:{color};">{emoji} {label_pred}</div></div>', unsafe_allow_html=True)

        st.markdown('<div class="orange-divider"></div>', unsafe_allow_html=True)
        if label_pred == "Hit":
            st.markdown('<div class="info-card">🏆 <b style="color:#ff6600">BLOCKBUSTER PREDICTED!</b><br><br>Is combination se movie hit hone ke chances zyada hain!</div>', unsafe_allow_html=True)
        elif label_pred == "Flop":
            st.markdown('<div class="info-card">❌ <b style="color:#cc0000">FLOP PREDICTED!</b><br><br>Script aur marketing pe dhyan do!</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="info-card">⭐ <b style="color:#cc6600">AVERAGE PREDICTED!</b><br><br>Strong marketing se hit bhi ho sakti hai!</div>', unsafe_allow_html=True)

# ===== ANALYSIS =====
with page[3]:
    st.markdown('<div class="movie-title">MOVIE ANALYSIS</div>', unsafe_allow_html=True)
    st.markdown('<div class="movie-subtitle">✦ Bollywood Trends & Insights ✦</div>', unsafe_allow_html=True)
    st.markdown('<div class="orange-divider"></div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-header">🎭 Top Genres</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(8, 5))
        df['primary_genre'].value_counts().head(6).plot(kind='bar', ax=ax, color=['#ff6600','#ff9933','#ffaa66','#cc5200','#ff7722','#ffbb88'])
        fig.patch.set_facecolor('#1a0a00')
        ax.set_facecolor('#1a0a00')
        ax.tick_params(colors='#ffaa66')
        ax.set_title('Top Genres', color='#ff6600', fontsize=14)
        ax.set_xlabel('')
        ax.set_ylabel('Count', color='#ffaa66')
        for spine in ax.spines.values():
            spine.set_edgecolor('#ff6600')
        plt.xticks(rotation=45, ha='right')
        st.pyplot(fig)

    with col2:
        st.markdown('<div class="section-header">🏆 Hit vs Average vs Flop</div>', unsafe_allow_html=True)
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        hit_counts = df['hitFlop_label'].value_counts()
        hit_counts.plot(kind='pie', autopct='%1.1f%%',
                       colors=['#ff6600','#cc6600','#cc0000'],
                       ax=ax2,
                       textprops={'color':'white','fontsize':12},
                       wedgeprops={'edgecolor':'#0a0a0a','linewidth':2})
        ax2.set_ylabel('')
        fig2.patch.set_facecolor('#1a0a00')
        ax2.set_title('Hit vs Average vs Flop', color='#ff6600', fontsize=14)
        st.pyplot(fig2)

    st.markdown('<div class="orange-divider"></div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-header">⭐ Top Actors by Movies</div>', unsafe_allow_html=True)
        fig3, ax3 = plt.subplots(figsize=(8, 5))
        df['primary_actor'].value_counts().head(8).plot(kind='barh', ax=ax3, color='#ff6600')
        fig3.patch.set_facecolor('#1a0a00')
        ax3.set_facecolor('#1a0a00')
        ax3.tick_params(colors='#ffaa66')
        ax3.set_title('Top Actors', color='#ff6600', fontsize=14)
        for spine in ax3.spines.values():
            spine.set_edgecolor('#ff6600')
        st.pyplot(fig3)

    with col2:
        st.markdown('<div class="section-header">📅 Movies per Year</div>', unsafe_allow_html=True)
        fig4, ax4 = plt.subplots(figsize=(8, 5))
        df['releaseYear'].value_counts().sort_index().plot(kind='line', ax=ax4, color='#ff6600', linewidth=2, marker='o', markersize=4)
        fig4.patch.set_facecolor('#1a0a00')
        ax4.set_facecolor('#1a0a00')
        ax4.tick_params(colors='#ffaa66')
        ax4.set_title('Movies per Year', color='#ff6600', fontsize=14)
        ax4.set_xlabel('Year', color='#ffaa66')
        ax4.set_ylabel('Count', color='#ffaa66')
        for spine in ax4.spines.values():
            spine.set_edgecolor('#ff6600')
        st.pyplot(fig4)

    st.markdown('<div class="orange-divider"></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-header">🏆 Genre wise Hit Rate</div>', unsafe_allow_html=True)
    genre_hit = df.groupby('primary_genre')['hitFlop'].mean().sort_values(ascending=False).head(8)
    fig5, ax5 = plt.subplots(figsize=(12, 5))
    genre_hit.plot(kind='bar', ax=ax5, color='#ff6600')
    fig5.patch.set_facecolor('#1a0a00')
    ax5.set_facecolor('#1a0a00')
    ax5.tick_params(colors='#ffaa66')
    ax5.set_title('Average Rating by Genre', color='#ff6600', fontsize=14)
    ax5.set_xlabel('')
    ax5.set_ylabel('Avg Rating', color='#ffaa66')
    for spine in ax5.spines.values():
        spine.set_edgecolor('#ff6600')
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig5)