import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from deep_translator import GoogleTranslator

st.set_page_config(layout="wide", page_title="CineIA 9000+", page_icon="🎬")

# Função de Tradução com Cache para não travar o site
@st.cache_data
def traduzir_texto(texto):
    if not texto: return ""
    try:
        return GoogleTranslator(source='en', target='pt').translate(texto)
    except:
        return texto

@st.cache_resource
def load_and_train():
    url = "https://huggingface.co/datasets/Pablinho/movies-dataset/resolve/main/9000plus.csv"
    df = pd.read_csv(url)
    c = {'genero': 'Genre', 'sinopse': 'Overview', 'titulo': 'Title', 'imagem': 'Poster_Url'}
    
    # Limpeza e Reset de Index
    df = df.dropna(subset=[c['titulo']]).drop_duplicates(subset=[c['titulo']]).reset_index(drop=True)
    for col in c.values(): df[col] = df[col].fillna('')
    
    # --- MELHORIA NA ACURÁCIA ---
    # Damos peso 10 para o gênero e repetimos o título na busca para forçar a similaridade entre sequências
    df['combined'] = (df[c['genero']] + " ") * 10 + (df[c['titulo']] + " ") * 2 + df[c['sinopse']]
    
    tfv = TfidfVectorizer(stop_words='english', ngram_range=(1,2)) # Analisa palavras em duplas também
    tfidf_matrix = tfv.fit_transform(df['combined'])
    return df, cosine_similarity(tfidf_matrix), c

try:
    df, cosine_sim, c = load_and_train()
    st.title("🎬 CineIA")

    movie_list = sorted(df[c['titulo']].unique())
    selecao = st.sidebar.selectbox("Escolha um filme:", movie_list)
    qtd = st.sidebar.slider("Sugestões:", 6, 12, 6)

    if st.button("Gerar Recomendações"):
        idx = df[df[c['titulo']] == selecao].index[0]
        scores = list(enumerate(cosine_sim[idx]))
        # Filtramos o próprio filme para ele não recomendar a si mesmo com 100%
        recomendados = sorted(scores, key=lambda x: x[1], reverse=True)[1:qtd+1]
        
        st.subheader(f"Baseado em: {selecao}")
        cols = st.columns(6)
        
        for i, (index, score) in enumerate(recomendados):
            with cols[i % 6]:
                row = df.iloc[index]
                st.image(row[c['imagem']] if row[c['imagem']] else "https://via.placeholder.com/500x750", use_container_width=True)
                
                # Exibição do Match
                st.markdown(f'<div style="font-size:2.2rem; font-weight:800; color:#2e7d32;">{round(score * 100)}%</div>', unsafe_allow_html=True)
                st.caption("Compatível")
                st.write(f"**{row[c['titulo']]}**")
                
                with st.expander("Sinopse"):
                    # Chama a função com cache
                    resumo_pt = traduzir_texto(row[c['sinopse']])
                    st.write(resumo_pt)
                    st.divider()
                    st.caption(f"Gênero: {row[c['genero']]}")

except Exception as e:
    st.error(f"Erro: {e}")