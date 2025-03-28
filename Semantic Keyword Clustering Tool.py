import os
import platform
import string
import time
from collections import Counter
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from polyfuzz import PolyFuzz
from polyfuzz.models import SentenceEmbeddings
from sentence_transformers import SentenceTransformer
from nltk.stem import PorterStemmer

# System check
IS_WINDOWS = platform.system() == 'Windows'
if IS_WINDOWS:
    import win32com.client as win32
    win32c = win32.constants

COMMON_COLUMN_NAMES = [
    "Keyword", "Keywords", "keyword", "keywords",
    "Search Terms", "Search terms", "Search term", "Search Term"
]

def stem_and_remove_punctuation(text: str, stem: bool):
    text = str(text) if pd.notna(text) else ''
    text = text.translate(str.maketrans('', '', string.punctuation))
    if stem and text:
        stemmer = PorterStemmer()
        text = ' '.join([stemmer.stem(word) for word in text.split()])
    return text

def create_unigram(cluster: str, stem: bool):
    cluster = str(cluster) if pd.notna(cluster) else ''
    words = cluster.split()
    word_counts = Counter({
        word: count for word, count in Counter(words).items() 
        if not (word.replace('.', '').isdigit() or word.replace(',', '').isdigit())
    })
    return stem_and_remove_punctuation(
        word_counts.most_common(1)[0][0] if word_counts else 'no_keyword',
        stem
    )

def load_file(uploaded_file):
    try:
        try:
            return pd.read_csv(uploaded_file, encoding='utf-8', on_bad_lines='skip')
        except UnicodeDecodeError:
            return pd.read_csv(uploaded_file, encoding='latin1', on_bad_lines='skip')
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

def create_chart(df, chart_type, volume):
    if df.empty or 'hub' not in df.columns or 'spoke' not in df.columns:
        return None
        
    chart_df = df.groupby(['hub', 'spoke'])[volume].sum().reset_index(name='cluster_size') if volume else df.groupby(['hub', 'spoke']).size().reset_index(name='cluster_size')

    if chart_type == "sunburst":
        return px.sunburst(chart_df, path=['hub', 'spoke'], values='cluster_size',
                         color_discrete_sequence=px.colors.qualitative.Pastel2)
    return px.treemap(chart_df, path=['hub', 'spoke'], values='cluster_size',
                    color_discrete_sequence=px.colors.qualitative.Pastel2)

def process_data(df, column_name, model_name, device, min_similarity, stem, volume):
    df = df.copy()
    df.rename(columns={column_name: 'keyword'}, inplace=True)
    df['keyword'] = df['keyword'].astype(str).str.strip().replace(['nan', '', 'None'], 'no_keyword')
    df = df[df["keyword"] != 'no_keyword']
    
    if df.empty:
        st.error("No valid keywords found")
        return pd.DataFrame()

    try:
        model = PolyFuzz(SentenceEmbeddings(SentenceTransformer(model_name, device=device)))
        model.fit(df['keyword'].tolist()).group(link_min_similarity=min_similarity)
        df_cluster = model.get_matches()
        df_cluster["Group"] = df_cluster["Similarity"].apply(lambda x: "no_cluster" if x < min_similarity else "cluster")
        df = pd.merge(df, df_cluster.rename(columns={"From": "keyword", "Group": "spoke"})[['keyword', 'spoke']], on='keyword', how='left')
        df['hub'] = df['spoke'].apply(lambda x: create_unigram(x, stem))
        df['hub'] = df['hub'].apply(lambda x: "no_cluster" if x in ["noclust", "nocluster"] else x)
        df.loc[df["hub"] == "no_cluster", "spoke"] = "no_cluster"
        return df[['hub', 'spoke'] + [col for col in df.columns if col not in ['hub', 'spoke']]]
    except Exception as e:
        st.error(f"Clustering error: {str(e)}")
        return pd.DataFrame()

def main():
    st.set_page_config(
        page_title="Keyword Clustering Tool",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("🔍 Semantic Keyword Clustering Tool")
    st.markdown("Cluster keywords by semantic similarity using Sentence Transformers.")

    with st.sidebar:
        st.header("Settings")
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
        
        if uploaded_file is not None:
            df = load_file(uploaded_file)
            if df is not None and not df.empty:
                # Fixed selectbox with properly closed parentheses
                column_name = st.selectbox(
                    "Select keyword column",
                    df.columns,
                    index=next(
                        (i for i, col in enumerate(df.columns) 
                        if col.lower() in [x.lower() for x in COMMON_COLUMN_NAMES]
                    )
                )
                
                volume = st.selectbox(
                    "Volume column (optional)",
                    [None] + [col for col in df.columns if col != column_name]
                )
                
                model_name = st.selectbox(
                    "Model",
                    ["all-MiniLM-L6-v2", "paraphrase-MiniLM-L6-v2", "all-mpnet-base-v2"],
                    index=0
                )
                
                device = st.selectbox(
                    "Device",
                    ["cpu", "cuda"] if platform.system() != "Darwin" else ["cpu"],
                    index=0
                )
                
                min_similarity = st.slider(
                    "Minimum similarity", 
                    min_value=0.1, 
                    max_value=1.0, 
                    value=0.8, 
                    step=0.05
                )
                
                chart_type = st.selectbox(
                    "Chart type", 
                    ["treemap", "sunburst"], 
                    index=0
                )
                
                stem = st.checkbox("Enable stemming", False)
                process_button = st.button("Process Data")

    if uploaded_file is not None and 'process_button' in locals() and process_button:
        with st.spinner('Processing...'):
            processed_df = process_data(
                df, column_name, model_name, device, min_similarity, stem, volume
            )
            
            if not processed_df.empty:
                st.success("Processing completed!")
                st.dataframe(processed_df.head())
                
                fig = create_chart(processed_df, chart_type, volume)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                    
                st.download_button(
                    "Download CSV",
                    processed_df.to_csv(index=False).encode('utf-8'),
                    "clustered_keywords.csv",
                    "text/csv"
                )
            else:
                st.warning("No valid clusters created. Try lowering the similarity threshold.")

if __name__ == "__main__":
    main()
