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

# Check if the system is Windows
IS_WINDOWS = platform.system() == 'Windows'

if IS_WINDOWS:
    import win32com.client as win32
    win32c = win32.constants

COMMON_COLUMN_NAMES = [
    "Keyword", "Keywords", "keyword", "keywords",
    "Search Terms", "Search terms", "Search term", "Search Term"
]

def stem_and_remove_punctuation(text: str, stem: bool):
    """Process text by removing punctuation and optionally stemming."""
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Stem the text if the stem flag is True
    if stem:
        stemmer = PorterStemmer()
        text = ' '.join([stemmer.stem(word) for word in text.split()])
    return text

def create_unigram(cluster: str, stem: bool):
    """Create unigram from the cluster and return the most common word."""
    words = cluster.split()
    word_counts = Counter(words)

    # Filter out number-only words
    word_counts = Counter({word: count for word, count in word_counts.items() if not word.isdigit()})

    if word_counts:
        most_common_word = word_counts.most_common(1)[0][0]
    else:
        most_common_word = 'no_keyword'

    return stem_and_remove_punctuation(most_common_word, stem)

def load_file(uploaded_file):
    """Load a CSV file and return a DataFrame with encoding detection."""
    try:
        # Try UTF-8 first (most common)
        try:
            return pd.read_csv(uploaded_file, encoding='utf-8', on_bad_lines='skip')
        except UnicodeDecodeError:
            # Fall back to latin1 if UTF-8 fails
            return pd.read_csv(uploaded_file, encoding='latin1', on_bad_lines='skip')
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

def create_chart(df, chart_type, volume):
    """Create a sunburst chart or treemap visualization."""
    if volume is not None and volume in df.columns:
        chart_df = df.groupby(['hub', 'spoke'])[volume].sum().reset_index(name='cluster_size')
    else:
        chart_df = df.groupby(['hub', 'spoke']).size().reset_index(name='cluster_size')

    if chart_type == "sunburst":
        fig = px.sunburst(chart_df, path=['hub', 'spoke'], values='cluster_size',
                         color_discrete_sequence=px.colors.qualitative.Pastel2)
    elif chart_type == "treemap":
        fig = px.treemap(chart_df, path=['hub', 'spoke'], values='cluster_size',
                        color_discrete_sequence=px.colors.qualitative.Pastel2)
    else:
        st.error("Invalid chart type. Valid options are 'sunburst' and 'treemap'.")
        return None
    return fig

def process_data(df, column_name, model_name, device, min_similarity, stem, volume):
    """Process and cluster the keyword data."""
    df = df.copy()
    df.rename(columns={column_name: 'keyword'}, inplace=True)
    df = df[df["keyword"].notna()]
    df['keyword'] = df['keyword'].astype(str)
    
    # Perform clustering
    embedding_model = SentenceTransformer(model_name, device=device)
    distance_model = SentenceEmbeddings(embedding_model)
    model = PolyFuzz(distance_model).fit(df['keyword'].tolist())
    model.group(link_min_similarity=min_similarity)
    
    # Process results
    df_cluster = model.get_matches()
    df_cluster["Group"] = df_cluster["Similarity"].apply(
        lambda x: "no_cluster" if x < min_similarity else x)
    
    # Merge results with original data
    df = pd.merge(
        df,
        df_cluster.rename(columns={"From": "keyword", "Group": "spoke"})[['keyword', 'spoke']],
        on='keyword',
        how='left'
    )
    
    # Post-processing
    df['hub'] = df['spoke'].apply(lambda x: create_unigram(x, stem))
    df['hub'] = df['hub'].apply(lambda x: "no_cluster" if x in ["noclust", "nocluster"] else x)
    df.loc[df["hub"] == "no_cluster", "spoke"] = "no_cluster"
    
    return df[['hub', 'spoke'] + [col for col in df.columns if col not in ['hub', 'spoke']]]

def main():
    """Main application function."""
    st.set_page_config(
        page_title="Keyword Clustering Tool",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("🔍 Semantic Keyword Clustering Tool")
    st.markdown("""
    Cluster keywords by semantic similarity using Sentence Transformers.
    Upload a CSV file to get started.
    """)

    with st.sidebar:
        st.header("Settings")
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
        
        if uploaded_file is not None:
            df = load_file(uploaded_file)
            if df is not None:
                column_name = st.selectbox(
                    "Select keyword column",
                    df.columns,
                    index=next(
                        (i for i, col in enumerate(df.columns) 
                        if col.lower() in [x.lower() for x in COMMON_COLUMN_NAMES]
                    ), 0)
                )
                
                # [Rest of your sidebar widgets...]
                process_button = st.button("Process Data")

    if uploaded_file is not None and df is not None and 'process_button' in locals():
        if process_button:
            with st.spinner('Processing...'):
                try:
                    processed_df = process_data(
                        df.copy(),
                        column_name,
                        "all-MiniLM-L6-v2",  # Default model
                        "cpu",  # Default device
                        0.8,    # Default similarity
                        False,  # Default stem
                        None    # Default volume
                    )
                    
                    st.success("Processing completed!")
                    st.dataframe(processed_df.head())
                    
                    fig = create_chart(processed_df, "treemap", None)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                        
                    csv = processed_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "Download CSV",
                        csv,
                        "clustered_keywords.csv",
                        "text/csv"
                    )

                except Exception as e:
                    st.error(f"Processing error: {str(e)}")

if __name__ == "__main__":
    main()
