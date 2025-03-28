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
from stqdm import stqdm

# System check
IS_WINDOWS = platform.system() == 'Windows'
if IS_WINDOWS:
    import win32com.client as win32
    win32c = win32.constants

COMMON_COLUMN_NAMES = [
    "Keyword", "Keywords", "keyword", "keywords",
    "Search Terms", "Search terms", "Search term", "Search Term"
]

# Cache the SentenceTransformer model to avoid reloading
@st.cache_resource
def load_embedding_model(model_name: str, device: str):
    """Load and cache the SentenceTransformer model."""
    with st.spinner(f'Loading {model_name}... This may take a minute.'):
        return SentenceEmbeddings(SentenceTransformer(model_name, device=device))

def stem_and_remove_punctuation(text: str, stem: bool, exclude_words: list = None):
    """Process text by removing punctuation and optionally stemming."""
    text = str(text) if pd.notna(text) else ''
    text = text.translate(str.maketrans('', '', string.punctuation))
    if stem and text:
        stemmer = PorterStemmer()
        words = text.split()
        if exclude_words:
            words = [word for word in words if word.lower() not in exclude_words]
        text = ' '.join([stemmer.stem(word) for word in words])
    return text

def create_unigram(cluster: str, stem: bool, exclude_words: list = None):
    """Create unigram from the cluster and return the most common word."""
    cluster = str(cluster) if pd.notna(cluster) else ''
    words = cluster.split()
    
    # Filter out numbers and excluded words
    word_counts = Counter({
        word: count for word, count in Counter(words).items() 
        if (not (word.replace('.', '').isdigit() or word.replace(',', '').isdigit()) and
            (not exclude_words or word.lower() not in exclude_words))
    })
    
    return stem_and_remove_punctuation(
        word_counts.most_common(1)[0][0] if word_counts else 'no_keyword',
        stem,
        exclude_words
    )

def load_file(uploaded_file):
    """Load a CSV file with automatic encoding detection."""
    try:
        try:
            return pd.read_csv(uploaded_file, encoding='utf-8', on_bad_lines='skip')
        except UnicodeDecodeError:
            return pd.read_csv(uploaded_file, encoding='latin1', on_bad_lines='skip')
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

def create_chart(df, chart_type, volume, top_n=None):
    """Create visualization of clustered data."""
    if df.empty or 'hub' not in df.columns or 'spoke' not in df.columns:
        return None
        
    chart_df = df.groupby(['hub', 'spoke'])[volume].sum().reset_index(name='cluster_size') if volume else df.groupby(['hub', 'spoke']).size().reset_index(name='cluster_size')
    
    # Optionally filter to top N clusters
    if top_n:
        top_hubs = chart_df.groupby('hub')['cluster_size'].sum().nlargest(top_n).index
        chart_df = chart_df[chart_df['hub'].isin(top_hubs)]
    
    if chart_df.empty:
        return None

    if chart_type == "sunburst":
        fig = px.sunburst(chart_df, path=['hub', 'spoke'], values='cluster_size',
                         color_discrete_sequence=px.colors.qualitative.Pastel2)
    else:
        fig = px.treemap(chart_df, path=['hub', 'spoke'], values='cluster_size',
                        color_discrete_sequence=px.colors.qualitative.Pastel2)
    
    fig.update_layout(margin=dict(t=0, l=0, r=0, b=0))
    return fig

def process_data(df, column_name, model_name, device, min_similarity, stem, volume, exclude_words=None):
    """Process and cluster the keyword data."""
    df = df.copy()
    df.rename(columns={column_name: 'keyword'}, inplace=True)
    df['keyword'] = df['keyword'].astype(str).str.strip().replace(['nan', '', 'None'], 'no_keyword')
    df = df[df["keyword"] != 'no_keyword']
    
    if df.empty:
        st.error("No valid keywords found")
        return pd.DataFrame()

    try:
        # Show progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Load model (cached)
        status_text.text("Loading model...")
        model = PolyFuzz(load_embedding_model(model_name, device))
        progress_bar.progress(20)
        
        # Fit model
        status_text.text("Processing keywords...")
        model.fit(df['keyword'].tolist())
        progress_bar.progress(40)
        
        # Group clusters
        status_text.text("Creating clusters...")
        model.group(link_min_similarity=min_similarity)
        progress_bar.progress(70)
        
        # Get matches
        df_cluster = model.get_matches()
        df_cluster["Group"] = df_cluster["Similarity"].apply(lambda x: "no_cluster" if x < min_similarity else "cluster")
        progress_bar.progress(85)
        
        # Merge results
        df = pd.merge(df, df_cluster.rename(columns={"From": "keyword", "Group": "spoke"})[['keyword', 'spoke']], on='keyword', how='left')
        df['hub'] = df['spoke'].apply(lambda x: create_unigram(x, stem, exclude_words))
        df['hub'] = df['hub'].apply(lambda x: "no_cluster" if x in ["noclust", "nocluster"] else x)
        df.loc[df["hub"] == "no_cluster", "spoke"] = "no_cluster"
        progress_bar.progress(100)
        status_text.text("Processing complete!")
        
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
    st.markdown("""
    Cluster keywords by semantic similarity using Sentence Transformers.
    Upload a CSV file containing your keywords to get started.
    """)
    
    # Add example download
    with st.expander("Need example data?"):
        example_data = pd.DataFrame({
            "Keyword": ["buy iphone", "purchase iphone", "iphone deals", 
                       "best android phone", "top android smartphones",
                       "cheap smartphones", "discount phones"],
            "Volume": [1000, 800, 750, 1200, 950, 500, 400]
        })
        st.dataframe(example_data)
        st.download_button(
            "Download example CSV",
            example_data.to_csv(index=False).encode('utf-8'),
            "example_keywords.csv",
            "text/csv"
        )

    with st.sidebar:
        st.header("Settings")
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
        
        if uploaded_file is not None:
            df = load_file(uploaded_file)
            if df is not None and not df.empty:
                column_name = st.selectbox(
                    "Select keyword column",
                    df.columns,
                    index=next(
                        (i for i, col in enumerate(df.columns) 
                        if col.lower() in [x.lower() for x in COMMON_COLUMN_NAMES]),
                        0
                    ),
                    help="Select the column containing your keywords"
                )
                
                volume = st.selectbox(
                    "Volume column (optional)",
                    [None] + [col for col in df.columns if col != column_name],
                    help="Select a column with volume/metric data if available"
                )
                
                model_name = st.selectbox(
                    "Model",
                    ["all-MiniLM-L6-v2", "paraphrase-MiniLM-L6-v2", "all-mpnet-base-v2"],
                    index=0,
                    help="Choose a language model (larger models are more accurate but slower)"
                )
                
                device = st.selectbox(
                    "Device",
                    ["cpu", "cuda"] if platform.system() != "Darwin" else ["cpu"],
                    index=0,
                    help="Select 'cuda' if you have an NVIDIA GPU for faster processing"
                )
                
                min_similarity = st.slider(
                    "Minimum similarity", 
                    min_value=0.1, 
                    max_value=1.0, 
                    value=0.8, 
                    step=0.05,
                    help="Higher values create more precise but smaller clusters"
                )
                
                chart_type = st.selectbox(
                    "Chart type", 
                    ["treemap", "sunburst"], 
                    index=0
                )
                
                top_n = st.number_input(
                    "Show top N clusters (0 for all)",
                    min_value=0,
                    value=0,
                    step=1,
                    help="Limit visualization to top N clusters by size"
                )
                
                stem = st.checkbox(
                    "Enable stemming", 
                    False,
                    help="Reduce words to their root form (e.g., 'running' → 'run')"
                )
                
                exclude_words = st.text_input(
                    "Words to exclude from hub terms (comma separated)",
                    "",
                    help="Prevent these words from becoming cluster labels"
                )
                exclude_words = [w.strip().lower() for w in exclude_words.split(",") if w.strip()] if exclude_words else None
                
                process_button = st.button("Process Data")

    if uploaded_file is not None and 'process_button' in locals() and process_button:
        with st.spinner('Processing...'):
            processed_df = process_data(
                df, column_name, model_name, device, min_similarity, stem, volume, exclude_words
            )
            
            if not processed_df.empty:
                st.success("Processing completed!")
                
                # Show stats
                cluster_stats = processed_df.groupby('hub').size().sort_values(ascending=False)
                st.write(f"**Cluster Summary:** {len(cluster_stats)} clusters created")
                st.dataframe(cluster_stats.head(10).rename("Count"))
                
                # Show data preview
                with st.expander("View clustered data"):
                    st.dataframe(processed_df.head(100))
                
                # Create and show chart
                fig = create_chart(processed_df, chart_type, volume, top_n if top_n > 0 else None)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No clusters to visualize. Try lowering the similarity threshold.")
                
                # Download button
                st.download_button(
                    "Download clustered data as CSV",
                    processed_df.to_csv(index=False).encode('utf-8'),
                    "clustered_keywords.csv",
                    "text/csv",
                    help="Download the full clustered dataset"
                )
            else:
                st.warning("No valid clusters created. Try lowering the similarity threshold or checking your input data.")

if __name__ == "__main__":
    main()
