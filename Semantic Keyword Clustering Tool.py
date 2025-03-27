import os
import platform
import string
import time
from collections import Counter

import chardet
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import streamlit as st
from polyfuzz import PolyFuzz
from polyfuzz.models import SentenceEmbeddings
from sentence_transformers import SentenceTransformer

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
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Stem the text if the stem flag is True
    if stem:
        from nltk.stem import PorterStemmer
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
        # If there are any words left after filtering, return the most common one
        most_common_word = word_counts.most_common(1)[0][0]
    else:
        # If all words were number-only and thus filtered out, return 'no_keyword'
        most_common_word = 'no_keyword'

    return stem_and_remove_punctuation(most_common_word, stem)

def get_model(model_name: str, device: str):
    """Create and return a SentenceTransformer model based on the given model name."""
    model = SentenceTransformer(model_name, device=device)
    return model

def load_file(uploaded_file):
    """Load a CSV file and return a DataFrame."""
    result = chardet.detect(uploaded_file.getvalue())
    encoding_value = result["encoding"]
    
    try:
        df = pd.read_csv(
            uploaded_file,
            encoding=encoding_value,
            on_bad_lines='skip',
        )
        return df
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

def create_chart(df, chart_type, volume):
    """Create a sunburst chart or a treemap."""
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
        st.error(f"Invalid chart type: {chart_type}. Valid options are 'sunburst' and 'treemap'.")
        return None

    return fig

def process_data(df, column_name, model_name, device, min_similarity, stem, volume):
    """Process the data and return clustered results."""
    df.rename(columns={column_name: 'keyword', "spoke": "spoke Old"}, inplace=True)
    df = df[df["keyword"].notna()]
    df['keyword'] = df['keyword'].astype(str)
    from_list = df['keyword'].to_list()

    embedding_model = SentenceTransformer(model_name, device=device)
    distance_model = SentenceEmbeddings(embedding_model)

    model = PolyFuzz(distance_model)
    model = model.fit(from_list)
    model.group(link_min_similarity=min_similarity)

    df_cluster = model.get_matches()
    df_cluster["Group"] = df_cluster.apply(lambda row: "no_cluster" if row["Similarity"] < min_similarity else row["Group"], axis=1)

    data_dict = df_cluster.groupby('Group')['From'].apply(list).to_dict()

    for group, from_values in data_dict.items():
        if group not in from_values:
            from_values.append(group)

    df_missing = pd.DataFrame([(k, v) for k, vs in data_dict.items() for v in vs], columns=['Group', 'From'])
    df_missing = pd.concat([df_cluster, df_missing], ignore_index=True)
    df_missing['Match'] = df_missing['From'] == df_missing['Group']
    df_missing = df_missing[df_missing["Match"].isin([True])]
    df_missing = df_missing.drop('Match', axis=1)
    df_missing.drop_duplicates(subset=["Group", "From"], keep="first", inplace=True)
    df_missing['Similarity'] = 1
    df_cluster = pd.concat([df_cluster, df_missing])
    df_cluster = df_cluster.sort_values(by="Similarity", ascending=False)
    df_cluster = df_cluster[df_cluster.duplicated(subset=['From'], keep='first') == False]

    df_cluster.rename(columns={"From": "keyword", "Similarity": "similarity", "Group": "spoke"}, inplace=True)
    df = pd.merge(df, df_cluster[['keyword', 'spoke']], on='keyword', how='left')

    df['cluster_size'] = df['spoke'].map(df.groupby('spoke')['spoke'].count())
    df.loc[df["cluster_size"] == 1, "spoke"] = "no_cluster"
    df.insert(0, 'spoke', df.pop('spoke'))
    df['keyword_len'] = df['keyword'].astype(str).apply(len)
    
    if volume is not None and volume in df.columns:
        df[volume] = df[volume].astype(str).replace({'': '0', 'nan': '0'}).str.replace('\D', '', regex=True).astype(int)
        df = df.sort_values(by=volume, ascending=False)
    else:
        df = df.sort_values(by="keyword_len", ascending=True)

    df.insert(0, 'hub', df['spoke'].apply(lambda x: create_unigram(x, stem)))
    df['hub'] = df['hub'].apply(lambda x: stem_and_remove_punctuation(x, stem))
    df = df[['hub', 'spoke', 'cluster_size'] + [col for col in df.columns if col not in ['hub', 'spoke', 'cluster_size']]]

    if volume is not None and volume in df.columns:
        df[volume] = df[volume].replace({'': 0, np.nan: 0}).astype(int)
        df = df.sort_values(by=volume, ascending=False)
    else:
        df['keyword_len'] = df['keyword'].astype(str).apply(len)
        df = df.sort_values(by="keyword_len", ascending=True)

    df['spoke'] = df.groupby('spoke')['keyword'].transform('first')
    df.sort_values(["spoke", "cluster_size"], ascending=[True, False], inplace=True)
    df['spoke'] = (df['spoke'].str.split()).str.join(' ')
    df["hub"] = df["hub"].apply(lambda x: "no_cluster" if x == "noclust" else x)
    df["hub"] = df["hub"].apply(lambda x: "no_cluster" if x == "nocluster" else x)
    df.loc[df["hub"] == "no_cluster", "spoke"] = "no_cluster"
    df.drop(columns=['cluster_size', 'keyword_len'], inplace=True)

    return df

def main():
    st.set_page_config(
        page_title="Keyword Clustering Tool",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("🔍 Semantic Keyword Clustering Tool")
    st.markdown("""
    This tool clusters keywords based on their semantic similarity using Sentence Transformers.
    Upload a CSV file containing your keywords to get started.
    """)

    with st.sidebar:
        st.header("Settings")
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
        
        if uploaded_file is not None:
            df = load_file(uploaded_file)
            if df is not None:
                column_options = df.columns.tolist()
                column_name = st.selectbox(
                    "Select column containing keywords",
                    column_options,
                    index=next((i for i, col in enumerate(column_options) if col.lower() in ['keyword', 'keywords', 'search term', 'search terms'], 0)
                )

                volume_options = [None] + [col for col in column_options if col != column_name]
                volume = st.selectbox(
                    "Select volume column (optional)",
                    volume_options,
                    index=0
                )

                model_name = st.selectbox(
                    "SentenceTransformer model",
                    ["all-MiniLM-L6-v2", "paraphrase-MiniLM-L6-v2", "all-mpnet-base-v2"],
                    index=0
                )

                device = st.selectbox(
                    "Device",
                    ["cpu", "cuda"] if platform.system() != "Darwin" else ["cpu"],
                    index=0
                )

                min_similarity = st.slider(
                    "Minimum similarity threshold",
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

                stem = st.checkbox("Enable stemming for cluster names", value=False)
                remove_dupes = st.checkbox("Remove duplicate keywords", value=True)
                
                if IS_WINDOWS:
                    excel_pivot = st.checkbox("Create Excel pivot table", value=False)
                else:
                    excel_pivot = False

                process_button = st.button("Process Data")

    if uploaded_file is not None and df is not None and 'process_button' in locals():
        if process_button:
            with st.spinner('Processing data... This may take a while depending on the dataset size.'):
                try:
                    start_time = time.time()
                    
                    if remove_dupes:
                        df.drop_duplicates(subset=column_name, inplace=True)
                    
                    processed_df = process_data(
                        df.copy(),
                        column_name,
                        model_name,
                        device,
                        min_similarity,
                        stem,
                        volume
                    )

                    st.success(f"Processing completed in {time.time() - start_time:.2f} seconds!")
                    
                    st.subheader("Preview of Clustered Data")
                    st.dataframe(processed_df.head())

                    st.subheader("Visualization")
                    fig = create_chart(processed_df, chart_type, volume)
                    if fig is not None:
                        st.plotly_chart(fig, use_container_width=True)

                    st.subheader("Download Results")
                    csv = processed_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download as CSV",
                        data=csv,
                        file_name='clustered_keywords.csv',
                        mime='text/csv'
                    )

                    if IS_WINDOWS and excel_pivot:
                        excel_file = processed_df.to_excel(index=False)
                        st.download_button(
                            label="Download as Excel",
                            data=excel_file,
                            file_name='clustered_keywords.xlsx',
                            mime='application/vnd.ms-excel'
                        )

                except Exception as e:
                    st.error(f"An error occurred during processing: {e}")

if __name__ == "__main__":
    main()
