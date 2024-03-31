from langdetect import detect
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from collections import Counter
from typing import List
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.util import ngrams
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Optional
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

### Constants ###
FITSPO_Q = "fitspo|fitspiration"
BODY_POSI_Q = "bodypositive|bodypositivity"
keyword_color_mapping = {FITSPO_Q: sns.color_palette("pastel")[0], BODY_POSI_Q: sns.color_palette("pastel")[1]}
lemmatizer = WordNetLemmatizer()

### Helper functions starts ###
def extract_hashtags(text: str) -> List[str]:
    """
    Extracts hashtags from a text.
    """
    return [word for word in text.split() if word.startswith("#")]

def clean_text(text: str) -> List[str]:
    """
    Clean the text.
    """
    # Convert text to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove special characters and digits
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    
    # Tokenize text
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    
    # Lemmatize words
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
    
    return lemmatized_tokens

def get_bigrams(cleaned_tokens: List[str]) -> List[tuple[str]]:
    """
    Get bigrams from a list of cleaned tokens.
    """
    bigrams = list(tuple(ngrams(cleaned_tokens, 2)))
    return bigrams
### Helper functions ends ###


class TagPreprocessor:
    """
    Preprocesses the column "tags" or "hashtags" in a dataframe.
    """
    def __init__(self) -> None:
        self.fitspo_kw = FITSPO_Q
        self.body_posi_kw = BODY_POSI_Q

    def get_tags(self, df: pd.DataFrame, keyword: str, column_name: str) -> List[np.ndarray[str]]:
        """
        Get a list of tags for the specified keyword.
        """

        assert keyword in [self.fitspo_kw, self.body_posi_kw], \
        "Keyword must be either fitspo or body positive query keyword."

        assert column_name in ['hashtags', 'tags'], \
        "Column name must be either hashtags or tags."
        
        tags = df[df['keyword']==keyword][column_name].to_list()
        return tags
    
    def flatten_list(self, lst: List[np.ndarray[str]], lowercase: bool = True):
        """
        Flatten a list of numpy arrays of strings, lowercase the strings.
        """
        
        rv = []
        for sublist in lst:
            if sublist is not None:
                for item in sublist:
                    if lowercase:
                        item = item.lower()
                        rv.append(item)
        return rv
    
    def get_tag_lists(self, df: pd.DataFrame, column_name: str, deduplicate: bool) -> dict[str: List[str]]:
        """
        Get a dictionary of tag lists for fitspo and body positive.
        Arg deduplicate: if True, return a deduplicated list of tags.
        """

        fitspo_tags = self.get_tags(df, self.fitspo_kw, column_name)
        fitspo_tags = self.flatten_list(fitspo_tags)
        
        body_posi_tags = self.get_tags(df, self.body_posi_kw, column_name)
        body_posi_tags = self.flatten_list(body_posi_tags)

        if deduplicate:
            fitspo_tags = list(set(fitspo_tags))
            body_posi_tags = list(set(body_posi_tags))

        return {self.fitspo_kw: fitspo_tags, self.body_posi_kw: body_posi_tags}

    def get_tag_counts(self, df: pd.DataFrame, column_name: str, top_n = 10) -> dict[str: Counter]:
        """
        Get the counts of tags for fitspo and body positive.
        """

        # get tag lists
        tag_list_dict = self.get_tag_lists(df, column_name, deduplicate=False)

        # get counts
        fitspo_tag_counts = Counter(tag_list_dict[self.fitspo_kw])
        body_posi_tag_counts = Counter(tag_list_dict[self.body_posi_kw])

        print(f"Top {top_n} {column_name} for fitspo: {fitspo_tag_counts.most_common(top_n)}")
        print(f"Top {top_n} {column_name} for body positive: {body_posi_tag_counts.most_common(top_n)}")

        return {self.fitspo_kw: fitspo_tag_counts, self.body_posi_kw: body_posi_tag_counts}
    
    def plot_tag_counts(self, df: pd.DataFrame, column_name: str, top_n: int = 10):
        """
        Plot the counts of tags for fitspo and body positive.
        """

        # get tag counts
        tag_count_dict = self.get_tag_counts(df, column_name, top_n)

        # Combine the top 10 hashtags and their counts for fitspo and body positivity
        fitspo_top_hashtags = [x[0] for x in tag_count_dict[FITSPO_Q].most_common(top_n)]
        fitspo_counts = [x[1] for x in tag_count_dict[FITSPO_Q].most_common(top_n)]

        body_posi_top_hashtags = [x[0] for x in tag_count_dict[BODY_POSI_Q].most_common(top_n)]
        body_posi_counts = [x[1] for x in tag_count_dict[BODY_POSI_Q].most_common(top_n)]

        # Create a DataFrame for Seaborn
        df = pd.DataFrame({
            'hashtag': body_posi_top_hashtags + fitspo_top_hashtags,
            'count': body_posi_counts + fitspo_counts,
            'keyword': [BODY_POSI_Q] * 10 + [FITSPO_Q] * 10
        })

        keyword_colors = [keyword_color_mapping[keyword] for keyword in df['keyword'].unique()]
        # Set Seaborn style
        sns.set(style="whitegrid")
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(x='count', y='hashtag', hue='keyword', data=df, ax=ax, palette=keyword_colors, linewidth=2)
        ax.set_xlabel('Frequency Count')
        ax.set_ylabel('Hashtag')
        ax.set_title('Top 10 Hashtags for Fitspo and Body Positivity')

        return tag_count_dict
    
    
class HashtagAnalyzer(TagPreprocessor):
    def __init__(self, column_name: str = "hashtags"):
        super().__init__()
        self.column_name = column_name

    def process_row(self, row: pd.Series) -> List[str]:
        """
        Extracts hashtags from a row. Hashtags are extracted from the title and description.
        """

        hashtags = []
        description = row["description"]
        hashtags.extend(extract_hashtags(description))
        title = row["title"]
        hashtags.extend(extract_hashtags(title))
        
        return hashtags
    
    def process_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extracts hashtags from a dataframe.
        """
        df["hashtags"] = df.apply(self.process_row, axis=1)
        # remove empty lists for NaN
        df["hashtags"] = df["hashtags"].apply(lambda x: x if len(x) > 0 else None)
        return df
    
    def plot_hashtag_counts(self, df: pd.DataFrame, top_n: int = 10):
        """
        Plot the counts of hashtags for fitspo and body positive.
        """

        hashtag_count_dict = self.plot_tag_counts(df, self.column_name, top_n)
        plt.savefig('data/results/top_hashtags.png', bbox_inches='tight', dpi=300)

        return hashtag_count_dict

    
class ClusterAnalyzer(TagPreprocessor):
    def __init__(self, model_ckpt: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        super().__init__()
        self.model = SentenceTransformer(model_ckpt)
    
    def get_embedding(self, text_list: List[str]):
        """
        Get the embeddings of a list of texts.
        """

        embeddings = self.model.encode(text_list)
        return embeddings

    def get_embeddings_dict(self, df: pd.DataFrame) -> dict[str: np.ndarray]:
        """
        Get the embeddings of 2 groups of tags.
        """ 

        # get tag lists
        tag_lists = self.get_tag_lists(df, deduplicate=True)

        # get embeddings
        fitspo_embeddings = self.get_embedding(tag_lists[self.fitspo_kw])
        body_posi_embeddings = self.get_embedding(tag_lists[self.body_posi_kw])

        # save to local
        np.save('data/preprocessed_data/fitspo_embeddings.npy', fitspo_embeddings)
        np.save('data/preprocessed_data/body_posi_embeddings.npy', body_posi_embeddings)

        return {self.fitspo_kw: fitspo_embeddings, self.body_posi_kw: body_posi_embeddings}

    def compute_group_centroids(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Compute the centroid of a group of embeddings.
        """

        return np.mean(embeddings, axis=0)
    
    def compute_intra_group_distance(self, embeddings: np.ndarray, centroid: np.ndarray) -> float:
        """
        Compute the intra-group distance of a group of embeddings.
        """

        return np.mean(np.linalg.norm(embeddings - centroid, axis=1))
    
    def compute_inter_group_distance(self, centroid1: np.ndarray, centroid2: np.ndarray) -> float:
        """
        Compute the inter-group distance of 2 centroids.
        """

        return np.linalg.norm(centroid1 - centroid2)
    
    def evaluate_group_distance(self, embeddings_dict: dict[str: np.ndarray]) -> dict[str: float]:
        """
        Evaluate the distance between the 2 groups of embeddings.
        """

        fitspo_embeddings = embeddings_dict[self.fitspo_kw]
        body_posi_embeddings = embeddings_dict[self.body_posi_kw]

        # compute centroids
        fitspo_centroid = self.compute_group_centroids(fitspo_embeddings)
        body_posi_centroid = self.compute_group_centroids(body_posi_embeddings)

        # compute intra-group distance
        fitspo_intra_group_distance = self.compute_intra_group_distance(fitspo_embeddings, fitspo_centroid)
        body_posi_intra_group_distance = self.compute_intra_group_distance(body_posi_embeddings, body_posi_centroid)

        # compute inter-group distance
        inter_group_distance = self.compute_inter_group_distance(fitspo_centroid, body_posi_centroid)

        if fitspo_intra_group_distance < body_posi_intra_group_distance:
            print("Fitspo group has tighter clustering.")
        else:
            print("Body positive has tighter clustering.")

        if inter_group_distance > fitspo_intra_group_distance + body_posi_intra_group_distance:
            print("Groups are well separated.")

        return {self.fitspo_kw: fitspo_intra_group_distance, 
                self.body_posi_kw: body_posi_intra_group_distance, 
                "inter_group_distance": inter_group_distance}
    
    def combine_embeddings(self, embeddings_dict: dict[str: np.ndarray]) -> np.ndarray:
        """
        Combine the embeddings of 2 groups.
        """

        body_posi_embeddings = embeddings_dict[self.body_posi_kw]
        fitspo_embeddings = embeddings_dict[self.fitspo_kw]

        combined_embeddings = np.vstack((body_posi_embeddings, fitspo_embeddings))
        
        # Create combined labels (0 for body positivity, 1 for fitspo)
        labels_A = np.zeros(len(body_posi_embeddings))
        labels_B = np.ones(len(fitspo_embeddings))
        combined_labels = np.concatenate((labels_A, labels_B))

        return combined_embeddings, combined_labels
        
    def compute_silhouette_score(self, embeddings_dict: dict[str: np.ndarray]) -> tuple[float, np.ndarray]:
        """
        Compute the silhouette score for two groups of embeddings.
        The best value is 1 and the worst value is -1. Values near 0 indicate overlapping clusters.
        """

        # Combine embeddings
        combined_embeddings, combined_labels = self.combine_embeddings(embeddings_dict)

        # Compute silhouette samples for combined embeddings
        silhouette_samples_values = silhouette_samples(combined_embeddings, combined_labels)
        
        # Compute silhouette score
        silhouette_avg = silhouette_score(combined_embeddings, combined_labels)
        
        return silhouette_avg, silhouette_samples_values
    
    def process_silhouette_vals(self, silhouette_samples_values: np.ndarray, tag_list_dict: dict) -> dict[str: pd.DataFrame]:
        """
        Process silhouette samples values.
        """

        body_posi_silhouette_vals = silhouette_samples_values[:len(tag_list_dict[self.body_posi_kw])]
        fitspo_silhouette_vals = silhouette_samples_values[len(tag_list_dict[self.body_posi_kw]):]

        # put the silhouette_samples_values into a dataframe
        df_silhouette_fitspo = pd.DataFrame(fitspo_silhouette_vals, columns=['silhouette_score'], 
                                            index=tag_list_dict[self.fitspo_kw])
        df_silhouette_body_posi = pd.DataFrame(body_posi_silhouette_vals, columns=['silhouette_score'], 
                                               index=tag_list_dict[self.body_posi_kw])

        return {self.fitspo_kw: df_silhouette_fitspo, self.body_posi_kw: df_silhouette_body_posi}
    
    def run_tsne(self, embeddings_dict: dict[str: np.ndarray], rerun: bool = False):
        """
        Run t-SNE on the combined embeddings.
        """
        # Combine embeddings, body posi is 0, fitspo is 1 
        combined_embeddings, combined_labels = self.combine_embeddings(embeddings_dict)

        if not rerun:
            try:
                embeddings_2d = joblib.load('data/preprocessed_data/tsne_embeddings_2d.pkl')
                return embeddings_2d, combined_labels
            except FileNotFoundError:
                pass

        # Perform t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        embeddings_2d = tsne.fit_transform(combined_embeddings)
        joblib.dump(embeddings_2d, 'data/preprocessed_data/tsne_embeddings_2d.pkl')

        return embeddings_2d, combined_labels

    def plot_clusters(self, embeddings_dict: dict[str: np.ndarray], rerun: bool = False):
        """
        Plot clusters using t-SNE.
        """

        # Run t-SNE
        embeddings_2d, combined_labels = self.run_tsne(embeddings_dict, rerun)

        # Create DataFrame for Seaborn
        df = pd.DataFrame({'x': embeddings_2d[:, 0], 'y': embeddings_2d[:, 1], 
                           'label': [BODY_POSI_Q if label == 0 else FITSPO_Q for label in combined_labels]})
        
        # Sample a subset of the data for visualization
        df_sampled = df.sample(frac=0.5, random_state=0)

        sns.set(style="whitegrid")
        keyword_colors = [keyword_color_mapping[keyword] for keyword in df_sampled['label'].unique()]

        plt.figure(figsize=(10, 5))
        sns.scatterplot(x='x', y='y', hue='label', data=df_sampled, 
                        palette=keyword_colors, linewidth=0.5, edgecolor='k', alpha=0.7, s=20)
        
        plt.title('t-SNE Visualization of Clusters')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.legend(title='Cluster', loc='upper right')

        plt.savefig('data/results/clusters_visualization.png')
        plt.show()

class CommentPreprocessor:
    def __init__(self, 
                 raw_data_path: str = 'data/query_results/comment_info.parquet', 
                 target_data_path: str = 'data/preprocessed_data/comment_info.parquet',
                 video_info_path: str = 'data/preprocessed_data/video_info_details.parquet'):
        self.raw_data_path = raw_data_path
        self.target_data_path = target_data_path
        self.video_info_path = video_info_path

    def process_df(self, rerun: bool = False, save: bool = False) -> pd.DataFrame:
        """
        Add keyword and language info to the comment dataframe.
        """

        if not rerun:
            df = pd.read_parquet(self.target_data_path)
            # check if the dataframe has been processed
            if 'keyword' in df.columns:
                return df

        # Load data
        df = pd.read_parquet(self.raw_data_path)
        df = df[df['text'] != '']  # remove empty comments
        video_info = pd.read_parquet(self.video_info_path)

        # Add keyword info
        df = df.merge(video_info[['video_id', 'keyword', 'lang']], on='video_id', how='left')
        if save:
            df.to_parquet(self.target_data_path, index=False)

        return df
    
    def plot_comment_distribution(self, df: pd.DataFrame) -> None:
        """
        Plot the distribution of number of comments per video for each keyword.
        """

        keyword_colors = [keyword_color_mapping[keyword] for keyword in df['keyword'].unique()]

        sns.set(style="whitegrid")
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.kdeplot(data=df[df['keyword'] == BODY_POSI_Q]['count'], label=BODY_POSI_Q, color=keyword_colors[0], fill=True, bw_adjust=0.5)
        sns.kdeplot(data=df[df['keyword'] == FITSPO_Q]['count'], label=FITSPO_Q, color=keyword_colors[1], fill=True, bw_adjust=0.5)

        ax.set_xlim(0, 100)
        ax.set_xlabel('Number of Comments')
        ax.set_ylabel('Density')
        ax.set_title('Distribution of Comments per Video')
        ax.legend(title='Keyword')
        plt.savefig('data/results/comment_distribution.png', bbox_inches='tight', dpi=300)

    def get_comment_distribution(self, plot: bool = True) -> pd.DataFrame:
        """
        Get the distribution of number of comments per video for each keyword.
        """

        df_comments = pd.read_parquet(self.target_data_path)
        df = df_comments.groupby(['keyword', 'video_id']).size().reset_index(name='count')

        if plot:
            self.plot_comment_distribution(df)

        return df
    
    def clean_comments_df(self, df: pd.DataFrame = None, rerun: bool = False) -> pd.DataFrame:
        """
        Clean the English comments in a dataframe. Run this function before adding bigrams.
        """

        if df is None:
            df = pd.read_parquet(self.target_data_path)
            df = df[df['lang'] == 'en']

        if not rerun:
            # check if comments have been cleaned
            if 'lemmatized_tokens' in df.columns:
                return df
        
        df['lemmatized_tokens'] = df['text'].apply(lambda x: clean_text(x))
        df['bigrams'] = df['lemmatized_tokens'].apply(lambda x: get_bigrams(x))

        # Save to local
        df.to_parquet(self.target_data_path, index=False)
        return df


class SentimentAnalyzer(CommentPreprocessor):
    """
    Sentiment analysis of original texts of comments.
    """
    def __init__(self, model_ckpt: str = 'cardiffnlp/twitter-roberta-base-sentiment-latest'):
        super().__init__()
        self.model_ckpt = model_ckpt

    def get_sentiment(self, text: str) -> List[dict]:
        """
        Get the sentiment of a string.
        Reference: https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest
        """

        sentiment_task = pipeline("sentiment-analysis", 
                                  model=self.model_ckpt, 
                                  tokenizer=self.model_ckpt, 
                                  truncation="longest_first",  # truncate the input if it is too long
                                  max_length=512)
        rv = sentiment_task(text)
        sentiment_label = rv[0]['label']
        sentiment_score = rv[0]['score']
        return sentiment_label, sentiment_score

    def add_sentiment(self, df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Process the comments in a dataframe. The result only contains English comments.
        """
        
        if df is None:
            df = pd.read_parquet(self.target_data_path)

            # check if the dataframe has been processed
            if 'lang' not in df.columns:
                df = self.process_df()
                # Filter out comments of non-English videos
                df = df[df['lang'] == 'en']

        for i, row in df.iterrows():
            # Register progress
            if i % 1000 == 0:
                print(f"Processing row {i}...")
            # Use the original text for sentiment analysis
            comment = row['text']
            try:
                sentiment_label, sentiment_score = self.get_sentiment(comment)
                df.at[i, 'sentiment_label'] = sentiment_label
                df.at[i, 'sentiment_score'] = sentiment_score
            except Exception as e:
                print(f"Error: {e}")
                df.at[i, 'sentiment_label'] = None
                df.at[i, 'sentiment_score'] = None

        # Save to local
        df.to_parquet(self.target_data_path, index=False)
        return df
    
    def plot_sentiment_distribution(self, df_comments_summary: pd.DataFrame) -> None:
        """
        Plot the distribution of sentiment scores for each keyword.
        """
        
        # bar plot of the number of videos returned for each keyword
        keyword_colors = [keyword_color_mapping[keyword] for keyword in df_comments_summary['keyword'].unique()]
        sns.set(style="whitegrid")
        fig, ax = plt.subplots(figsize=(10, 5))
        # specify the order of the bars
        order = ['positive', 'neutral', 'negative']
        sns.barplot(x='sentiment_label', y='normalized_count', hue='keyword', ax=ax, data=df_comments_summary, palette=keyword_colors, order=order, linewidth=2.0)
        ax.set_title('Sentiment Distribution for Each Keyword')
        ax.set_xlabel('Sentiment Label')
        ax.set_ylabel('Normalized Count')
        ax.set_xticklabels(['Positive', 'Neutral', 'Negative'])
        ax.legend(title='Keyword')
        plt.savefig('data/results/sentiment_distribution.png', bbox_inches='tight', dpi=300)

    def get_sentiment_summary(self, df: pd.DataFrame = None, plot: bool = True) -> pd.DataFrame:
        """
        Get the summary of sentiment scores for each keyword.
        """

        if df is None:
            df = pd.read_parquet(self.target_data_path)

        # Get the summary of sentiment scores for each keyword
        df_comments_summary = df.groupby('keyword')['sentiment_label'].value_counts(normalize=True)
        # Reset index to convert the groupby result to a DataFrame
        df_comments_summary = df_comments_summary.reset_index(name='normalized_count')

        # Rename the columns for clarity
        df_comments_summary.columns = ['keyword', 'sentiment_label', 'normalized_count']

        # Save to local
        df_comments_summary.to_csv('data/results/sentiment_summary.csv', index=False)

        if plot:
            self.plot_sentiment_distribution(df_comments_summary)

        return df_comments_summary
    

class TokenAnalyzer(CommentPreprocessor):
    def __init__(self):
        super().__init__()
    
    def calculate_tf_idf_ngrams(self, ngram: int, ngram_list: List) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate the TF-IDF of bigrams or unigrams.
        """
        
        if ngram == 1:
            vectorizer = TfidfVectorizer(
                analyzer='word',  
                ngram_range=(1, 1),  # Use unigrams
            )
            ngram_texts = ngram_list
        else:
            vectorizer = TfidfVectorizer(
                analyzer='word', 
                ngram_range=(2, 2),  # Use bigrams
            )
            ngram_texts = [' '.join(ngram) for ngram in ngram_list]
        
        # Fit the vectorizer to the data and transform the data into TF-IDF matrix
        tfidf_matrix = vectorizer.fit_transform(ngram_texts)
        
        # Get feature names (bigrams)
        feature_names = vectorizer.get_feature_names_out()
        
        return tfidf_matrix, feature_names
    
    def plot_top_ngrams(self, keyword: str, df_ngrams: pd.DataFrame, ngram: int, measure: str = 'tfidf_score', top_n: int = 10) -> None:
        """
        Plot the top ngrams for each keyword.
        """

        assert measure in ['tfidf_score', 'count'], \
        "Measure must be either tfidf_score or count."

        ngram_dict = {1: 'Unigrams', 2: 'Bigrams'}
        measure_dict = {'tfidf_score': 'TF-IDF Score', 'count': 'Count'}
        
        df_ngrams = df_ngrams.sort_values(measure, ascending=False).reset_index(drop=True)
        
        # Set Seaborn style
        sns.set(style="whitegrid")
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(x=measure, y='ngram', data=df_ngrams.head(top_n), ax=ax, color=keyword_color_mapping[keyword], linewidth=2.0)
        ax.set_title(f'Top {top_n} {ngram_dict[ngram]} for {keyword}')
        ax.set_xlabel(measure_dict[measure])
        ax.set_ylabel(ngram_dict[ngram])
        plt.savefig(f'data/results/top_{top_n}_{ngram_dict[ngram]}_{keyword}_{measure}.png', bbox_inches='tight', dpi=300)
        plt.show()

    def get_top_ngrams(self, ngram: int, keyword: str, df: pd.DataFrame = None, plot: bool = True) -> pd.DataFrame:
        """
        Get the top unigrams/bigrams for each keyword.
        """

        assert keyword in [FITSPO_Q, BODY_POSI_Q], \
        "Keyword must be either fitspo or body positive query keyword."

        assert ngram in [1, 2],\
        "Ngram must be either 1 or 2."

        if df is None:
            df = pd.read_parquet(self.target_data_path)
            df = self.process_df()
            df = self.clean_comments_df()
        
        if ngram == 2:
            # Get bigrams
            ngram_list = df[df['keyword'] == keyword]['bigrams'].to_list()
            ngram_list = [tuple(item) for sublist in ngram_list for item in sublist]   # a list of bigrams
        else:
            # Get lemmatized tokens
            ngram_list = df[df['keyword'] == keyword]['lemmatized_tokens'].to_list()
            ngram_list = [item for sublist in ngram_list for item in sublist]   # a list of unigrams
        
        # Calculate TF-IDF of bigrams
        tfidf_matrix, feature_names = self.calculate_tf_idf_ngrams(ngram, ngram_list)

        # Get ngram counts
        ngram_counts = Counter(ngram_list)
        
        if ngram == 2:
            count = [ngram_counts[tuple(bigram.split())] for bigram in feature_names]
        else:
            count = [ngram_counts[unigram] for unigram in feature_names]

        # Get the TF-IDF score of each unigram/bigram
        tfidf_scores = tfidf_matrix.toarray().mean(axis=0)

        # Create a DataFrame for Seaborn
        df_ngrams = pd.DataFrame({'ngram': feature_names, 'tfidf_score': tfidf_scores, 'count': count})

        # Save to local
        df_ngrams.to_parquet(f'data/results/top_{ngram}_grams_{keyword}.parquet', index=False)

        if plot:
            for measure in ['tfidf_score', 'count']:
                self.plot_top_ngrams(keyword, df_ngrams, ngram, measure=measure)

        return df_ngrams
    
class TopicClassifier(CommentPreprocessor):
    def __init__(self, 
                 topic_results_path: str = 'data/preprocessed_data/comment_topic_results.parquet',
                 model_ckpt: str = 'facebook/bart-large-mnli',
                 comment_candidate_labels: List[str] = ['Gratitude or Appreciation for Content',
                                                        'Physical Appearance or Body Shape/Parts',
                                                        'Weight Loss/Body Transformation',
                                                        'Dieting/Calorie Count',
                                                        'Body Functionality or Physical Performance',
                                                        'Encouragement of Body Acceptance',
                                                        'Conceptualize Beauty Broadly', 
                                                        'Personal Life Stories',
                                                        'Advertisement or Commercialism',
                                                        'Motivational Quotes']):
        """
        Reference: https://huggingface.co/facebook/bart-large-mnli
        """
        super().__init__()
        self.topic_results_path = topic_results_path    # path to save the final topic classification results
        self.model_ckpt = model_ckpt
        self.comment_candidate_labels = comment_candidate_labels

    def classify_text(self, text: str, multi_label: bool = False) -> dict:
        """
        Classify the text.
        """

        classifier = pipeline("zero-shot-classification", model=self.model_ckpt)
        rv = classifier(text, self.comment_candidate_labels, multi_label=multi_label)
        return rv

    def add_comment_topic(self, df: pd.DataFrame = None, 
                          source_path: Optional[str] = None, 
                          cache_path: str = None,
                          save: bool = False    
                          ) -> pd.DataFrame:
        """
        Add the topic of each comment to the dataframe.
        """

        # prevent overwriting the original dataframe
        if save and cache_path is None:
            raise ValueError("Please specify the cache path to save the results of this run.")

        if df is None:
            try:
                if source_path is None:
                    source_path = self.target_data_path
                # use cached results
                df = pd.read_parquet(source_path)
            except:
                # make sure the 'keyword' and 'lang' columns have been added to comments dataframe
                df = self.process_df(rerun=False, save=False)
                # only process English comments
                df = df[df['lang'] == 'en']

        if cache_path is None:
            cache_path = self.topic_results_path

        # add the columns for the candidate labels
        for label in self.comment_candidate_labels:
            # if the label column already exists
            # don't overwrite the existing values
            if label in df.columns:
                break
            else:
                df[label] = None

        for i, row in df.iterrows():
            # Register progress
            if i % 100 == 0:
                print(f"Processing row {i}...")
                if save:
                    # save the progress
                    df.to_parquet(cache_path, index=False)
            else:
                pass

            if pd.isna(row[self.comment_candidate_labels[0]]):
                # Use the original text for topic classification
                comment = row['text']
                try:
                    rv = self.classify_text(comment, multi_label=False)
                    for label in self.comment_candidate_labels:
                        df.at[i, label] = rv['scores'][rv['labels'].index(label)]
                except Exception as e:
                    print(f"Error: {e}")
                    for label in self.comment_candidate_labels:
                        df.at[i, label] = None
            else:
                pass    # skip the row if the candidate labels have been added
        
        if save:
            # Save the final result to local
            df.to_parquet(cache_path, index=False)

        return df
    