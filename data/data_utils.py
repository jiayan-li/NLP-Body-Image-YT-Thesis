import os
import googleapiclient.discovery
import requests
import time
import webbrowser
from datetime import datetime
import joblib
from typing import List, Tuple
from langdetect import detect
import pandas as pd
from transformers import pipeline

### YouTube API Functions ###
FITSPO_Q = "fitspo|fitspiration"
BODY_POSI_Q = "bodypositive|bodypositivity"

def get_yt_service() -> googleapiclient.discovery.Resource:
    """
    Get the YouTube API service
    """

    os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"

    api_service_name = "youtube"
    api_version = "v3"
    DEVELOPER_KEY = 'AIzaSyCPDou4em5yRR4iv5T55CoC7g2ZOvlNCkY'   # Replace with your own API key

    youtube = googleapiclient.discovery.build(
    api_service_name, api_version, developerKey = DEVELOPER_KEY)
    return youtube

def search_yt_videos(keyword: str, published_after: str, published_before: str, num_results: int = 50, youtube=get_yt_service()):
    """
    Search for YouTube videos based on a keyword, published date range, and number of results
    Referrence: https://developers.google.com/youtube/v3/docs/search/list
    """

    request = youtube.search().list(
        part="snippet",     
        maxResults=num_results,   # [0, 50]
        order="viewCount",      
        publishedAfter=published_after,
        publishedBefore=published_before,
        q=keyword,
        relevanceLanguage="en",     # results that are most relevant to the specified language
        regionCode='US',    # Videos can be viewed in the US 
        type="video"    # Only search for videos
    )

    video_response = request.execute()

    return video_response

def get_top_comments(video_id: str, num_results: int = 100, youtube=get_yt_service()) -> dict:
    """
    Search for the top comments of a YouTube video.
    Reference: https://developers.google.com/youtube/v3/docs/commentThreads/list
    """

    request = youtube.commentThreads().list(
        part="snippet,replies",
        order="relevance",      # top comments, time or relevance
        maxResults=num_results,       # 100 is the maximum
        videoId=video_id
    )
    comment_response = request.execute()

    return comment_response

def get_video_details(video_id: str, youtube=get_yt_service()) -> dict:
    """
    Get the details of a YouTube video.
    Reference: https://developers.google.com/youtube/v3/docs/videos/list
    """

    request = youtube.videos().list(
        part="snippet,contentDetails,statistics",
        id=video_id
    )
    video_response = request.execute()

    return video_response

def get_category_mapping(youtube=get_yt_service()) -> pd.DataFrame:
    """
    Get the mapping of category IDs to category names.
    Reference: https://developers.google.com/youtube/v3/docs/videoCategories/list
    """

    try:
        df_category = pd.read_csv('query_results/video_categories.csv')
        return df_category
    except:
        request = youtube.videoCategories().list(
            part="snippet",
            regionCode="US"
        )
        response = request.execute()

        data = []

        for i in response['items']:
            category = {}
            id = i['id']
            title = i['snippet']['title']
            category['id'] = id
            category['title'] = title
            data.append(category)

        df_category = pd.DataFrame(data)
        df_category.to_csv('query_results/video_categories.csv', index=False)

        return df_category


### Information Extraction Functions ###
def extract_pageInfo(year_list: List[Tuple[str, str]], keyword_list: List[str]=[FITSPO_Q, BODY_POSI_Q]) -> pd.DataFrame:
    """
    Extract pageInfo from the video search results saved in the pickle files
    """

    pageInfo_df = pd.DataFrame(columns=['keyword', 'published_after', 'published_before', 'total_results', 'returned_results'])
    
    for year_tup in year_list:
        start_date, end_date = year_tup
        for q in keyword_list:
            path = f'query_results/q_{q}_{start_date}_{end_date}_50.pkl'
            video_response = joblib.load(path)
            pageInfo = video_response['pageInfo']
            pageInfo_df.loc[len(pageInfo_df)] = {'keyword': q, 'published_after': start_date, 'published_before': end_date, 'total_results': pageInfo['totalResults'], 'returned_results': pageInfo['resultsPerPage']}

    # Save the pageInfo to a csv file
    pageInfo_df.to_csv('query_results/pageInfo.csv', index=False)

    return pageInfo_df

def extract_videoInfo(video_dict: dict) -> dict:
    """
    Extract videoInfo from ONE ITEM of the video search query return. Most video search query return has multiple items. 
    """

    video_info_dict = {}
    
    video_info_dict['video_id'] = video_dict['id']['videoId']
    video_info_dict['title'] = video_dict['snippet']['title']
    video_info_dict['description'] = video_dict['snippet']['description']
    video_info_dict['published_at'] = video_dict['snippet']['publishedAt']
    video_info_dict['channel_id'] = video_dict['snippet']['channelId']
    video_info_dict['channel_title'] = video_dict['snippet']['channelTitle']
    video_info_dict['thumbnail_url'] = video_dict['snippet']['thumbnails']['high']['url']

    return video_info_dict


def extract_videoInfo_from_all(year_list: List[Tuple[str, str]], keyword_list: List[str]=[FITSPO_Q, BODY_POSI_Q]) -> pd.DataFrame:
    """
    Extract videoInfo from all the video search results saved in the pickle files
    """

    data = []  # List to store dictionaries

    for year_tup in year_list:
        start_date, end_date = year_tup
        for q in keyword_list:
            path = f'query_results/q_{q}_{start_date}_{end_date}_50.pkl'
            video_response = joblib.load(path)
            for video_dict in video_response['items']:
                video_info = extract_videoInfo(video_dict)
                video_info['keyword'] = q
                video_info['published_after'] = start_date
                video_info['published_before'] = end_date
                data.append(video_info)
            print(f"Extracted video info for {q} from {start_date} to {end_date}")
    
    # Create DataFrame from the list of dictionaries
    video_info_df = pd.DataFrame(data)

    # Save the video_info to a parquet file
    video_info_df.to_parquet('query_results/video_info.parquet', index=False)

    return video_info_df 


def extract_commentInfo(comment_dict: dict) -> dict:
    """
    Extract commentInfo from ONE ITEM of the comment query return. Most comment query return has multiple items.
    """

    comment_info_dict = {}

    comment_info_dict['comment_id'] = comment_dict['snippet']['topLevelComment']['id']
    comment_info_dict['text'] = comment_dict['snippet']['topLevelComment']['snippet']['textOriginal']
    comment_info_dict['comment_published_at'] = comment_dict['snippet']['topLevelComment']['snippet']['publishedAt']
    comment_info_dict['like_count'] = comment_dict['snippet']['topLevelComment']['snippet']['likeCount']
    comment_info_dict['video_id'] = comment_dict['snippet']['topLevelComment']['snippet']['videoId']

    return comment_info_dict


def extract_commentInfo_from_response(comment_response: dict, video_id: str) -> List[dict]:
    """
    Extract commentInfo from ONE comment query return
    """

    data = []  # List to store dictionaries

    for comment_dict in comment_response['items']:
        comment_info = extract_commentInfo(comment_dict)
        comment_info['num_comments'] = comment_response['pageInfo']['totalResults'] # Total number of comments returned, max is 100
        # The video_id in the comment_info should match the video_id passed in
        if comment_info['video_id'] != video_id:
            print(f"Video ID mismatch: {comment_info['video_id']} vs {video_id}")
        data.append(comment_info)

    return data


def extract_videoDetails(video_response: dict) -> dict:
    """
    Extract video details from the video query return.
    """

    video_details_dict = {}

    video_details_dict['description'] = video_response['items'][0]['snippet']['description']
    try:
        video_details_dict['tags'] = video_response['items'][0]['snippet']['tags']
    except:     # Some videos may not have tags
        video_details_dict['tags'] = None
    video_details_dict['category_id'] = video_response['items'][0]['snippet']['categoryId']
    video_details_dict['title'] = video_response['items'][0]['snippet']['title']
    video_details_dict['channel_title'] = video_response['items'][0]['snippet']['channelTitle']
    try:
        video_details_dict['view_count'] = video_response['items'][0]['statistics']['viewCount']
    except:
        video_details_dict['view_count'] = None
    try:
        video_details_dict['like_count'] = video_response['items'][0]['statistics']['likeCount']
    except:
        video_details_dict['like_count'] = None
    try:
        video_details_dict['comment_count'] = video_response['items'][0]['statistics']['commentCount']
    except:
        video_details_dict['comment_count'] = None
    video_details_dict['duration'] = video_response['items'][0]['contentDetails']['duration']
    video_details_dict['definition'] = video_response['items'][0]['contentDetails']['definition']

    return video_details_dict


### Query Functions ###
def query_yt_videos(year_list: List[Tuple[str, str]], keyword_list: List[str]=[FITSPO_Q, BODY_POSI_Q]) -> None:
    """
    Query YouTube videos based on a list of keywords and a list of year tuples
    """

    for year_tup in year_list:
        start_date, end_date = year_tup
        for q in keyword_list:
            video_response = search_yt_videos(q, start_date, end_date)

            # Create the directory if it doesn't exist
            os.makedirs('query_results/raw', exist_ok=True)
            # Save the video response to a pickle file
            joblib.dump(video_response, f'query_results/raw/q_{q}_{start_date}_{end_date}_50.pkl')
            print(f"Search results for {q} from {start_date} to {end_date} saved to pickle file")


def collect_commentInfo(video_id_list: List[str]) -> pd.DataFrame:
    """
    Pipeline to query and collect commentInfo from a list of video_ids.
    """

    data = []  # List to store dictionaries

    n = 0 # Counter to keep track of the number of video_ids processed
    for video_id in video_id_list:
        try:
            comment_response = get_top_comments(video_id)
            comment_info = extract_commentInfo_from_response(comment_response, video_id)
            data.extend(comment_info)
        except:
            print(f"Failed to collect comments for {video_id}")
        n += 1

        # Save the comment_info to a parquet file every 10 video_ids
        if n % 10 == 0:
            print(f"Processed {n} video_ids")
            middle_comment_info_df = pd.DataFrame(data)
            middle_comment_info_df.to_parquet('query_results/middle_comment_info.parquet', index=False)

    # Save final results to a parquet file
    comment_info_df = pd.DataFrame(data)
    comment_info_df.to_parquet('query_results/comment_info.parquet', index=False)

    return comment_info_df


def collect_videoDetails(video_id_list: List[str]) -> pd.DataFrame:
    """
    Pipeline to query and collect videoDetails from a list of video_ids.
    """

    data = []  # List to store dictionaries
    not_collected = []  # List to store video_ids that failed to collect video details

    n = 0 # Counter to keep track of the number of video_ids processed
    for video_id in video_id_list:
        try:
            video_response = get_video_details(video_id)
            video_details = extract_videoDetails(video_response)
            video_details['video_id'] = video_id
            data.append(video_details)
        except:
            not_collected.append(video_id)
            print(f"Failed to collect video details for {video_id}")
        n += 1

        # Save the video_details to a parquet file every 10 video_ids
        if n % 10 == 0:
            print(f"Processed {n} video_ids")
            middle_video_details_df = pd.DataFrame(data)
            middle_video_details_df.to_parquet('query_results/video_details.parquet', index=False)

    # Save final results to a parquet file
    video_details_df = pd.DataFrame(data)
    video_details_df.to_parquet('query_results/video_details.parquet', index=False)

    return video_details_df, not_collected


class LanguageDetector:
    def __init__(self, model_ckpt: str = "papluca/xlm-roberta-base-language-detection"):
        self.model_ckpt = model_ckpt

    def detect_language(self, text_list: List[str]) -> List[str]:
        """
        Detects the language of a list of texts using the XLM-RoBERTa model.
        Reference: https://huggingface.co/papluca/xlm-roberta-base-language-detection
        """
        
        pipe = pipeline("text-classification", model=self.model_ckpt)
        rv = pipe(text_list, top_k=1, truncation=True)

        return rv
    
    def process_row(self, row: pd.Series) -> str:
        """
        Detects the language of a row and returns the row with the detected language.
        """
        description = row["description"]
        description = remove_hashtags(description)
        description = description.strip()
        # if description is empty, use the title
        if description == "":
            text = remove_hashtags(row["title"])
        else:
            text = description

        # detect language
        lang = self.detect_language([text])[0][0]["label"]

        return lang
    
    def process_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detects the language of a dataframe and returns the dataframe with the detected language.
        """
        df["lang"] = df.apply(self.process_row, axis=1)
        return df


### Other Helper Functions ###
def remove_hashtags(text: str) -> str:
    """
    Removes hashtags from a text.
    """
    return " ".join([word for word in text.split() if not word.startswith("#")])

def generate_half_year_tuples(year_list: List[int]) -> List[Tuple[str, str]]:
    """
    Generate a list of tuples of half year start and end dates for each year in the year_list
    """

    half_year_tuples = []
    for year in year_list:
        jan_to_jun_start = datetime(year, 1, 1, 0, 0, 0).strftime("%Y-%m-%dT%H:%M:%SZ")
        jan_to_jun_end = datetime(year, 6, 30, 23, 59, 59).strftime("%Y-%m-%dT%H:%M:%SZ")
        jul_to_dec_start = datetime(year, 7, 1, 0, 0, 0).strftime("%Y-%m-%dT%H:%M:%SZ")
        jul_to_dec_end = datetime(year, 12, 31, 23, 59, 59).strftime("%Y-%m-%dT%H:%M:%SZ")
        half_year_tuples.append((jan_to_jun_start, jan_to_jun_end))
        half_year_tuples.append((jul_to_dec_start, jul_to_dec_end))
    return half_year_tuples


def convert_duration(duration: str) -> int:
    """
    Convert duration from ISO 8601 format to seconds, for instance, PT10M41S to 641 seconds.
    """
    try:
        duration = duration.replace('PT', '')
        if 'M' not in duration:
            duration = '0M' + duration
        if 'H' not in duration:
            duration = '0H' + duration
        if 'S' not in duration:
            duration = duration + '0S'
        duration = duration.replace('PT', '').replace('H', ':').replace('M', ':').replace('S', '')
        duration = duration.split(':')
        duration = list(map(int, duration))
        return duration[0]*3600 + duration[1]*60 + duration[2]
    except:
        return duration

def is_yt_shorts(video_id: str, sleep_time: int = 1) -> bool:
    """
    Check if a video is a YouTube Shorts video
    """
    url = f"https://www.youtube.com/shorts/{video_id}"

    try:
        response = requests.head(url)
        time.sleep(sleep_time)  # Adding sleep delay
        return response.status_code == 200
    except requests.RequestException:
        return False
    
def get_yt_url(video_id: str) -> str:
    return f"https://www.youtube.com/watch?v={video_id}" 

def open_yt_url(video_id: str) -> None:
    url = get_yt_url(video_id)
    webbrowser.open(url)

