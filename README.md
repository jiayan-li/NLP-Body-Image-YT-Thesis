# Body Positivity vs. Fitspiration: Comparing the Video Content and Comments on YouTube

## Overview

This repository contains the data, code, and analysis for the thesis titled "Body Positivity vs. Fitspiration: Comparing the Video Content and Comments on YouTube" by Jiayan Li. This study explores the differences and similarities between body positivity and fitspiration trends on YouTube, focusing on video content, popularity, and viewer comments.

## Abstract

The emergence of social media has revolutionized the discourse surrounding body image, particularly through trends like fitspiration and body positivity. This study addresses the gap in research by directly comparing these trends on YouTube. It aims to provide insights into their popularity, content themes, and sentiment in comments.

## Data Collection

Data was gathered from YouTube using the YouTube API, covering videos published between 2013 and 2023. The dataset includes:
- Video titles
- Descriptions
- User-defined content tags
- Comments
- Contextual data such as view count, comment count, duration, and category

## Methods

A mixed-methods approach was employed to analyze various aspects of video content and comments:
1. **Descriptive Analysis**: Examined sample volume, view count, comment count, and video duration.
2. **Cluster Analysis**: Evaluated cohesion within each trend and distinction between trends using video categories and frequently used hashtags.
3. **Sentiment Analysis**: Used pre-trained Large Language Models (LLMs) to analyze the sentiment of comments.
4. **Thematic Analysis**: Identified prominent themes present in the comments.

## Results

### Popularity and Duration Comparison
- Body positivity videos have surged in popularity, particularly from 2021 to 2023, and have longer average durations compared to fitspiration videos.

### Topic Comparison
- "People & Blogs" is the most prominent category for both topics. Fitspiration videos have a stronger presence in the "Sports" category, while body positivity videos are more prevalent in "Howto & Style" and "Entertainment."

### Sentiment and Themes in Comments
- The majority of comments under both trends exhibited positive sentiments. However, body positivity videos had a higher proportion of negative comments. Common themes include gratitude for content, discussions about physical appearance, and personal life stories.

## Repository Structure

- `data/`: Contains raw and processed data files and code used for data collection.
- `analysis.ipynb` and `analysis.py`: Code and results for data analysis.
- `README.md`: This file.

## Requirements

- Python 3.8 or higher
- Jupyter Notebook
- Libraries: pandas, numpy, scikit-learn, transformers, matplotlib, seaborn

## Installation

Clone the repository and install the required libraries:

```bash
git clone https://github.com/username/thesis-body-positivity-vs-fitspiration.git
cd thesis-body-positivity-vs-fitspiration
pip install -r requirements.txt
```