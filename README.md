# Real-time Twitter Hashtag Analysis and Sentiment Analyzer

![image](https://user-images.githubusercontent.com/98437584/236606451-143f7e71-3a0d-4a5a-bdf9-0395c1e2ec26.png)

![image](https://user-images.githubusercontent.com/98437584/236606546-ca970b53-1dcc-4b79-a0e8-6ab6ec730136.png)

This project provides a website that allows users to analyze real-time tweets from Twitter based on a specific hashtag. The website includes a tweet sentiment analyzer to determine the sentiment (positive, negative, or neutral) of the collected tweets. Data collection is performed using Snscrape, data cleaning and transformation are done using NumPy and Pandas, data preprocessing utilizes NLTK, sentiment analysis is performed using Textblob, and the application is deployed and hosted on an AWS EC2 instance. The backend is implemented using Django.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Setup](#setup)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Introduction

The Real-time Twitter Hashtag Analysis and Sentiment Analyzer website aims to provide users with a platform to analyze real-time tweets from Twitter based on a specific hashtag. The backend, implemented using Django, handles data collection, cleaning, preprocessing, sentiment analysis, and serves the data to the frontend for visualization and user interaction. The website is deployed and hosted on an AWS EC2 instance for easy access and scalability.

## Features

The Real-time Twitter Hashtag Analysis and Sentiment Analyzer website offers the following features:

- **Hashtag Selection**: Users can input a specific hashtag to analyze tweets related to that hashtag in real-time.
- **Data Collection**: The backend uses Snscrape to collect real-time tweet data based on the selected hashtag.
- **Data Cleaning and Transformation**: NumPy and Pandas are utilized to clean and transform the collected tweet data into a suitable format for analysis.
- **Data Preprocessing**: NLTK (Natural Language Toolkit) is used to preprocess the tweet text data, including tokenization, stopword removal, and stemming.
- **Sentiment Analysis**: The backend employs Textblob, a Python library, to perform sentiment analysis on the preprocessed tweet data, determining the sentiment (positive, negative, or neutral) of each tweet.
- **Interactive Visualizations**: The website generates interactive visualizations, such as word clouds or sentiment distribution charts, to present the analyzed tweet data.
- **User-friendly Interface**: The frontend provides a user-friendly interface for users to input hashtags, view analysis results, and interact with the visualizations.
- **Deployment on AWS EC2**: The website is deployed and hosted on an AWS EC2 instance, providing easy access and scalability.

## Setup

To use this project locally, follow these steps:

1. Clone the repository:

```bash
git clone https://github.com/your-username/twitter-hashtag-analysis.git
```

2. Install the required dependencies by running:

```bash
pip install -r requirements.txt
```

3. Set up an AWS EC2 instance to deploy the application. Make sure to configure the necessary security groups, networking, and storage according to your requirements.

4. Update the necessary configurations in the project files, including Twitter API credentials, AWS EC2 instance details, Django settings, and any other relevant settings.

## Usage

To use the Real-time Twitter Hashtag Analysis and Sentiment Analyzer website, follow these steps:

1. Start the Django development server using the provided command:

```bash
python manage.py runserver
```

2. Access the website by opening the provided URL in your web browser.

3. Enter the desired hashtag in the input field to start analyzing real-time tweets related to that hashtag.

4. Explore the visualizations and sentiment analysis results displayed on the website.

5. Customize the application as needed by modifying the code, adding more visualizations, or incorporating additional analysis techniques.

6. Monitor the AWS EC2 instance
