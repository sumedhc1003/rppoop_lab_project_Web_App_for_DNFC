from flask import Flask, render_template, request, redirect, url_for
import plotly
import plotly.graph_objs as go
import plotly.express as px
import json
from wordcloud import WordCloud
import os
import requests

app = Flask(__name__)

API_KEY = "AIzaSyApo5tz7UNU0ppnwIU5tROTQ0XbMinG2qQ"
base_url = "https://www.googleapis.com/youtube/v3/"

def gen_word_cloud(data):
    comments=""
    for item in data["items"]:
        comments+=((item['snippet']['topLevelComment']['snippet']['textDisplay']))
    wordcloud = WordCloud(width = 400, height = 400, 
                background_color ='white', 
                stopwords = None, 
                min_font_size = 10).generate(comments)
    image_path = "static/wordcloud.png"
    wordcloud.to_file(image_path)
    
@app.route('/', methods=['GET', 'POST'])
def home():
    #loading the comments data
    with open('video_comments.json', 'r') as f:
        data = json.load(f)

    #loading sentiment analysis data
    with open("video_sentiment_analysis.json","r") as f:
        sentiments=json.load(f)

    #incase a wordcloud already exists
    file='./static/wordcloud.png'
    if os.path.exists(file):
        os.remove(file)

    #generating wordcloud
    gen_word_cloud(data)

    data1 = [
            go.Pie(
                labels=list(sentiments.keys()),
                values=list(sentiments.values())
            )
        ]
    
    graphJSON1 = json.dumps(data1, cls=plotly.utils.PlotlyJSONEncoder)
  
    return render_template('home.html',graphJSON1=graphJSON1,title="Analytics for the given url",  )

if __name__ == '__main__':
    app.run(debug=True)