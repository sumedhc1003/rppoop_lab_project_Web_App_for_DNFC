import requests
import json
import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser
from langchain.prompts import ChatPromptTemplate
from transformers import pipeline
from langchain.chains import SequentialChain, LLMChain
from dotenv import load_dotenv
import re
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Load environment variables from the .env file
load_dotenv()

st.title("Comment Analysis of YouTube videos")

API_KEY = os.environ.get("YOUTUBE_DATA_API")
api_key = os.environ.get("GOOGLE_API_KEY")
base_url = "https://www.googleapis.com/youtube/v3/"


#function for fetching the comments"""

def get_comments(video_id):
  endpoint = "commentThreads"
    

  params = {
      "key": API_KEY,
      "part": "snippet",
      "video_id":video_id,
      "maxResults":50,
  }

  response = requests.get(base_url + endpoint, params=params)

  if response.status_code == 200:
    data = response.json()
    print(f"Successfully retrieved {len(data['items'])} comments.")
  else:
      print(f"Error fetching comments: {response.status_code}")
  data = response.json()
  return(data)


def is_marathi(comment):
    # Regular expression pattern to match Marathi characters
    marathi_pattern = re.compile(r'[\u0900-\u097F]')  # Range for Marathi Unicode characters

    # Check if the comment contains Marathi characters
    if marathi_pattern.search(comment):
        return True
    else:
        return False

def separate_comments(comments):
    english_comments = []
    marathi_comments = []

    for comment in comments:
        if is_marathi(comment):
            marathi_comments.append(comment)
        else:
            english_comments.append(comment)

    return {
        "english_comments": english_comments,
        "marathi_comments": marathi_comments
    }   

#sentiment analysis: Both english and Marathi
def sentiment_analysis(output_dict):
    checkpoint = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    sentiment_task = pipeline("sentiment-analysis", model=checkpoint, tokenizer=checkpoint)

    # Define batch size (experiment to find optimal value)
    batch_size = 5

    # for marathi
    english_sentiment_results = []
    for i in range(0, len(output_dict['english_comments']), batch_size):
        batch = output_dict['english_comments'][i:i+batch_size]
        batch_results = sentiment_task(batch)
        english_sentiment_results.extend(batch_results)

    #marathi sentiment analysis
    checkpoint = "l3cube-pune/marathi-sentiment-md"
    marathi_sentiment_task = pipeline("sentiment-analysis", model=checkpoint, tokenizer=checkpoint)
    
    marathi_sentiment_analysis = []
    for i in range(0, len(output_dict['marathi_comments']), batch_size):
        batch = output_dict['marathi_comments'][i:i+batch_size]
        batch_results = marathi_sentiment_task(batch)
        marathi_sentiment_analysis.extend(batch_results)

    #counting the sentiments
    positive_cmts = 0
    neg_cmts = 0
    neut_cmts = 0
    
    #for english comments
    for sentiment in english_sentiment_results:
      if sentiment['label'] == 'positive':
        positive_cmts += 1
      elif sentiment['label'] == 'negative':
        neg_cmts += 1
      else:
        neut_cmts += 1
    
    #for marathi comments
    for sentiment in marathi_sentiment_analysis:
      if sentiment['label'] == 'positive':
        positive_cmts += 1
      elif sentiment['label'] == 'negative':
        neg_cmts += 1
      else:
        neut_cmts += 1

    #appending the result
    vid_to_sentiment = {"pos": positive_cmts, "neg": neg_cmts, "neut": neut_cmts}
    
    # Write sentiment analysis results to a JSON file
    with open("video_sentiment_analysis.json", "w") as file:
        json.dump(vid_to_sentiment, file)
    
    print("Sentiment analysis completed for the video comments.")

    return vid_to_sentiment


    
#Sarcasm Detection
def detect_sarcasm(comments, llm):

    context = "A short film created by drama and film club of a college"

    
    first_prompt = ChatPromptTemplate.from_template(
        '''
          For the given {context}, pick from {comments} the ones which are out of context.
          and return the list of out of context comments
        '''
    )

    # chain 1: input= context, comments and output= out_of_context_comments
    chain_one = LLMChain(llm=llm, prompt=first_prompt,
                         output_key="out_of_context_comments"
                        )

    #prompt template and chain 2: to detect exaggerated and absurd comments
    second_prompt = ChatPromptTemplate.from_template(
        '''
          from {out_of_context_comments} pick the comments which seem to be exaggerated or nonsensical.
          and return a list of such comments
        '''
    )

    #chain 2:  input= out_of_context_commentsand output= exagg_comments
    chain_two =  LLMChain(llm=llm, prompt=second_prompt,
                         output_key="exagg_comments"
                        )

    #incongruence of the sentiment with the context"""
    
    third_prompt = ChatPromptTemplate.from_template(
        '''
          from {exagg_comments} pick the comments which are incongruent with the {context}.
          And return back a list of such comments
        '''
    )
    
    #chain 3: input=exagg_comments,comments  output=sarcastic_comments
    chain_three =  LLMChain(llm=llm, prompt=third_prompt,
                         output_key="incongruent_comments"
                        )

    #overall chain
    overall_chain = SequentialChain(
        chains=[chain_one, chain_two, chain_three],
        input_variables=["context", "comments"],
        output_variables=["out_of_context_comments", "exagg_comments","incongruent_comments"]
    )

    input_data = {"context": context, "comments": comments}
    sarcastic_comments = overall_chain(input_data)

    return sarcastic_comments['incongruent_comments']


###Constructive feedback example
def constructive_feedback(english_comments, llm):

    context = "A short film created by drama and film club of a college"
    
    #selecting comments greater than 35 chars

    long_comments = [comment for comment in english_comments if len(comment) > 75]

    #calling llm to understand the feedback"""

    feedback_template = '''
      for the given list of comments, choose the comments which give constructive feedback based on the context. Summarize and output this feedback.
    
      list of comments={long_comments}
      context={context}
    '''

    prompt = ChatPromptTemplate.from_template(template=feedback_template)
    messages = prompt.format_messages(long_comments=long_comments,
                                      context=context)

    contructive_response = llm.invoke(messages)

    return contructive_response.content

def generate_wordcloud(text):
  wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
  fig, ax = plt.subplots(figsize=(10, 5))  # Create a figure and axis
  ax.imshow(wordcloud, interpolation='bilinear')
  ax.axis('off')
  st.pyplot(fig)  # Pass the figure to st.pyplot()

    

###main starts here
        
def main():
    
    #setting up gemini
    llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=api_key)

    # Create a form to capture user input
    with st.form(key="url_form"):
        # Display a text input for the URL
        video_url = st.text_input("Enter a URL:")
        submit_button = st.form_submit_button(label="Continue")

    # Check if the button was clicked
    if submit_button:
        # Process the URL (replace with your actual logic)
        st.write(f"You entered: {video_url}")
      
        #splitting the url to get the video id
        video_id = video_url.split("?v=")[1]

        #calling the function defined earlier to get comment data
    
        # Get comments
        comments_data = get_comments(video_id)

        #storing in json file
        with open("video_comments.json", "w") as file:
            json.dump(comments_data, file, indent=4)
    
        print("\nyoutube data exported\n")
    
        #reading from json file
        with open('video_comments.json', 'r') as file:
          data = json.load(file)
    
        comments = []
        for item in data["items"]:
          comments.append(item['snippet']['topLevelComment']['snippet']['textDisplay'])
      

        #separating English and marathi comments
        print("\n starting separation of comments\n")
        separated_comments = separate_comments(comments)
    
        #outputting to json file"""
        with open("separated_comments.json", "w") as file:
          json.dump(separated_comments, file, indent=4)

        print("\n done separating comments\n")
    
        print("\n starting sentiment analysis\n")
        ###SENTIMENT ANALYSIS
        stats = sentiment_analysis(separated_comments)
        st.title("Sentiment analysis")

        ### Pie chart

        # Convert dictionary to DataFrame for Plotly
        sentiment_df = {"Sentiment": list(stats.keys()), "Count": list(stats.values())}

        # Create a pie chart using Plotly Express
        fig = px.pie(sentiment_df, values='Count', names='Sentiment')

        # Show plot in Streamlit
        st.plotly_chart(fig)

        print("\ndone sentiment analysis")

        ### Wordcloud of comments

        st.title("WordCloud of comments")

        with open("separated_comments.json", "r") as file:
            comments = json.load(file)

        # Combine comments into a single string
        all_comments = ' '.join(comments["english_comments"])

        # Display word cloud
        generate_wordcloud(all_comments)


        ###SARCASM DETECTION
        print("\n starting sarcasm detection")
        sarcastic_comments = detect_sarcasm(comments, llm)
        st.title("Sarcastic comments")
        st.write(sarcastic_comments)
        print("\n DONE sarcasm detection")

        ###CONSTRCUTIVE FEEDBACK
        print("\n starting constr feedback")
        feedback = constructive_feedback(separated_comments['english_comments'], llm)
        st.title("Constructive feedback")
        st.write(feedback)
        print("\n DONE constr feedback")


if __name__ == "__main__":
  main()











