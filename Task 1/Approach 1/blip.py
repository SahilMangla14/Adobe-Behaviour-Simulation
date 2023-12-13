import requests
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Define classes for different media types
class Photo:
    def __init__(self, previewUrl, fullUrl):
        self.previewUrl = previewUrl
        self.fullUrl = fullUrl
        self.name = "Photo"

class VideoVariant:
    def __init__(self, contentType, url, bitrate):
        self.contentType = contentType
        self.url = url
        self.bitrate = bitrate

class Video:
    def __init__(self, thumbnailUrl, variants, duration, views):
        self.thumbnailUrl = thumbnailUrl
        self.variants = variants
        self.duration = duration
        self.views = views
        self.name = "Video"

class Gif:
    def __init__(self, thumbnailUrl, variants):
        self.thumbnailUrl = thumbnailUrl
        self.variants = variants
        self.name = "Gif"
        
        
def desc_image(image):
  try:
    # img_url = tweet['media'][0].fullUrl
    # raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')

    inputs = processor(image, return_tensors="pt")

    generated_ids = model.generate(**inputs, max_new_tokens=20)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    # print(generated_text)
    # tweet_description = f"The tweet was posted by {tweet['inferred company']} on {tweet['date']} containing the tweet content : {tweet['content']}. The tweet contains the image which describes the following information : {generated_text}. The username of the company is {tweet['username']}"

    # tweet_desc.append(tweet_description)
    # tweet_likes.append(tweet['likes'])

    # print(f"Image with id {tweet['id']} done")
    return generated_text
  except Exception as e:
    # print(f"Error processing image with id: {tweet['id']}")
    return ""


if __name__ == "__main__":
    
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", device_map="auto")
    
    # Load the CSV dataset
    df = pd.read_csv('./behaviour_simulation_train.csv')
    df = df.head(n=5)
    pd.options.mode.chained_assignment = None  # default='warn'

    for index, row in df.iterrows():
            df['media'].iloc[index] = eval(df['media'].iloc[index])
    
    for index,row in df.iterrows():
        if(row['media'][0].name == 'Photo'):
            img_url = row['media'][0].fullUrl
            raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')
            description = desc_image(raw_image)
            print(description)
