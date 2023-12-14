# import numpy as np
# import pandas as pd
# import spacy

# # model_name = "tunedModels/adobemid-fwdqq210yvfu"
# # model_name = "models/text-bison-001"

# actual_and_generated = []

# # Load spaCy language model
# nlp = spacy.load("en_core_web_sm")

# def extract_keywords(text):
#     # Process the text using spaCy
#     doc = nlp(text)

#     # Filter out stop words and non-content words
#     keywords = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
#     verbs = [token.lemma_ for token in doc if token.pos_ == "VERB"]

#     return keywords+verbs

# def get_generated_tweets():

#     tweet_desc = np.load('data/tweet_desc.npy', allow_pickle=True)
#     tweet_likes = np.load('data/tweet_likes.npy', allow_pickle=True)

#     for tweet, like in zip(tweet_desc, tweet_likes):
#         company = tweet['company']
#         username = tweet['username']
#         keywords = extract_keywords(tweet['media'])

#         prompt_given_company = f"As the social media manager for '{company}' (Twitter: @{username}), create a tweet using the following keywords: {str(keywords)}. Craft a message that aligns with our brand and is likely to receive at least {like} likes."

#         actual_and_generated.append({
#             'input': prompt_given_company,
#             'output': tweet['content']
#         })

#     df = pd.DataFrame(actual_and_generated)
#     df.to_csv('saved/palm_fine_tuning_approach_3.csv', index=False)

# get_generated_tweets()

import numpy as np
import pandas as pd
import spacy
from tqdm import tqdm
from joblib import Parallel, delayed

# Load spaCy language model
nlp = spacy.load("en_core_web_sm")

def extract_keywords(text):
    # Process the text using spaCy
    doc = nlp(text)

    # Filter out stop words and non-content words
    keywords = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
    verbs = [token.lemma_ for token in doc if token.pos_ == "VERB"]

    return keywords + verbs

def process_tweet(tweet):
    company = tweet['company']
    username = tweet['username']
    keywords = extract_keywords(tweet['media'])

    prompt_given_company = f"As the social media manager for '{company}' (Twitter: @{username}), create a tweet using the following keywords: {str(keywords)}. Craft a message that aligns with our brand and is likely to receive at least {like} likes."

    return {
        'input': prompt_given_company,
        'output': tweet['content']
    }

def get_generated_tweets():
    tweet_desc = np.load('data/tweet_desc.npy', allow_pickle=True)
    tweet_likes = np.load('data/tweet_likes.npy', allow_pickle=True)

    # Use parallel processing to speed up the extraction
    processed_tweets = Parallel(n_jobs=-1)(
        delayed(process_tweet)(tweet) for tweet in tqdm(tweet_desc)
    )

    df = pd.DataFrame(processed_tweets)
    df.to_csv('saved/palm_fine_tuning_approach_3.csv', index=False)

get_generated_tweets()
