import numpy as np
import pandas as pd
import logging
import re

from datetime import datetime, timedelta
from nltk.corpus import words, stopwords


def parse_date(string_date):
    '''
    Takes string dates from Google review that are normally in a format of '6 days ago' or '3 months ago'
    and turns them into actual dates by comparing relative to the current date.

    Slight flaw with Google reviews is that we lose specifics over time, e.g. anything > 1 year old will
    be assigned to the same date as any other review within that year.

    :param string_date: Text in the format of '1 day ago', '2 years ago', '3 weeks ago' etc.
    :return: Date parsed from string_date
    '''
    curr_date = datetime.now()
    split_date = string_date.split(' ')

    # First split will be numeric quantity
    n = split_date[0]
    n = 1 if n in ('a', 'an') else int(n)  # Replace 'a week ago' / 'an hour ago' with '1 week ago'

    # Days/weeks/months etc. will be the next split
    delta = split_date[1]

    # Check assumption that any n>1 must be plural - i.e. delta will have a trailing s
    assert (n == 1) or (n > 1 and delta[-1] == 's')

    # Strip trailing 's' from delta
    if n > 1:
        delta = delta[:-1]

    if delta == 'year':
        return curr_date - timedelta(days=365 * n)
    elif delta == 'month':
        return curr_date - timedelta(days=30 * n)
    elif delta == 'week':
        return curr_date - timedelta(weeks=n)
    elif delta == 'day':
        return curr_date - timedelta(days=n)
    elif delta == 'hour':
        return curr_date - timedelta(hours=n)
    elif delta == 'minute':
        return curr_date - timedelta(minutes=n)
    elif delta == 'moment':
        return curr_date - timedelta(seconds=n)
    else:
        raise ValueError(f'Unhandled delta type {delta}')


def pct_review_words_in_eng_corpus(text, n_tokens: int):
    '''
    Takes a review, removes anything that isn't a letter or a space, splits by spaces to get tokens,
    then checks how many of the first n_tokens appear in the nltk English corpus.

    If n_tokens is > the number of tokens, this will just check the whole review.

    :param text: Review text
    :param n_tokens: How many tokens from the review text you want to check are English or not
    :return: A percentage of how many of the tokens appear in the nltk English corpus
    '''

    #assert n_tokens >= 1 and isinstance(n_tokens, int), 'n_tokens must be an int > 0'

    # Get list of English words
    eng_words = words.words('en')
    eng_words_lower = [w.lower() for w in eng_words]

    # Keep characters and spaces
    text_alpha = re.sub(r'[^A-z ]', '', text.lower())

    # Only take first n tokens
    tokens = text_alpha.split(' ')[:n_tokens]
    n_tokens = len(tokens)
    tokens_in_eng_corpus = np.sum([1 if t in eng_words_lower else 0 for t in tokens])

    return float(tokens_in_eng_corpus) / n_tokens


def clean_text(text):
    '''
    Takes text and applies lower case, removes stopwords (from nltk English stopwords), strips punctuation/numbers,
    and returns the cleaned text.

    :param text:
    :return: Text in lower case, with no stop words, punctuation, or numbers
    '''
    tokens = text.split(' ')

    # Get lower case
    tokens_lower = [t.lower() for t in tokens]

    # Then remove stopwords
    sw = stopwords.words('english')
    tokens_no_sw = [t for t in tokens_lower if t not in sw]

    # Then strip punctuation
    tokens_alpha = [re.sub('[^A-z]', '', t) for t in tokens_no_sw]

    # Then take lemmas
    # lemmas = [wnl.lemmatize(t, pos='v') for t in tokens_alpha]

    # Rejoin as single string
    text_clean = ' '.join(tokens_alpha)  # lemmas

    # Handle duplicate spaces
    return re.sub(r'(\s)+', ' ', text_clean)


# Set up logger
log = logging.getLogger('data_prep')

# Set up stream & file handlers with formatting
sh = logging.StreamHandler()  # Writes to console
fh = logging.FileHandler('data/logs/logging.txt', mode='w')  # Writes to file, overwriting
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', '', '%')
sh.setFormatter(formatter)
fh.setFormatter(formatter)

# Add handlers
log.addHandler(sh)
log.addHandler(fh)
log.setLevel(logging.DEBUG)
log.info('Starting run of data_prep.py')

# Load Big Easy review data from Google
df_big_easy = pd.read_excel('data/input_data/Reviews.xlsx')
log.info(f'Big Easy data loaded with shape {df_big_easy.shape}')

# Keep the interesting columns from the file, and remove rows with no review text
log.info(f'df_big_easy columns: {df_big_easy.columns.values}')
df_big_easy_simple = df_big_easy[['reviewBody',
                                  'reviewRating',
                                  'dateCreated']].copy(deep=True)
df_big_easy_simple = df_big_easy_simple.dropna(subset=['reviewBody'])
log.info(f'df_big_easy_simple shape: {df_big_easy_simple.shape}')


# Parse the dates provided by Google - i.e. '3 days ago' would become '2021-08-01' if today was '2021-08-04'
log.info('Parsing dates')
df_big_easy_simple['date_clean'] = df_big_easy_simple['dateCreated'].apply(
    lambda d: parse_date(d)).astype('datetime64[D]')


# Then remove non-English reviews
log.info('Removing non-English reviews')
df_big_easy_eng = df_big_easy_simple.copy(deep=True)

# Vectorize to improve performance
vect_pct_review_words_in_eng_corpus = np.vectorize(pct_review_words_in_eng_corpus)
df_big_easy_eng['pct_tokens_english'] = vect_pct_review_words_in_eng_corpus(df_big_easy_eng['reviewBody'], 10)

# Assume any review where >=50% of first 10 words are in English corpus, is an English review
df_big_easy_eng['is_english'] = [pct >= 0.5 for pct in df_big_easy_eng['pct_tokens_english']]

# Drop rows that aren't in English
df_big_easy_eng = df_big_easy_eng[df_big_easy_eng['is_english'] == True].drop(['is_english',
                                                                               'pct_tokens_english'], axis=1)
log.info(f'{str(df_big_easy_simple.shape[0] - df_big_easy_eng.shape[0])} non-English rows dropped')


# Then clean review text & remove stopwords
log.info('Cleaning review text')
df_big_easy_clean = df_big_easy_eng.copy(deep=True)

df_big_easy_clean['review_clean'] = df_big_easy_clean.apply(lambda row: clean_text(row['reviewBody']),
                                                            axis=1)

df_big_easy_clean.to_csv('data/processed_data/df_big_easy_clean.csv', index=False)
log.info('Data prep finished')