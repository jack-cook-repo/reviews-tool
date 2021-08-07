import re
import pandas as pd
import numpy as np
import streamlit as st
import seaborn as sns

from matplotlib import pyplot as plt
from datetime import datetime
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

# Set up initial config
st.set_page_config(layout='wide')
sns.set(font='sans-serif',
        style='ticks',
        font_scale=1.2,
        rc={'axes.facecolor': 'ghostwhite'})

# Load initial dataframe
FILE_PATH = '/Users/jackcook/PycharmProjects/reviews-tool/data/processed_data/df_big_easy_clean.csv'
raw_data = pd.read_csv(FILE_PATH)


@st.cache
def get_date():
    return datetime.now()


# By default, set up dataframe without date filters
if 'review_data' not in st.session_state:
    st.session_state.review_data = raw_data


@st.cache
def get_review_data(start_date, end_date):
    '''
    Filters the session state review_data dataframe based on start and end dates
    '''
    st.session_state.review_data = raw_data.query(f'date_clean >= "{start_date}" and date_clean <= "{end_date}"')


@st.cache
def get_num_reviews_by_star_rating(df, star_rating_col, agg_col):
    '''
    Given a dataframe, will group by star_rating_col, count the agg_col, and return a dataframe with
    all possible star ratings (from 1-5) and the number of reviews - including zeroes.

    This ensures that we will always have a row of data for each possible star rating
    '''
    # Group by star rating and count number of reviews
    df_reviews_agg = df.groupby(star_rating_col).agg(num=(agg_col, 'count')).reset_index()

    # Get continuous set of star ratings
    df_reviews_filled = pd.DataFrame(columns=[star_rating_col, 'num_reviews'],
                                     data=list(zip([1, 2, 3, 4, 5], [0, 0, 0, 0, 0])))

    # Join dataframes
    df_reviews_filled = pd.merge(df_reviews_filled,
                                 df_reviews_agg,
                                 how='left',
                                 on=star_rating_col)

    # Fill missing values with zeroes
    df_reviews_filled['num_reviews'] = [0 if n != n else n for n in df_reviews_filled['num']]

    return df_reviews_filled[[star_rating_col, 'num_reviews']]


def add_data_labels_and_bar_widths(axis, label_fmt, new_width=1):
    '''
    Takes a pyplot axis with a bar chart and adds labels above each bar,
    with format of label_fmt, and sets the bar width to new_width
    '''
    for p in axis.patches:
        if p.get_height() == 0:
            # Don't label cases where we have removed the review score
            continue
        axis.annotate(label_fmt.format(p.get_height()),  # Get value
                      (p.get_x() + p.get_width() / 2., p.get_height()),  # Get position
                      ha='center', va='center', color='black',
                      xytext=(0, 10), textcoords='offset points')

        # Get current width
        current_width = p.get_width()
        diff = current_width - new_width

        # we change the bar width
        p.set_width(new_width)

        # we recenter the bar
        p.set_x(p.get_x() + diff * .5)


def plot_review_score_by_month(ax, color_scheme, df, date_col, reviews_col, ylim=(0, 5.3)):
    '''
    Given a Matplotlib axis, and a colour scheme, input dataframe (with date and reviews columns),
    plus optional y limits, plots a reviews by month barplot onto the axis
    '''

    # Set up colour palette with each colour
    pal = sns.color_palette(color_scheme, n_colors=51)

    # Pick colours based on review score, rounded to nearest decimal place
    rank = [0 if rev != rev else int(rev * 10) for rev in df[reviews_col]]

    # Plot
    sns.barplot(data=df,
                x=date_col,
                y=reviews_col,
                ax=ax,
                # alpha=0.5,
                palette=np.array(pal)[rank])  # p.array(pal)[rank])

    # Get x labels
    x_labels = [d.strftime('%b\n%y') if i % 2 == 0 else '' for (i, d) in enumerate(df[date_col])]
    ax.set_xticklabels(labels=x_labels, color='black')

    # Chart labels, axes, and title
    ax.set_ylim(ylim)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_title('Average review score (stars) by month')

    # Finally, add data labels and set bar widths
    add_data_labels_and_bar_widths(ax, label_fmt='{:.1f}')


def plot_reviews_by_star_rating(ax, color_scheme, df, num_reviews_col, star_rating_col):
    '''
    Given a Matplotlib axis, color scheme, input dataframe (with star rating and number of reviews columns),
    plus optional y limits, plots a reviews by month barplot onto the axis
    '''
    # Plot
    sns.barplot(data=df,
                x=star_rating_col,
                y=num_reviews_col,
                ax=ax,
                dodge=True,
                palette=np.array(sns.color_palette(color_scheme, n_colors=df.shape[0])))

    # Get y limit by taking max number of reviews for given rating and adding 20%
    max_y = df[num_reviews_col].max() * 1.2

    # Chart labels and axes
    ax.set_xticklabels(labels=[f'{str(i + 1)} star{"s" if i > 1 else ""}\n' for i in range(5)],
                        color='black')
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_title('Number of reviews by score (stars)')
    ax.set_ylim((0, max_y))

    # Get data labels
    add_data_labels_and_bar_widths(ax, label_fmt='{:.0f}')


# Set up initial values for start/end dates for filters
current_date = get_date()
min_date = datetime(2020, 8, 1)


# Set up sidebar with date filters
st.sidebar.write('''
    #### Use the below boxes to pick what time period you want to look at
    ''')
date_start = st.sidebar.date_input(
    'from:',
    min_value=min_date,
    max_value=current_date,
    value=min_date)
date_end = st.sidebar.date_input(
    'to:',
    min_value=min_date,
    max_value=current_date,
    value=current_date)

# Set up a button to filter data
st.sidebar.button('Click to update app',
                  on_click=get_review_data,
                  args=(date_start, date_end))

# Load data
df_big_easy_clean = st.session_state.review_data
df_big_easy_clean['date_clean'] = df_big_easy_clean['date_clean'].astype('datetime64[D]')

# Write title
text_colour = '#ed3e1f'
st.markdown(f'<h1 style=color:{text_colour}>🦞 Big Easy reviews app</h1>', unsafe_allow_html=True)
st.write('''
    Welcome! 
    
    This app uses Google reviews data for your Canary Wharf restaurant to help you better
    understand your customers and explore:
    - What do your customers like and dislike?
    - How does this change over time?
    
    Use the sidebar on the left to look at different time periods.
    
    ---
''')


# ################### #
# ---Overview-------- #
# ################### #

st.write(f'''
    ## 🌏 Overview
    
    This section covers some high level metrics on reviews - average scores by month, and number of reviews by star rating.
    (For average scores by month, we exclude months with 5 reviews or fewer).
    
    Over the time period selected:
    - there were **{df_big_easy_clean.shape[0]} reviews**
    - with an average score of **{np.around(df_big_easy_clean['reviewRating'].mean(),1)} stars**.
''')

left, right = st.beta_columns(2)

# 1: Reviews over time
df_monthly_output = df_big_easy_clean.set_index('date_clean').resample('M')['reviewRating'].agg(('mean',
                                                                                                 'count')).reset_index()

# Due to lockdown, some months have next to no reviews - set months with 5 or less reviews to score of 0
df_monthly_output['mean'] = np.where(df_monthly_output['count'] > 5,
                                     df_monthly_output['mean'],
                                     0)

# Round 'mean' to 1 decimal place
df_monthly_output['mean'] = df_monthly_output['mean'].map(lambda n: np.around(n, 1))

# Set up plot space
fig1, ax1 = plt.subplots(figsize=(6, 4), constrained_layout=True)

# Plot barplot onto axis
plot_review_score_by_month(ax=ax1, color_scheme='Blues',
                           df=df_monthly_output, date_col='date_clean',
                           reviews_col='mean')

# Write to streamlit
left.pyplot(fig1)


# 2: Review score breakdown
# Set up plot space
fig2, ax2 = plt.subplots(figsize=(6, 4), constrained_layout=True)

# Group number of reviews by score
df_reviews_filled = get_num_reviews_by_star_rating(df=df_big_easy_clean,
                                                   star_rating_col='reviewRating',
                                                   agg_col='date_clean')

# Plot
plot_reviews_by_star_rating(ax=ax2, color_scheme='RdYlGn', df=df_reviews_filled,
                            num_reviews_col='num_reviews', star_rating_col='reviewRating')

right.pyplot(fig2)


# ################### #
# ---Detail plots---- #
# ################### #
st.write('''
    ---
    
    ## 🗣 Things customers talk about in reviews

    This section looks into terms that come up in reviews, and how that affects the score

''')

fig3, axs3 = plt.subplots(nrows=1, ncols=3, figsize=(12, 6),
                          constrained_layout=True)

# Set up word cloud
wc = WordCloud(prefer_horizontal=1,
               max_words=100,
               background_color='ghostwhite',
               min_word_length=2,
               relative_scaling=0.8,
               width=300,
               height=200)

# Set up count vectorizer
C = CountVectorizer(ngram_range=(2, 3),
                    max_df=0.4,
                    min_df=5,
                    binary=True,
                    stop_words='english')

# Strip 'big easy' and 'canary wharf'
df_big_easy_clean['review_tfidf'] = [re.sub(r'(big easy|canary wharf)', '', s) for s in
                                     df_big_easy_clean['review_clean'].values]

# Get scores
res = C.fit_transform(df_big_easy_clean['review_tfidf'])

# Set up dataframe
df_res = pd.DataFrame(data=res.todense(),
                      columns=C.get_feature_names())

# Get counts by each term
df_term_counts = pd.DataFrame(data=df_res.T.sum(axis=1).reset_index().values,
                              columns=['term', 'count'])

# For each bigram, figure out if there's a reverse of it with a higher count
term_list = list(df_term_counts['term'].values)
terms_to_reverse = []

for term in term_list:
    term_rev = ' '.join(reversed(term.split()))

    # Check if reversed term has higher count
    if term_rev in term_list:

        # Don't bother checking the reverse term once we've checked the first term
        term_list.remove(term_rev)

        term_count = df_term_counts.query(f'term=="{term}"')['count'].values[0]
        term_rev_count = df_term_counts.query(f'term=="{term_rev}"')['count'].values[0]

        if term_rev_count >= term_count:
            terms_to_reverse.append(term)

df_res_rev = df_res.copy(deep=True)

# Then, for bigrams to reverse, add the count to it's reverse
for term in terms_to_reverse:
    term_rev = ' '.join(reversed(term.split()))

    # Add less frequent term column into frequent one
    df_res_rev[term_rev] += df_res_rev[term]

# Then, drop columns we don't need
df_res_rev = df_res_rev.drop(terms_to_reverse, axis=1)

# Now apply tfidf transformer
T = TfidfTransformer()

# Get scores
tfidf = T.fit_transform(df_res_rev)

# Set up dataframe
df_res = pd.DataFrame(data=tfidf.todense(),
                      columns=df_res_rev.columns)

# Bucket reviews into 1-2 stars, 3-4 stars, and 5 stars
review_buckets = {0: [1, 2],
                  1: [3, 4],
                  2: [5, 5]}


for i in range(3):

    # Get review bucket
    rev_min, rev_max = review_buckets[i]

    # Filter for certain reviews
    df_res_filt = df_res[[rev_min <= r <= rev_max for r in df_big_easy_clean['reviewRating']]].copy(deep=True)

    # Turn into plottable format
    df_plot = df_res_filt.T.sum(axis=1).sort_values(ascending=False)[:30]

    wc.generate_from_frequencies(df_plot)
    axs3[i].imshow(wc, interpolation='bilinear')

    if rev_min == rev_max:
        title_stars = f'{rev_min} star'
    else:
        title_stars = f'{rev_min}-{rev_max} star'

    axs3[i].set_title(f'Most common terms in {title_stars} reviews')
    axs3[i].axis('off')

fig3.tight_layout()

c1, c2, c3 = st.beta_columns(3)
st.pyplot(fig3)


# ################### #
# ------Topics------- #
# ################### #
st.write('''
    ---
    ## 🔎 Deep dive into specific topics
    
    This section allows you to really zoom in on a specific word or phrase, and see what your customers think.
''')

terms = ['birthday', 'service', 'food cold', 'lobster roll', 'mac and cheese', 'service']

term = st.selectbox('Pick a word/term from below and the charts below will change!',
                    options=sorted(terms))


# Filter dataframe for reviews containing that specific term
term_filt = term.replace(' and ', ' ')  # Hacky
df_big_easy_filt = df_big_easy_clean[df_big_easy_clean['review_clean'].str.contains(term_filt)].copy(deep=True)

st.write(f'''
    Over the time period selected, there were **{df_big_easy_filt.shape[0]} reviews** that mentioned "{term}", 
    with an average score of **{np.around(df_big_easy_filt['reviewRating'].mean(),1)} stars**
''')

# Set up partition
left2, right2 = st.beta_columns(2)

# 1: Reviews over time
df_monthly_filt = df_big_easy_filt.set_index('date_clean').resample('M')['reviewRating'].agg(('mean',
                                                                                              'count')).reset_index()

# Round 'mean' to 1 decimal place
df_monthly_filt['mean'] = df_monthly_filt['mean'].map(lambda n: np.around(n, 1))

# Set up plot space
fig4, ax4 = plt.subplots(figsize=(6, 4), constrained_layout=True)

# Plot barplot onto axis
plot_review_score_by_month(ax=ax4, color_scheme='Blues',
                           df=df_monthly_filt, date_col='date_clean',
                           reviews_col='mean', ylim=(0, 5.5))

# Write to streamlit
left2.pyplot(fig4)


# 2: Review score breakdown
# Set up plot space
fig5, ax5 = plt.subplots(figsize=(6, 4), constrained_layout=True)

# Group number of reviews by score
df_reviews_filled_filt = get_num_reviews_by_star_rating(df=df_big_easy_filt,
                                                        star_rating_col='reviewRating',
                                                        agg_col='date_clean')

# Plot
plot_reviews_by_star_rating(ax=ax5, color_scheme='RdYlGn', df=df_reviews_filled_filt,
                            num_reviews_col='num_reviews', star_rating_col='reviewRating')

right2.pyplot(fig5)


# Get relevant parts of reviews
def review_extract_term(text, term):

    # For terms with >1 word, allow for gaps between word with spaces/characters, up to 15 characters/spaces total
    term_split = term.split()
    if len(term_split) == 2:
        # Example - 'food cold', we'd want to match: 'food was cold', 'food arrived very cold'
        match_str = '(' + term_split[0] + r'[\s,]([a-z\s\,]{1,15})?' + term_split[1] + ')'

        # Because we have 2 capture groups, the overall term, and the optional characters/spaces in the middle,
        # any splits will return 2 terms: the split itself, and the optional capture group in the middle.
        #
        # For example, a match of 'food cold' on 'my food was very cold today' returns this list:
        # ['my ', 'food was very cold', 'was very ', ' today'].
        #
        # We want to ignore the 2nd capture group, or every 3rd item in the list.
        matches_prelim = re.split(match_str,
                                  text.lower())

        # Ditch every 3rd term as per above logic
        matches = [m for (i, m) in enumerate(matches_prelim) if (i+1) % 3 != 0]
    else:
        match_str = term  # No further processing needed
        matches = re.split(match_str,
                           text.lower())

    # See how many matches there are
    n_matches = len(matches) - 1

    # Loop through each match, remembering that every 2nd item is the match string
    # So every 3rd item we want to put a separator to keep multiple matches apart (if there are more than 3)
    res = []
    for i in range(n_matches+1):
        pos = i+1
        n_tokens = 30
        # Every first word in group of 3
        if (pos-1) % 3 == 0:
            res.append(' '.join(matches[i].split()[-n_tokens:]) + ' ')
        # Every second word is the match term
        elif (pos+1) % 3 == 0:
            res.append(matches[i])
        # Third word
        else:
            trailing_str_trim = ' '.join(matches[i].split()[:n_tokens]) + (' *** ' if n_matches > 3 else '')

            # Check if we need a space or not
            trailing_space = '' if (len(trailing_str_trim) == 0
                                    or re.search(r'[^A-z]', trailing_str_trim[0]) is not None
                                    or re.search(r's[^A-z]', trailing_str_trim[:3]) is not None) else ' '  # Plurals

            res.append(trailing_space + trailing_str_trim)

    return ''.join(res).strip()


# Write to session state
df_big_easy_filt['Review extract'] = df_big_easy_filt.apply(lambda row: review_extract_term(row['reviewBody'],
                                                                                            term), axis=1)


def show(df):
    '''
    Takes a dataframe and figures out buttons
    '''
    n_rows = df.shape[0]
    rows_per_page = 10
    n_pages = (n_rows // rows_per_page) + (1 if n_rows % rows_per_page != 0 else 0)

    if 'page' not in st.session_state:
        st.session_state.page = 0

    def next_page():
        st.session_state.page += 1

    def prev_page():
        st.session_state.page -= 1

    col1, col2, col3, _ = st.beta_columns([0.1, 0.17, 0.1, 0.63])

    if st.session_state.page < n_pages-1:
        col3.button('>', on_click=next_page)
    else:
        col3.write('')  # this makes the empty column show up on mobile

    if st.session_state.page > 0:
        col1.button('<', on_click=prev_page)
    else:
        col1.write('')  # this makes the empty column show up on mobile

    col2.write(f'Page {1 + st.session_state.page} of {n_pages}')
    start = rows_per_page * st.session_state.page
    end = start + rows_per_page
    st.write('')
    st.table(df.iloc[start:end])


df_review_display = df_big_easy_filt[['reviewRating', 'Review extract', 'date_clean']]
df_review_display = df_review_display.rename(columns={'date_clean': 'Review date'})


def get_stars(n):
    return int(n) * '⭐'


df_review_display['Rating'] = [f'{get_stars(stars)}' for stars in df_review_display['reviewRating']]
df_review_display['Review date'] = [str(d)[:10] for d in df_review_display['Review date']]

with st.beta_expander(f'Click here to look at extracts from customer reviews that contain "{term}"'):
    st.write(' ')
    show(df_review_display[['Review date', 'Rating', 'Review extract']].sort_values(by='Review date',
                                                                                    ascending=False).reset_index(drop=True))
