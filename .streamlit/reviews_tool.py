import re
# import mpld3
import warnings
import pandas as pd
import numpy as np
import streamlit as st
import seaborn as sns

from matplotlib import pyplot as plt
from datetime import datetime
from dateutil.relativedelta import relativedelta
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

# Ignore warnings when we filter by a regex string
warnings.filterwarnings("ignore", 'This pattern has match groups')
warnings.filterwarnings('ignore', 'The DataFrame has column names of mixed type')

# Set up initial config
st.set_page_config(layout='wide')
sns.set(font='sans-serif',
        style='ticks',
        font_scale=1.2,
        rc={'axes.facecolor': 'ghostwhite'})


# Set up sidebar
st.sidebar.write('### I want to look at reviews over...')

if 'period_filt' not in st.session_state:
    st.session_state.period_filt = 'All time'

st.session_state.period_filt = st.sidebar.radio(label='',
                                                options=['All time', 'Past month', 'Past 3 months', 'Past year'])


@st.cache
def get_date():
    return datetime.now()


def get_review_data(date_filt):
    '''
    Filters the session state review_data dataframe based on start and end dates
    '''
    # Load initial dataframe
    FILE_PATH = '/Users/jackcook/PycharmProjects/reviews-tool/data/processed_data/df_big_easy_clean.csv'
    raw_data = pd.read_csv(FILE_PATH)

    current_date = get_date()

    if date_filt == 'All time':
        return raw_data
    else:
        if date_filt == 'Past month':
            start_date = current_date - relativedelta(months=1)
        elif date_filt == 'Past year':
            start_date = current_date - relativedelta(years=1)
        elif date_filt == 'Past 3 months':
            start_date = current_date - relativedelta(months=3)
        else:
            raise ValueError('Invalid date filter picked')

        return raw_data.query(f'date_clean > "{start_date}"')


# Caches results for speed
df_big_easy_clean = get_review_data(st.session_state.period_filt)


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


def plot_review_score_by_month(color_scheme, df, date_col, reviews_col,
                               n_reviews_col, ylim=(0, 5.3)):
    '''
    Given a Matplotlib axis, and a colour scheme, input dataframe (with date and reviews columns),
    plus optional y limits, plots a reviews by month barplot onto the axis
    '''
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)

    # Set up colour palette with each colour
    pal = sns.color_palette(color_scheme, n_colors=51)

    # Pick colours based on review score, rounded to nearest decimal place
    rank = [0 if rev != rev else int(rev * 10) for rev in df[reviews_col]]

    if df.shape[0] > 0:
        # Plot
        sns.barplot(data=df,
                    x=date_col,
                    y=reviews_col,
                    ax=ax,
                    # alpha=0.5,
                    palette=np.array(pal)[rank])

        # Get x labels
        if st.session_state.period_filt in ('Past year', 'All time'):
            x_labels = [d.strftime('%b\n%y') if i % 2 == 0 else '' for (i, d) in enumerate(df[date_col])]
        else:
            x_labels = [d.strftime('%b\n%y') for d in df[date_col]]
        ax.set_xticklabels(labels=x_labels, color='black')

    # Chart labels, axes, and title
    ax.set_ylim(ylim)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_title('Average review score (stars) by month')

    # Add data labels and set bar widths
    add_data_labels_and_bar_widths(ax, label_fmt='{:.1f}')

    # Then add tooltips
    # labels = [f'{n} reviews' for n in df[n_reviews_col]]
    # tooltip = mpld3.plugins.PointLabelTooltip(ax, labels=labels)
    # mpld3.plugins.connect(fig, tooltip)

    return fig


def plot_reviews_by_star_rating(color_scheme, df, num_reviews_col, star_rating_col):
    '''
    Given a Matplotlib axis, color scheme, input dataframe (with star rating and number of reviews columns),
    plus optional y limits, plots a reviews by month barplot onto the axis
    '''
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)

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

    return fig


# Load data
df_big_easy_clean['date_clean'] = df_big_easy_clean['date_clean'].astype('datetime64[D]')

# Write title
text_colour = '#e03326'
st.markdown(f'<h1 style=color:{text_colour}>🦞 Big Easy reviews app</h1>', unsafe_allow_html=True)
st.write('''
    ### Welcome! 
    
    This app uses Google reviews data for your Canary Wharf restaurant to help you better
    understand your customers and explore:
    - What do your customers like and dislike?
    - How does this change over time?
    
    Use the sidebar on the left to look at different time periods.
    
    ---
''')


# ################### #
# -----Overview------ #
# ################### #

period_lower = st.session_state.period_filt.lower()
if period_lower == 'all time':
    leading_text = 'Over all time'
else:
    leading_text = f'Over the {period_lower}'

st.write(f'''
    ## 🌏 Overview
    
    This section covers some high level metrics on reviews - average scores by month, and number of reviews by star rating.
    (For average scores by month, we exclude months with 5 reviews or fewer).
    
    {leading_text}:
    - there were **{df_big_easy_clean.shape[0]} reviews**
    - with an average score of **{np.nan_to_num(np.around(df_big_easy_clean['reviewRating'].mean(),1))} ⭐️**
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

# Plot barplot onto axis
fig1 = plot_review_score_by_month(color_scheme='Blues',
                                  df=df_monthly_output,
                                  date_col='date_clean',
                                  n_reviews_col='count',
                                  reviews_col='mean')

# Write to streamlit
left.pyplot(fig1)
# mpld3.show(fig=fig1)


# 2: Review score breakdown
# Group number of reviews by score
df_reviews_filled = get_num_reviews_by_star_rating(df=df_big_easy_clean,
                                                   star_rating_col='reviewRating',
                                                   agg_col='date_clean')

# Plot
fig2 = plot_reviews_by_star_rating(color_scheme='RdYlGn', df=df_reviews_filled,
                                   num_reviews_col='num_reviews', star_rating_col='reviewRating')

# Write to streamlit
right.pyplot(fig2)


# ################### #
# ------Topics------- #
# ################### #
st.write('''
    ---
    
    ## 🗣 Things customers are talking about
''')

st.write('')

# Create themed columns for review topics
dict_themed_topics = {'time & waiting': ['time', 'minutes', 'wait', 'later', 'slow', 'hour', 'ages'],
                      'staff & service': ['staff', 'service'],
                      'lobster(s)': ['lobsters', 'lobster'],
                      'atmosphere & music': ['atmosphere', 'music'],
                      'price & money': ['price', 'cost', 'value', 'money', 'expensive', 'overprice']}


def get_review_topics(df):

    # Set up count vectorizer
    C = CountVectorizer(ngram_range=(1, 3),
                        max_df=0.4,
                        min_df=5,
                        binary=True,
                        stop_words='english')

    # Get scores
    res = C.fit_transform(df['review_clean'])

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
        if len(term.split()) == 1:
            continue
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
    df_res = df_res_rev.drop(terms_to_reverse, axis=1).copy(deep=True)

    # ***** Correlations ***** #
    # Get correlations for terms with > 10 reviews
    df_correl_counts = pd.DataFrame(data=df_res.T.sum(axis=1).reset_index().values,
                                    columns=['term', 'count'])
    terms_to_keep = df_correl_counts.query('count >= 10')['term'].values

    # Compute the correlation matrix
    corr = df_res[terms_to_keep].corr()

    # Unpivot
    df_corr = corr.unstack().reset_index()

    # Rename columns, and filter out cases when terms are identical
    df_corr = df_corr.rename(columns={'level_0': 'term_1',
                                      'level_1': 'term_2',
                                      0: 'correl'}).query('term_1 != term_2').reset_index(drop=True).copy(deep=True)

    # Filter out cases when a term in one column is present in another
    df_corr = df_corr[[(t1 not in t2) and (t2 not in t1) for (t1, t2) in zip(df_corr['term_1'],
                                                                             df_corr['term_2'])]].copy(deep=True)

    # Get absolute correlations and rank
    df_corr['abs_correl'] = [abs(n) for n in df_corr['correl']]
    df_corr['rank'] = df_corr.groupby('term_1')['abs_correl'].rank(method='dense',
                                                                   ascending=False)
    # st.write(df_corr.query('rank <= 5').drop('abs_correl', axis=1).sort_values(by=['term_1', 'rank']))

    # ***** Themes ***** #
    # Group columns by themes
    for theme in dict_themed_topics.keys():
        df_res[theme] = df_res[[col for col in df_res.columns if col in dict_themed_topics[theme]]].max(axis=1).values

    # Get terms that crop up in reviews, and add ratings
    df_rating_by_term = pd.concat((df_res, df['reviewRating'].astype(int)), axis=1)

    # Group by rating, and sum up number of reviews where that term is present
    df_rating_by_term_agg = df_rating_by_term.groupby('reviewRating').sum()

    # Transpose, so that number of ratings are columns and rows are terms
    df_rating_counts = df_rating_by_term_agg.T

    # Then bucket by 1-2 and 4-5 star
    df_rating_counts['total_reviews'] = df_rating_counts.sum(axis=1)
    df_rating_counts['bad_reviews'] = df_rating_counts[1] + df_rating_counts[2]
    df_rating_counts['good_reviews'] = df_rating_counts[4] + df_rating_counts[5]

    # Then get percentages
    df_rating_counts['pct_reviews_bad'] = 100*(df_rating_counts['bad_reviews'] / [1 if n == 0 else n for n in df_rating_counts['total_reviews']])
    df_rating_counts['pct_reviews_good'] = 100*(df_rating_counts['good_reviews'] / [1 if n == 0 else n for n in df_rating_counts['total_reviews']])
    df_rating_counts['pct_reviews_bad'] = df_rating_counts['pct_reviews_bad'].astype(int)
    df_rating_counts['pct_reviews_good'] = df_rating_counts['pct_reviews_good'].astype(int)

    # Add total number of ratings
    df['reviewRating'] = df['reviewRating'].astype(int)
    series_total_reviews_by_star = df.groupby('reviewRating').agg(num_reviews=('date_clean', 'count')).T.iloc[0]
    df_rating_counts['bad_reviews_all'] = series_total_reviews_by_star[1] + series_total_reviews_by_star[2]
    df_rating_counts['good_reviews_all'] = series_total_reviews_by_star[4] + series_total_reviews_by_star[5]
    df_rating_counts['total_reviews_all'] = series_total_reviews_by_star.sum()

    # For debugging
    # st.write(df_rating_counts.query('total_reviews >= 10').sort_index())

    return df_rating_counts.query('total_reviews >= 10').sort_index(), df_rating_by_term


df_rating_counts, df_rating_by_term = get_review_topics(df_big_easy_clean)
good_reviews_all = df_rating_counts.reset_index(drop=True).loc[0, 'good_reviews_all']
bad_reviews_all = df_rating_counts.reset_index(drop=True).loc[0, 'bad_reviews_all']
total_reviews_all = df_rating_counts.reset_index(drop=True).loc[0, 'total_reviews_all']


def write_review_topics(df, topic, topic_type):

    assert topic_type in ('good', 'bad', 'other'), 'topic_type must be one of: good, bad, other'

    topic_cased = (topic[0].upper() + topic[1:]).replace('Mac cheese', 'Mac & cheese')

    good_reviews_w_topic = df.loc[topic, 'good_reviews']
    good_reviews_all = df.loc[topic, 'good_reviews_all']
    pct_good_reviews_contained_topic = int(100*good_reviews_w_topic / good_reviews_all)

    bad_reviews_w_topic = df.loc[topic, 'bad_reviews']
    bad_reviews_all = df.loc[topic, 'bad_reviews_all']
    pct_bad_reviews_contained_topic = int(100*bad_reviews_w_topic / bad_reviews_all)

    if topic_type == 'good':
        return [pct_good_reviews_contained_topic,
                f'- **{pct_good_reviews_contained_topic}%** mentioned {topic}']
    elif topic_type == 'bad':
        return [pct_bad_reviews_contained_topic,
                f'- **{pct_bad_reviews_contained_topic}%** mentioned {topic}']
    else:
        # Get the number of reviews for that topic, by star rating
        reviews_by_star = df.loc[topic, 1:5]

        # Take the weighted average to get the review score to 1 decimal place
        weighted_score = np.around(np.sum(reviews_by_star.values * reviews_by_star.index) / np.sum(reviews_by_star.values), 1)

        return [weighted_score,
                f'- **{topic_cased}** ({np.sum(reviews_by_star.values)} reviews) had an average score of {weighted_score} ⭐️']


# Write section intro
st.write(f'''
    In this section, a 4-5 star review is considered "good", and a 1-2 star review is considered "bad". 3 star reviews
    are considered to be neutral and are ignored.
    
    ### {leading_text}:
    
    - **{good_reviews_all}** reviews were "good" reviews (**{int(100*good_reviews_all / total_reviews_all)}%** of all reviews)
    - **{bad_reviews_all}** reviews were "bad" reviews (**{int(100*bad_reviews_all / total_reviews_all)}%** of all reviews)
''')

# For each topic, write summaries
left_good_topics, right_bad_topics = st.beta_columns(2)

# Good topics
good_topics = ['staff & service', 'atmosphere & music',
               'lobster(s)', 'bbq', 'bottomless']
good_topic_summaries = []
for gt in good_topics:
    good_topic_summaries.append(write_review_topics(df_rating_counts, gt, "good"))
df_gt = pd.DataFrame(good_topic_summaries, columns=['score', 'text'])
df_gt = df_gt.sort_values(by='score', ascending=False)

left_good_topics.write('## 👍 Of all good reviews...')
left_good_topics.write('\n'.join(df_gt['text'].values))

# Bad topics
bad_topics = ['staff & service', 'time & waiting',
              'price & money', 'cold']
bad_topics_summaries = []
for bt in bad_topics:
    bad_topics_summaries.append(write_review_topics(df_rating_counts, bt, "bad"))
df_bt = pd.DataFrame(bad_topics_summaries, columns=['score', 'text'])
df_bt = df_bt.sort_values(by='score', ascending=False)

right_bad_topics.write('## 👎 Of all bad reviews...')
right_bad_topics.write('\n'.join(df_bt['text'].values))


st.write('''
    We can also take a bit of a closer look at some other interesting topics that come up in reviews,
    and how different menu items are rated in reviews that mention them.
''')
# Set up sub columns
left_interesting_topics, right_food_topics = st.beta_columns(2)

# Food insights
food_topics = ['lobster(s)', 'shrimp', 'rib', 'bbq', 'mac cheese', 'chicken',
               'chip', 'lobster roll', 'seafood', 'steak', 'wing']
food_topics_summaries = []
for ft in food_topics:
    food_topics_summaries.append(write_review_topics(df_rating_counts, ft, "other"))
df_ft = pd.DataFrame(food_topics_summaries, columns=['score', 'text'])
df_ft = df_ft.sort_values(by='score', ascending=False)

left_interesting_topics.write('## 🦞 Food reviews that mentioned...')
left_interesting_topics.write('\n'.join(df_ft['text'].values))

# Then bring up some interesting insights
interesting_topics = ['birthday', 'lunch', 'brunch', 'deal', 'portion', 'bar',
                      'book', 'clean', 'cocktails']
interesting_topics_summaries = []
for it in interesting_topics:
    interesting_topics_summaries.append(write_review_topics(df_rating_counts, it, "other"))
df_it = pd.DataFrame(interesting_topics_summaries, columns=['score', 'text'])
df_it = df_it.sort_values(by='score', ascending=False)

right_food_topics.write('## 🤔 Other reviews that mentioned...')
right_food_topics.write('\n'.join(df_it['text'].values))



# ################### #
# -----Deep dive----- #
# ################### #
st.write('''
    ---
    ## 🔎 Deep dive
    
    This section allows you to really zoom in on a specific word or phrase, and see what your customers think.
''')

terms = list(set(good_topics + bad_topics + food_topics + interesting_topics))

term = st.selectbox('Pick a word/term from below and the charts below will change!',
                    options=sorted(list(set(terms))))


# Filter dataframe for reviews containing that specific term
term_split = term.split()

# For terms with >1 word, allow for gaps between word with spaces/characters, up to 15 characters/spaces total
if len(term_split) == 2:
    match_str = r'(\b' + term_split[0] + r'(ed|ing)?[\s,]([0-9a-z\s\,-]{1,15})?' + term_split[1] + r')s?\b'
else:
    match_str = r'(\b' + term + r')s?\b'  # No further processing needed
    if term == 'bring':
        match_str = r'(\bbring|\bbrought)s?\b'

# df_big_easy_filt = df_big_easy_clean[df_big_easy_clean['review_clean'].str.contains(match_str)].copy(deep=True)
df_big_easy_filt = df_big_easy_clean[df_rating_by_term[term] == 1].copy(deep=True)

st.write(f'''
    {leading_text}, there were **{df_big_easy_filt.shape[0]} reviews** that mentioned **"{term}"**, 
    with an average score of **{np.nan_to_num(np.around(df_big_easy_filt['reviewRating'].mean(),1))} ⭐️**
''')

# Set up partition
left2, right2 = st.beta_columns(2)

# 1: Reviews over time
df_monthly_filt = df_big_easy_filt.set_index('date_clean').resample('M')['reviewRating'].agg(('mean',
                                                                                              'count')).reset_index()

# Round 'mean' to 1 decimal place
df_monthly_filt['mean'] = df_monthly_filt['mean'].map(lambda n: np.around(n, 1))

# Plot barplot onto axis
fig4 = plot_review_score_by_month(color_scheme='Blues', df=df_monthly_filt,
                                  date_col='date_clean', n_reviews_col='count',
                                  reviews_col='mean', ylim=(0, 5.5))

# Write to streamlit
left2.pyplot(fig4)


# 2: Review score breakdown
# Group number of reviews by score
df_reviews_filled_filt = get_num_reviews_by_star_rating(df=df_big_easy_filt,
                                                        star_rating_col='reviewRating',
                                                        agg_col='date_clean')

# Plot
fig5 = plot_reviews_by_star_rating(color_scheme='RdYlGn', df=df_reviews_filled_filt,
                                   num_reviews_col='num_reviews', star_rating_col='reviewRating')

# Write to streamlit
right2.pyplot(fig5)

st.write('Placeholder - number of reviews mentioning this term over time')


# Get relevant parts of reviews
def review_extract_term(text):

    # Apply extra replacement steps
    text_further_rep = re.sub(r'ambience', 'atmosphere', text.lower())
    text_further_rep = re.sub(r'barbeque', 'bbq', text_further_rep)
    text_further_rep = re.sub(r'b.b.q', 'bbq', text_further_rep)
    text_further_rep = re.sub(r'(server|waiter|waitress)', 'staff', text_further_rep)
    text_further_rep = text_further_rep.replace('mins', 'minutes')
    text_further_rep = text_further_rep.replace('hrs', 'hours')
    text_further_rep = re.sub('£[0-9.]+', 'money', text_further_rep)

    # For terms with >1 word, allow for gaps between word with spaces/characters, up to 15 characters/spaces total
    if len(term_split) == 2:
        # Because we have 3 capture groups, the overall term, the optional (ed|ing), and the optional characters/spaces
        # in the middle, any splits will return 3 terms:
        # the split itself, optional (ed|ing), and the optional capture group in the middle.
        #
        # For example, a match of 'food cold' on 'my food was very cold today' returns this list:
        # ['my ', 'food was very cold', None, 'was very ', ' today'].
        #
        # We want to ignore the 2nd & 3rd capture group, or every 3rd & 4th item in the list.
        matches_prelim = re.split(match_str,
                                  text_further_rep.lower())

        # Ditch every 3rd & 4th term as per above logic
        matches = []
        pos = 0
        for (i, m) in enumerate(matches_prelim):
            if pos in (0, 1, 4):
                matches.append(m)
                if pos == 4:
                    pos = 0
            pos += 1
        # matches = [m for (i, m) in enumerate(matches_prelim) if (i+1) % 3 != 0]
    else:
        matches = re.split(match_str,
                           text_further_rep.lower())

    # See how many match string there are
    n_matches = len(matches)

    # See how many triplets there are
    n_triplets = int((n_matches - 1) / 2)

    # Loop through each match, remembering that every 2nd item is the match string
    # So every 3rd item we want to put a separator to keep multiple matches apart (if there are more than 3)
    res = []
    slice_start = 0
    for triplet in range(n_triplets):
        slice_end = slice_start + 3
        matches_triplet = matches[slice_start: slice_end]
        for i in range(3):
            n_tokens = 30
            # Every first word in group of 3
            if i == 0:
                res.append(' '.join(matches_triplet[i].split()[-n_tokens:]) + ' ')
            # Every second word is the match term
            elif i == 1:
                res.append(matches_triplet[i])
            # Third word
            else:
                trailing_str_trim = ' '.join(matches_triplet[i].split()[:n_tokens]) + (' *** ' if n_matches > 3 else '')

                # Check if we need a space or not
                trailing_space = '' if (len(trailing_str_trim) == 0
                                        or re.search(r'[^A-z]', trailing_str_trim[0]) is not None
                                        or re.search(r's[^A-z]', trailing_str_trim[:2]) is not None) else ' '  # Plurals

                res.append(trailing_space + trailing_str_trim)
        slice_start += 2
    return ''.join(res).strip()


# Get review extract
if df_big_easy_filt.shape[0] > 0:
    df_big_easy_filt['Review extract'] = df_big_easy_filt.apply(lambda row: review_extract_term(row['reviewBody']),
                                                                axis=1)
else:
    df_big_easy_filt['Review extract'] = None


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

    # Put buttons at bottom of table
    col1b, col2b, col3b, _ = st.beta_columns([0.1, 0.17, 0.1, 0.63])
    if st.session_state.page < n_pages-1:
        col3b.button('>', on_click=next_page, key='col3b')
    else:
        col3b.write('')  # this makes the empty column show up on mobile

    if st.session_state.page > 0:
        col1b.button('<', on_click=prev_page, key='col1b')
    else:
        col1b.write('')  # this makes the empty column show up on mobile
    col2b.write(f'Page {1 + st.session_state.page} of {n_pages}')


df_review_display = df_big_easy_filt[['reviewRating', 'Review extract', 'date_clean']]
df_review_display = df_review_display.rename(columns={'date_clean': 'Review date'})


def get_stars(n):
    return int(n) * '⭐'


df_review_display['Rating'] = [f'{get_stars(stars)}' for stars in df_review_display['reviewRating']]
df_review_display['Review date'] = [str(d)[:10] for d in df_review_display['Review date']]

# st.write('')
# # Set up word cloud
# w2 = WordCloud(prefer_horizontal=1,
#                max_words=100,
#                background_color='ghostwhite',
#                min_word_length=2,
#                relative_scaling=0.8,
#                collocation_threshold=10,
#                width=900,
#                height=300)
# fig6, ax6 = plt.subplots()
# w2.generate(re.sub('(' + '|'.join(term.split()) + ')',
#                    '',
#                    ' '.join(df_big_easy_filt['review_clean'].values)))
# ax6.imshow(w2, interpolation='bilinear')
# ax6.axis('off')
# st.pyplot(fig6)

st.write('')
st.write(f'#### See what people have to say about "{term}"')
st.write('')

if 'show_button' not in st.session_state:
    st.session_state.show_button = 'date'

button_show = st.session_state.show_button


def update_button(bt_show):
    st.session_state.show_button = bt_show


with st.beta_expander('Click to expand'):
    st.write(' ')
    bt1, bt2, bt3, _ = st.beta_columns((2, 2, 2, 5))
    bt1.button(label='Highest scores',
               on_click=update_button,
               key='bt1',
               args=('score_desc', ))
    bt2.button(label='Lowest scores',
               on_click=update_button,
               key='bt2',
               args=('score_asc', ))
    bt3.button(label='Newest',
               on_click=update_button,
               key='bt3',
               args=('date',))

    # if button_show == 'date':
    #     show(df_review_display[['Review date', 'Rating', 'Review extract']].sort_values(by=['Review date'],
    #                                                                                     ascending=False).reset_index(
    #         drop=True))
    # else:
    #     asc = True if button_show == 'score_asc' else False
    #     show(
    #         df_review_display[['Review date', 'Rating', 'Review extract']].sort_values(by=['Rating', 'Review date'],
    #                                                                                    ascending=asc).reset_index(
    #             drop=True))

