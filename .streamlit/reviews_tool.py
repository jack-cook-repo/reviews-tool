import warnings
import pandas as pd
import numpy as np
import streamlit as st
import seaborn as sns

from utils import get_review_data, get_num_reviews_by_star_rating, update_button, show, get_stars
from text_utils import get_dict_terms_to_replace, get_review_topics, write_review_topics, \
                       get_terms_to_highlight, highlight_text
from plot_utils import plot_reviews_by_month, plot_reviews_by_star_rating

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


# Caches results for speed
df_big_easy_prelim = get_review_data(st.session_state.period_filt)
df_big_easy_clean = df_big_easy_prelim.copy(deep=True)  # To make sure we don't mutate original
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
fig1 = plot_reviews_by_month(color_scheme='Blues',
                             df=df_monthly_output,
                             date_col='date_clean',
                             num_col='mean',
                             scores_or_counts='scores')

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
                      'price & money': ['price', 'cost', 'value', 'money', 'expensive', 'overprice'],
                      'great / good food': ['good food', 'great food', 'amazing food', 'delicious']}





df_rating_counts, df_rating_by_term = get_review_topics(df_big_easy_clean, dict_themed_topics)
good_reviews_all = df_rating_counts.reset_index(drop=True).loc[0, 'good_reviews_all']
bad_reviews_all = df_rating_counts.reset_index(drop=True).loc[0, 'bad_reviews_all']
total_reviews_all = df_rating_counts.reset_index(drop=True).loc[0, 'total_reviews_all']




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
               'lobster(s)', 'drink', 'great / good food']
good_topic_summaries = []
for gt in good_topics:
    good_topic_summaries.append(write_review_topics(df_rating_counts, gt, "good"))
df_gt = pd.DataFrame(good_topic_summaries, columns=['score', 'text'])
df_gt = df_gt.sort_values(by='score', ascending=False)

left_good_topics.write('### 👍 Of all good reviews...')
left_good_topics.write('\n'.join(df_gt['text'].values))

# Bad topics
bad_topics = ['staff & service', 'time & waiting',
              'price & money', 'cold']
bad_topics_summaries = []
for bt in bad_topics:
    bad_topics_summaries.append(write_review_topics(df_rating_counts, bt, "bad"))
df_bt = pd.DataFrame(bad_topics_summaries, columns=['score', 'text'])
df_bt = df_bt.sort_values(by='score', ascending=False)

right_bad_topics.write('### 👎 Of all bad reviews...')
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

left_interesting_topics.write('### 🦞 Food reviews that mentioned...')
left_interesting_topics.write('\n'.join(df_ft['text'].values))

# Then bring up some interesting insights
interesting_topics = ['birthday', 'lunch', 'brunch', 'deal', 'portion', 'bar',
                      'book', 'clean', 'cocktails']
interesting_topics_summaries = []
for it in interesting_topics:
    interesting_topics_summaries.append(write_review_topics(df_rating_counts, it, "other"))
df_it = pd.DataFrame(interesting_topics_summaries, columns=['score', 'text'])
df_it = df_it.sort_values(by='score', ascending=False)

right_food_topics.write('### 🤔 Other reviews that mentioned...')
right_food_topics.write('\n'.join(df_it['text'].values))



# ################### #
# -----Deep dive----- #
# ################### #
st.write('''
    ---
    ## 🔎 Deep dive
    
    This section allows you to really zoom in on a specific word or phrase, and see what your customers think.
''')

# Set up session state for later on
if 'show_button' not in st.session_state:
    st.session_state.show_button = 'date'

button_show = st.session_state.show_button




# Set up first partition
left2a, right2a = st.beta_columns(2)

# Then pick term
terms = list(set(good_topics + bad_topics + food_topics + interesting_topics))

left2a.write('')
left2a.write('**Pick a word/term from below to update this section**')

term = left2a.selectbox(' ',
                        options=sorted(list(set(terms))),
                        on_change=update_button,
                        key='select_term',
                        args=('date',))

# Filter dataframe for reviews containing that specific term / topic
df_big_easy_filt = df_big_easy_clean[df_rating_by_term[term] == 1].copy(deep=True)

left2a.write(f'''
    {leading_text}, there were:
    - **{df_big_easy_filt.shape[0]} reviews** that mentioned **"{term}"**
    - with an average score of **{np.nan_to_num(np.around(df_big_easy_filt['reviewRating'].mean(),1))} ⭐️**
''')

# 1: Review score breakdown
# Group number of reviews by score
df_reviews_filled_filt = get_num_reviews_by_star_rating(df=df_big_easy_filt,
                                                        star_rating_col='reviewRating',
                                                        agg_col='date_clean')

# Plot
fig4 = plot_reviews_by_star_rating(color_scheme='RdYlGn',
                                   df=df_reviews_filled_filt,
                                   num_reviews_col='num_reviews',
                                   star_rating_col='reviewRating')

# Write to streamlit
right2a.pyplot(fig4)

# Set up second partition for this section
left2b, right2b = st.beta_columns(2)

# 2: Reviews over time
df_monthly_filt = df_big_easy_filt.set_index('date_clean').resample('M')['reviewRating'].agg(('mean',
                                                                                              'count')).reset_index()

# Round 'mean' to 1 decimal place
df_monthly_filt['mean'] = df_monthly_filt['mean'].map(lambda n: np.around(n, 1))

# Plot barplot onto axis
fig5 = plot_reviews_by_month(color_scheme='Blues',
                             df=df_monthly_filt,
                             date_col='date_clean',
                             num_col='mean',
                             scores_or_counts='scores')

# Write to streamlit
left2b.pyplot(fig5)


figx = plot_reviews_by_month(color_scheme='Oranges',
                             df=df_monthly_filt,
                             date_col='date_clean',
                             num_col='count',
                             scores_or_counts='counts',
                             figsize=(6, 4))
right2b.pyplot(figx)


# Get dictionary of terms we replaced in our clean text (e.g. barbeque --> bbq)
# This will help us in the highlighting text step
dict_terms_to_replace = get_dict_terms_to_replace()


terms_to_highlight = get_terms_to_highlight(term, dict_themed_topics, dict_terms_to_replace)


# Apply to our dataframe
vect_highlight_text = np.vectorize(highlight_text, excluded=['term', 'terms_to_highlight'])
df_review_display = df_big_easy_filt.copy(deep=True)
df_review_display['review_highlighted'] = vect_highlight_text(text=df_review_display['reviewBody'],
                                                              term=term,
                                                              terms_to_highlight=terms_to_highlight)


df_review_display['rating'] = [f'{get_stars(stars)}' for stars in df_review_display['reviewRating']]
df_review_display['review_date'] = [str(d)[:10] for d in df_review_display['date_clean']]
df_review_display = df_review_display[['rating', 'review_date', 'review_highlighted']].copy(deep=True)

st.write('')
st.write(f'#### See what people have to say about "{term}"')
st.write('')


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

    if button_show == 'date':
        show(df_review_display.sort_values(by=['review_date'],
                                           ascending=False).reset_index(drop=True))
    else:
        asc = True if button_show == 'score_asc' else False
        show(df_review_display.sort_values(by=['rating', 'review_date'],
                                           ascending=asc).reset_index(drop=True))
