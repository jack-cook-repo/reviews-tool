import pandas as pd
import numpy as np
import streamlit as st
import seaborn as sns

from matplotlib import pyplot as plt
from datetime import datetime
from nltk.collocations import BigramAssocMeasures, TrigramAssocMeasures, \
    BigramCollocationFinder, TrigramCollocationFinder

st.set_page_config(layout='wide')
sns.set(font='sans-serif',
        style='darkgrid',
        font_scale=1.3)

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
    st.session_state.review_data = raw_data.query(f'date_clean >= "{start_date}" and date_clean <= "{end_date}"')


def add_data_labels(axis, label_fmt):
    '''
    Takes a pyplot axis with a bar chart and adds labels above each bar,
    with format of label_fmt
    '''
    for p in axis.patches:
        if p.get_height() == 0:
            # Don't label cases where we have removed the review score
            continue
        axis.annotate(label_fmt.format(p.get_height()),  # Get value
                      (p.get_x() + p.get_width() / 2., p.get_height()),  # Get position
                      ha='center', va='center', color='black',
                      xytext=(0, 10), textcoords='offset points')


current_date = get_date()
min_date = datetime(2020, 8, 1)

st.sidebar.write('''
    # Welcome!
    
    This app uses Google reviews to help understand your customers better.
''')

# Set up sidebar with date filters
st.sidebar.write('### Use the below boxes to pick what time period you want to look at')
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
st.markdown("<h1 style='text-align: center;'>Summary</h1>", unsafe_allow_html=True)
st.write(' ')  # Space

st.write(f'''
    Over the time period selected, there were **{df_big_easy_clean.shape[0]} reviews**, with an average score of
    **{np.around(df_big_easy_clean['reviewRating'].mean(),1)} stars**.
''')


# ################### #
# ---First plots----- #
# ################### #

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
fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)

# Get colours for bars
pal = sns.color_palette('Blues', df_monthly_output.shape[0])
rank = df_monthly_output['mean'].argsort().argsort()

sns.barplot(data=df_monthly_output,
            x='date_clean',
            y='mean',
            ax=ax,
            palette=np.array(pal)[rank])

# Get x labels
x_labels = [d.strftime('%b\n%y') for d in df_monthly_output['date_clean']]
ax.set_xticklabels(labels=x_labels, color='black')

# Chart labels and axes
ax.set_ylim((0, 5.3))
ax.set_xlabel('')
ax.set_ylabel('')
#ax.set_yticks([])

ax.set_title('Average review score (stars) by month',
             )

# Get data labels
add_data_labels(ax, label_fmt='{:.1f}')

# Despine
# sns.despine(fig=fig, left=True)

# Write to streamlit
left.pyplot(fig)


# 2: Review score breakdown

# Set up plot space
fig2, ax2 = plt.subplots(figsize=(6, 4), constrained_layout=True)

# Group number of reviews by score
df_reviews_agg = df_big_easy_clean.groupby('reviewRating').agg(num=('date_clean', 'count')).reset_index()

# Get continuous set of star ratings
df_reviews_filled = pd.DataFrame(columns=['reviewRating', 'num_reviews'],
                                 data=list(zip([1, 2, 3, 4, 5], [0, 0, 0, 0, 0])))
df_reviews_filled = pd.merge(df_reviews_filled,
                             df_reviews_agg,
                             how='left',
                             on='reviewRating')
df_reviews_filled['num_reviews'] = [0 if n != n else n for n in df_reviews_filled['num']]

# Plot
sns.barplot(data=df_reviews_filled,
            x='reviewRating',
            y='num_reviews',
            ax=ax2,
            dodge=True,
            palette=np.array(sns.color_palette("RdYlGn", n_colors=df_reviews_filled.shape[0])))

# Get y limit by taking max number of reviews for given rating and adding 20%
max_y = df_reviews_filled['num_reviews'].max() * 1.2

# Chart labels and axes
ax2.set_xticklabels(labels=[f'{str(i+1)} star{"s" if i>1 else ""}\n' for i in range(5)],
                    
                    color='black')
ax2.set_alpha(0.1)
ax2.set_xlabel('')
ax2.set_ylabel('')
# ax2.set_yticks([])
ax2.set_title('Number of reviews by score (stars)',
              )
ax2.set_ylim((0, max_y))

# Get data labels
add_data_labels(ax2, label_fmt='{:.0f}')

# Despine
# sns.despine(fig=fig2, left=True)

right.pyplot(fig2)


# ################### #
# ---Detail plots---- #
# ################### #

st.markdown("<h1 style='text-align: center;'>Detail</h1>", unsafe_allow_html=True)
st.write('This section looks into terms that come up in reviews, and how that affects the score')  # Space

st.write('**Placeholder for terms that turn up frequently in reviews**')

st.markdown("<h2 style='text-align: center;'>Specific terms</h2>", unsafe_allow_html=True)

term = st.selectbox('Pick what term you want to look at in more detail',
                    options=['birthday', 'lobster roll', 'service'])


df_big_easy_filt = df_big_easy_clean[df_big_easy_clean['review_clean'].str.contains(term)]
df_monthly_output_filt = df_big_easy_filt.set_index('date_clean').resample('M')['reviewRating'].agg(('mean',
                                                                                                 'count')).reset_index()






