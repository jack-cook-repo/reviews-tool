import altair as alt
import pandas as pd
import numpy as np
import streamlit as st
import seaborn as sns

from matplotlib import pyplot as plt
from datetime import datetime

# Load initial dataframe
FILE_PATH = '/Users/jackcook/PycharmProjects/reviews-tool/data/processed_data/df_big_easy_clean.csv'
raw_data = pd.read_csv(FILE_PATH)

@st.cache
def get_date():
    return datetime.now()


# By default, set up dataframe without date filters
if 'review_data' not in st.session_state:
    st.session_state.review_data = raw_data


def get_review_data(start_date, end_date):
    review_data = pd.read_csv(
        '/Users/jackcook/PycharmProjects/reviews-tool/data/processed_data/df_big_easy_clean.csv')
    st.session_state.review_data = review_data.query(f'date_clean >= "{start_date}" and date_clean <= "{end_date}"')


current_date = get_date()
min_date = datetime(2020, 8, 1)

# Set up sidebar with date filters
st.sidebar.write('### Look at reviews from:')
date_start = st.sidebar.date_input(
    '',
    min_value=min_date,
    max_value=current_date,
    value=min_date)
st.sidebar.write('### to:')
date_end = st.sidebar.date_input(
    '',
    min_value=min_date,
    max_value=current_date,
    value=current_date)

# Set up a button to
st.sidebar.button('Click to refresh data',
                  on_click=get_review_data,
                  args=(date_start, date_end))


# Load data
# df_big_easy_clean = pd.read_csv(
#     f'/Users/jackcook/PycharmProjects/reviews-tool/data/processed_data/df_big_easy_clean.csv')
df_big_easy_clean = st.session_state.review_data
df_big_easy_clean['date_clean'] = df_big_easy_clean['date_clean'].astype('datetime64[D]')


# Hack to align title to center
_, header, _ = st.beta_columns((1, 2, 1))
header.write('# High level analysis')
st.write(' ') # Space

df_monthly_output = df_big_easy_clean.set_index('date_clean').resample('M')['reviewRating'].agg(('mean',
                                                                                                 'count')).reset_index()

# Due to lockdown, some months have next to no reviews - set months with 5 or less reviews to score of 0
df_monthly_output['mean'] = np.where(df_monthly_output['count'] > 5,
                                     df_monthly_output['mean'],
                                     0)

# Round 'mean' to 1 decimal place
df_monthly_output['mean'] = df_monthly_output['mean'].map(lambda n: np.around(n, 1))

row1_1, row1_2, row1_3 = st.beta_columns(3)
row1_1.write(f'''**Avg. rating**:\n
{np.around(df_big_easy_clean['reviewRating'].mean(),1)} stars
''')

# Plotting trends over time
# Get colours for bars
pal = sns.color_palette('Blues', df_monthly_output.shape[0])
rank = df_monthly_output['mean'].argsort().argsort()

# Set up sub plots
fig, axs = plt.subplots(nrows=2, ncols=1, constrained_layout=True)

for i in range(2):
    ax = axs[i]
    sns.barplot(data=df_monthly_output,
                x='date_clean',
                y='mean',
                ax=ax,
                palette=np.array(pal)[rank])

    # Get x labels
    x_labels = [d.strftime('%b\n%y') for d in df_monthly_output['date_clean']]
    ax.set_xticklabels(labels=x_labels)

    # Chart labels and axes
    ax.set_ylim((0, 5))
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_yticks([])

    ax.set_title('Average review score (stars) by month\n(months with 5 or less reviews are omitted)')

    # Get data labels
    for p in ax.patches:
        if p.get_height() == 0:
            # Don't label cases where we have removed the review score
            continue
        ax.annotate('{:.1f}'.format(p.get_height()),  # Get value
                    (p.get_x() + p.get_width() / 2., p.get_height()),  # Get position
                    ha='center', va='center', fontsize=10, color='black',
                    xytext=(0, 5), textcoords='offset points')

st.pyplot(fig)

# # Set up column names for plotting
# df_chart = df_monthly_output.copy(deep=True).rename(columns={'date_clean': 'Month',
#                                                              'mean': 'Average review',
#                                                              'count': 'Number of reviews'})

# Plot first chart
# st.altair_chart(alt.Chart(df_chart).mark_bar()
#     .encode(
#         x=alt.X('Month:O'),
#         y=alt.Y('Average review:Q', scale=alt.Scale(domain=[0, 5])),
#         tooltip=['Month', 'Average review']
#     ).configure_mark(
#         opacity=0.9,
#         #color='blue'
#     ).configure_axis(
#         labelFontSize=12,
#         titleFontSize=15,
#         gridOpacity=0.5,
#         labelAngle=0,
#         minExtent=1,
#         labelSeparation=100
#     ).properties(
#         width=800,
#         height=400), use_container_width=True)



