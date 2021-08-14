import streamlit as st
import pandas as pd

from text_utils import get_reviews_formatted
from datetime import datetime
from dateutil.relativedelta import relativedelta


@st.cache
def get_date():
    '''
    Returns current date (hard coded at the moment as we aren't refreshing the file)
    '''
    # return datetime.now()
    return datetime(2021, 7, 29)  # Hard coding as we currently aren't refreshing the file


@st.cache
def get_review_data(date_filt):
    '''
    Filters a dataframe of review data based on a selected filter, and returns the filtered dataframe
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


@st.cache
def get_num_reviews_by_star_rating(df, star_rating_col, agg_col):
    '''
    Given a dataframe, will group by star_rating_col, count the agg_col, and return a dataframe with
    all possible star ratings (from 1-5) and the number of reviews - including zeroes.

    This ensures that we will always have a row of data for each possible star rating.
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


def update_button(bt_show):
    '''
    Updates session state for keys:
     - 'show_button': Used for ordering reviews by date, score ascending, and score descending
     - 'page': Resets it to zero when ordering is changed
    '''
    st.session_state.show_button = bt_show
    st.session_state.page = 0


def show(df):
    '''
    Given an input dataframe of reviews with at least 1 row and the columns:
    'rating', 'review_date', and 'review_highlighted', will create a table
    with custom formatting.

    The columns of this table will be as above, but the advantage of using this method
    is being able to render text with bold highlighting, bullet point formatting
    etc. using st.write() instead of st.table().

    This table also features back/forward buttons and page numbers.

    This will write 10 reviews per table page. If the input dataframe has no rows,
    it will just write a message to that effect and not render anything else.
    '''
    n_rows = df.shape[0]
    if n_rows == 0:
        st.write('No reviews with this topic in the selected timeframe')
    else:
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

        # Write each review up
        col1b, col2b, col3b = st.beta_columns([1, 2, 5])
        col1b.write('### Rating')
        col2b.write('### Date')
        col3b.write('### Review')
        st.write('---')

        # Get reviews
        formatted_reviews = get_reviews_formatted(df.iloc[start:end])
        for dict_rev in formatted_reviews:
            col1b, col2b, col3b = st.beta_columns([1, 2, 5])
            col1b.write(dict_rev['rating'])
            col2b.write(dict_rev['review_date'])
            col3b.write(dict_rev['review_bullets_highlighted'])
            st.write('---------')

        # Put buttons at bottom of table
        col1c, col2c, col3c, _ = st.beta_columns([0.1, 0.17, 0.1, 0.63])
        if st.session_state.page < n_pages-1:
            col3c.button('>', on_click=next_page, key='col3b')
        else:
            col3c.write('')  # this makes the empty column show up on mobile

        if st.session_state.page > 0:
            col1c.button('<', on_click=prev_page, key='col1b')
        else:
            col1c.write('')  # this makes the empty column show up on mobile
        col2c.write(f'Page {1 + st.session_state.page} of {n_pages}')


@st.cache
def get_stars(n):
    return int(n) * '⭐'
