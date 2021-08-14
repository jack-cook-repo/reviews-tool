import re
import matplotlib
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer


def get_dict_terms_to_replace():
    '''
    Utility function, gives you a dict with keys as the terms we want to replace to,
    and items as terms we want to replace from.

    e.g. We'd want to replace Barbecue, B.B.Q, and barbeque with bbq, so this dict
    entry would be 'bbq': r'(barbe(c|q)ue|b.b.q|b(ar)?(\s)?b(\s)?q)'.

    The reason for defining this as a function is to make it usable in 2 separate places,
    1st to clean our text, and 2nd for highlighting text within a review.
    '''

    dict_terms_to_replace = {
        'atmosphere': 'ambience',
        'bbq': r'(barbe(c|q)ue|b\.b\.q|b(ar)?(\s)?b(\s)?q)',
        'staff': r'(server|waiter|waitress)',
        'minutes': 'mins',
        'hours': 'hrs',
        'seafood': r'sea(\s)food',
        'money': r'£[0-9.]+'
    }

    return dict_terms_to_replace


@st.cache
def get_review_topics(df, dict_themed_topics, debug_mode=False):

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
    if debug_mode:
        st.write(df_rating_counts.query('total_reviews >= 10').sort_index())

    return df_rating_counts.query('total_reviews >= 10').sort_index(), df_rating_by_term


@st.cache
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
                f"- **{pct_good_reviews_contained_topic}%** mentioned '{topic}'"]
    elif topic_type == 'bad':
        return [pct_bad_reviews_contained_topic,
                f"- **{pct_bad_reviews_contained_topic}%** mentioned '{topic}'"]
    else:
        # Get the number of reviews for that topic, by star rating
        reviews_by_star = df.loc[topic, 1:5]

        # Take the weighted average to get the review score to 1 decimal place
        weighted_score = np.around(np.sum(reviews_by_star.values * reviews_by_star.index) / np.sum(reviews_by_star.values), 1)

        return [weighted_score,
                f'- **{topic_cased}** ({np.sum(reviews_by_star.values)} reviews) had an average score of {weighted_score} ⭐️']


@st.cache
def get_terms_to_highlight(term, dict_themed_topics, dict_terms_to_replace):
    '''
    Given a term, provides all terms to highlight
    '''

    # If our term covers 1 or more topics, identify all terms to highlight
    if term in dict_themed_topics.keys():
        terms_to_highlight = dict_themed_topics[term]
    else:
        terms_to_highlight = [term]

    # Workaround for phrases with > 1 word:
    terms_to_highlight = [t.replace('mac cheese', r'mac(\s)?(and|&|n)(\s)?cheese') for t in terms_to_highlight]

    # If in our text cleaning process we substituted words (e.g. waiter/waitress --> staff),
    # we still want to highlight those terms in the original review text. To do so, we need
    # to look at what terms we replaced and add those to the list to be highlighted
    for t in terms_to_highlight:
        if t in dict_terms_to_replace.keys():
            # Original regex string
            match_str = dict_terms_to_replace[t]

            # After stripping capture groups and splitting OR logic
            # other_terms_to_highlight = match_str.strip('(').strip(')').split('|')
            terms_to_highlight.append(match_str)

    return terms_to_highlight


@st.cache
def highlight_text(text, term, terms_to_highlight):
    '''
    Given a review body of text, and a term of interest, returns the text with the review
    term highlighted.

    If the term is a topic covering multiple other terms - highlights all terms within
    that topic.
    '''

    # Special case, prevents double highlighting
    if term == 'lobster(s)':
        text = text.replace('lobsters', '**lobsters**')
        text = text.replace('lobster', '**lobster**')
    else:
        # Run through passage and highlight text
        for t in terms_to_highlight:
            # Replace, ignoring case
            text = re.sub(r'(?:(?<!\*\*))(' + t + r')', r'**\1**', text, flags=re.IGNORECASE)

    return text


@st.cache
def get_reviews_formatted(df):
    '''
    Given a dataframe, returns a list of dict, with each dict being a dataframe row's data for:
    - rating column (in stars)
    - review_date column
    - review_highlighted column, split into bullet points by a regex match of !/?/.
    '''
    formatted_reviews = []
    for idx, row in df.iterrows():
        review_bullets = re.split(r'(?<=\.|\!|\?) ', row['review_highlighted'])
        formatted_reviews.append({'rating': row['rating'],
                                  'review_date': row['review_date'],
                                  'review_bullets_highlighted': '- ' + '\n- '.join(review_bullets)})
    return formatted_reviews
