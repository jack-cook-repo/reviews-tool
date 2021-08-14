import matplotlib
import streamlit as st
import seaborn as sns
import numpy as np

from matplotlib import pyplot as plt


mpl_hash_funcs = {matplotlib.figure.Figure: hash}


@st.cache(hash_funcs=mpl_hash_funcs, allow_output_mutation=True)
def plot_reviews_by_month(color_scheme, df, date_col, scores_or_counts,
                          num_col, figsize=(6, 4)):
    '''
    Given a Matplotlib axis, and a colour scheme, input dataframe (with a date column and a review scores or
    count column), plus optional y limits, plots the numerical by month barplot onto the axis
    '''

    assert scores_or_counts in ('scores', 'counts')

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    # Set up colour palette and rankings to select colours from each palette, and ylim
    if scores_or_counts == 'scores':
        pal_colour = [0 if rev != rev else int(rev * 10) for rev in df[num_col]]
        pal = sns.color_palette(color_scheme, n_colors=51)
        ylim = (0, 5.5)
        label_fmt = '{:.1f}'
    else:
        max_n_reviews = df[num_col].max()
        pal_colour = [0 if num != num else num for num in df[num_col]]
        pal = sns.color_palette(color_scheme, n_colors=max_n_reviews+1)
        ylim = (0, int(max_n_reviews*1.3))
        label_fmt = '{:.0f}'

    if df.shape[0] > 0:
        # Plot
        sns.barplot(data=df,
                    x=date_col,
                    y=num_col,
                    ax=ax,
                    palette=np.array(pal)[pal_colour])

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
    title_insert = 'Number of reviews' if scores_or_counts == 'counts' else 'Average review score (stars)'
    ax.set_title(f'{title_insert} by month')

    # Get data labels
    for p in ax.patches:
        if p.get_height() == 0:
            # Don't label cases where we have removed the review score
            continue
        ax.annotate(label_fmt.format(p.get_height()),  # Get value
                    (p.get_x() + p.get_width() / 2., p.get_height()),  # Get position
                    ha='center', va='center', color='black',
                    xytext=(0, 10), textcoords='offset points')

        # Get current width
        current_width = p.get_width()
        diff = current_width - 1

        # we change the bar width
        p.set_width(1)

        # we recenter the bar
        p.set_x(p.get_x() + diff * .5)

    return fig


@st.cache(hash_funcs=mpl_hash_funcs, allow_output_mutation=True)
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
    # Get data labels
    for p in ax.patches:
        if p.get_height() == 0:
            # Don't label cases where we have removed the review score
            continue
        ax.annotate('{:.0f}'.format(p.get_height()),  # Get value
                    (p.get_x() + p.get_width() / 2., p.get_height()),  # Get position
                    ha='center', va='center', color='black',
                    xytext=(0, 10), textcoords='offset points')

        # Get current width
        current_width = p.get_width()
        diff = current_width - 1

        # we change the bar width
        p.set_width(1)

        # we recenter the bar
        p.set_x(p.get_x() + diff * .5)

    return fig