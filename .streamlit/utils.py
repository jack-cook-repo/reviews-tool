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

    return fig
