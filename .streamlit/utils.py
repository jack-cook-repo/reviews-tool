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
