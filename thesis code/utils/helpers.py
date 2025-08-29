def _min_max_scale(values):
    """
    Min max scales all values to be in range [0,1]
        $$X = \frac{X - X_{min}}{X_{max} - X_{min}}$$   

    Parameters:
        values (Pandas Series): Pandas Series containing values to scale.

    Returns:
        Pandas Series with scaled values  
    """
    return (values - values.min()) / (values.max() - values.min())