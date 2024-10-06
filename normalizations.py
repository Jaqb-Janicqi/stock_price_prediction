def point_relative_normalization(series):
    """
    Computes the normalized value for the values of a
    given series by using the first element of the serie as p_0
    as a reference for each p_i.
    
    Technique comes from Siraj Raval's YouTube video
    "How to Predict Stock Prices Easily - Intro to Deep Learning #7"
    Link: https://www.youtube.com/watch?v=ftMq5ps503w

    series: List with sequential values to use.
    result: List with the normalized results.

    """
    result = (series / series.values[0]) - 1
    return result
