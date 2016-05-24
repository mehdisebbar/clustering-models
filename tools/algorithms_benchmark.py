import plotly.graph_objs as go
from plotly.offline import plot


def view2Ddata(X):
    """
    Plot the sample, Only in 2D
    :param X:
    :return:
    """
    # Create random data with numpy,
    import numpy as np

    N = 1000
    random_x = np.random.randn(N)
    random_y = np.random.randn(N)

    # Create a trace
    trace = go.Scatter(
        x=X[:, 0],
        y=X[:, 1],
        mode='markers'
    )

    data = [trace]

    # Plot and embed in ipython notebook!
    plot(data, filename='Plot')
