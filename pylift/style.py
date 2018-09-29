import matplotlib.pyplot as plt, matplotlib as mpl

def _plot_defaults(figsize=(15,10), **kwargs):
    """Some aesthetically-pleasing default plot parameters.

    """
    # Change fonts to serif.
    mpl.rcParams['mathtext.fontset'] = 'stix'
    mpl.rcParams['font.family'] = 'STIXGeneral'

    mpl.rcParams['font.size'] = 20
    fig, ax = plt.subplots(figsize=figsize, **kwargs)
    return fig, ax
