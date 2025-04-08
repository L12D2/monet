"""Colorbar helper functions for MONET plots."""

import matplotlib.pyplot as plt


def colorbar_index(ncolors, cmap, minval=None, maxval=None, dtype="int", basemap=None):
    """Create a colorbar with discrete colors.

    Parameters
    ----------
    ncolors : int
        Number of discrete colors.
    cmap : str or matplotlib.colors.Colormap
        Colormap to discretize.
    minval : float, optional
        Minimum value for the colorbar. If None, uses 0.
    maxval : float, optional
        Maximum value for the colorbar. If None, uses ncolors.
    dtype : str, default: "int"
        Data type for colorbar labels ('int' or 'float').
    basemap : basemap.Basemap, optional
        Basemap instance to add colorbar to. If None, uses plt.colorbar.

    Returns
    -------
    tuple
        (colorbar, discretized_colormap)
    """
    import matplotlib.cm as cm
    import numpy as np

    cmap = cmap_discretize(cmap, ncolors)
    mappable = cm.ScalarMappable(cmap=cmap)
    mappable.set_array([])
    mappable.set_clim(-0.5, ncolors + 0.5)
    if basemap is not None:
        colorbar = basemap.colorbar(mappable, format="%1.2g")
    else:
        colorbar = plt.colorbar(mappable, format="%1.2g", fontsize=12)
    colorbar.set_ticks(np.linspace(0, ncolors, ncolors))
    if (minval is None) & (maxval is not None):
        colorbar.set_ticklabels(np.around(np.linspace(0, maxval, ncolors).astype(dtype), 2))
    elif (minval is None) & (maxval is None):
        colorbar.set_ticklabels(np.around(np.linspace(0, ncolors, ncolors).astype(dtype), 2))
    else:
        colorbar.set_ticklabels(np.around(np.linspace(minval, maxval, ncolors).astype(dtype), 2))

    return colorbar, cmap


def cmap_discretize(cmap, N):
    """Return a discrete colormap from a continuous colormap.

    Parameters
    ----------
    cmap : str or matplotlib.colors.Colormap
        Colormap instance or name to discretize.
    N : int
        Number of colors in the discrete colormap.

    Returns
    -------
    matplotlib.colors.LinearSegmentedColormap
        Discretized colormap.

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> x = np.resize(np.arange(100), (5, 100))
    >>> djet = cmap_discretize('jet', 5)
    >>> plt.imshow(x, cmap=djet)
    """
    import matplotlib.colors as mcolors
    import numpy as np

    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
    colors_i = np.concatenate((np.linspace(0, 1.0, N), (0.0, 0.0, 0.0, 0.0)))
    colors_rgba = cmap(colors_i)
    indices = np.linspace(0, 1.0, N + 1)
    cdict = {}
    for ki, key in enumerate(("red", "green", "blue")):
        cdict[key] = [
            (indices[i], colors_rgba[i - 1, ki], colors_rgba[i, ki]) for i in range(N + 1)
        ]
    # Return colormap object.
    return mcolors.LinearSegmentedColormap(cmap.name + "_%d" % N, cdict, 1024)


# def o3cmap():
#     import matplotlib.cm as cm
#     # This function returns the colormap and bins for the ozone spatial plots
#     # this is designed to have a vmin =0 and vmax = 140
#     # return cmap,bins
#     colors1 = cm.viridis(linspace(0, 1, 128))
#     colors2 = cm.OrRd(linspace(.2, 1, 128))
#     colors = vstack((colors1, colors2))
#     return mcolors.LinearSegmentedColormap.from_list('o3cmap', colors), arange(
#         0, 140.5, .5)
#
#
# def pm25cmap():
#     from matplotlib.cm import viridis, OrRd
#     # This function returns the colormap and bins for the PM spatial plots
#     # this is designed to have a vmin =0 and vmax = 140
#     # return cmap,bins
#     colors1 = viridis(linspace(0, 1, 128))
#     colors2 = OrRd(linspace(.2, 1, 128))
#     colors = vstack((colors1, colors2))
#     cc = mcolors.LinearSegmentedColormap.from_list('pm25cmap', colors), arange(
#         0, 70.2, .2)
#     return cc
#
#
# def wscmap():
#     from matplotlib.cm import viridis, OrRd
#     # This function returns the colormap and bins for the PM spatial plots
#     # this is designed to have a vmin =0 and vmax = 140
#     # return cmap,bins
#     colors1 = viridis(linspace(0, 1, 128))
#     colors2 = OrRd(linspace(.2, 1, 128))
#     colors = vstack((colors1, colors2))
#     return mcolors.LinearSegmentedColormap.from_list('wscmap', colors), arange(
#         0, 40.2, .2)
#
#
# def tempcmap():
#     from matplotlib.cm import viridis, OrRd
#     # This function returns the colormap and bins for the PM spatial plots
#     # this is designed to have a vmin =0 and vmax = 140
#     # return cmap,bins
#     colors1 = viridis(linspace(0, 1, 128))
#     colors2 = OrRd(linspace(.2, 1, 128))
#     colors = vstack((colors1, colors2))
#     return mcolors.LinearSegmentedColormap.from_list('tempcmap',
#                                                      colors), arange(
#                                                          250, 320.5, .5)
#
#
# def sradcmap():
#     from matplotlib.cm import viridis, plasma_r
#     # This function returns the colormap and bins for the PM spatial plots
#     # this is designed to have a vmin =0 and vmax = 140
#     # return cmap,bins
#     colors1 = viridis(linspace(0, 1, 128))
#     colors2 = plasma_r(linspace(.2, 1, 128))
#     colors = vstack((colors1, colors2))
#     return mcolors.LinearSegmentedColormap.from_list('sradcmap',
#                                                      colors), arange(
#                                                          0, 1410., 10)
#
#
# def noxcmap():
#     """Short summary.
#
#     Returns
#     -------
#     type
#         Description of returned object.
#
#     """
#     from matplotlib.cm import viridis, plasma_r
#     # This function returns the colormap and bins for the NO2/NO/NOx spatial plots
#     # this is designed to have a vmin =0 and vmax = 140
#     # return cmap,bins
#     colors1 = viridis(linspace(0, 1, 128))
#     colors2 = plasma_r(linspace(.042, .75, 128))
#     colors = vstack((colors1, colors2))
#     return mcolors.LinearSegmentedColormap.from_list('noxcmap',
#                                                      colors), arange(
#                                                          0, 40.2, .2)
#
#
# def rhcmap():
#     """Short summary.
#
#     Returns
#     -------
#     type
#         Description of returned object.
#
#     """
#     from matplotlib.cm import viridis, plasma_r
#     # This function returns the colormap and bins for the NO2/NO/NOx spatial
#     # plots
#     # this is designed to have a vmin =0 and vmax = 140
#     # return cmap,bins
#     colors1 = viridis(linspace(0, 1, 128))
#     colors2 = plasma_r(linspace(.042, .75, 128))
#     colors = vstack((colors1, colors2))
#     return mcolors.LinearSegmentedColormap.from_list('noxcmap',
#                                                      colors), arange(
#                                                          0, 100.5, .5)
#
#
# def so2cmap():
#     """Short summary.
#
#     Returns
#     -------
#     type
#         Description of returned object.
#
#     """
#     from matplotlib.cm import viridis, plasma_r
#     colors1 = viridis(linspace(0, 1, 128))
#     colors2 = plasma_r(linspace(.042, .75, 128))
#     colors = vstack((colors1, colors2))
#     return mcolors.LinearSegmentedColormap.from_list('noxcmap',
#                                                      colors), arange(
#                                                          0, 14.1, .1)
#
#
# def pm10cmap():
#     import matplotlib.cm as cm
#     # This function returns the colormap and bins for the NO2/NO/NOx spatial plots
#     # this is designed to have a vmin =0 and vmax = 140
#     # return cmap,bins
#     colors1 = cm.viridis(linspace(0, 1, 128))
#     colors2 = cm.plasma_r(linspace(.042, .75, 128))
#     colors = vstack((colors1, colors2))
#     return mcolors.LinearSegmentedColormap.from_list('noxcmap',
#                                                      colors), arange(
#                                                          0, 150.5, .5)
