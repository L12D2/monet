"""Map utilities for creating cartographic plots."""

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt


def draw_map(
    *,
    crs=None,
    natural_earth=False,
    coastlines=True,
    states=False,
    counties=False,
    countries=True,
    resolution="10m",
    extent=None,
    figsize=(10, 5),
    linewidth=0.25,
    return_fig=False,
    **kwargs
):
    """Draw a map with Cartopy.

    Parameters
    ----------
    crs : cartopy.crs.Projection, optional
        The map projection.
        If set, takes precedence over kwargs['subplot_kw']['projection'].
        If unset, defaults to ccrs.PlateCarree().
    natural_earth : bool, default: False
        Add Cartopy Natural Earth ocean, land, lakes, and rivers features.
    coastlines : bool, default: True
        Add coastlines.
    states : bool, default: False
        Add states/provinces boundaries.
    counties : bool, default: False
        Add US counties boundaries.
    countries : bool, default: True
        Add country borders.
    resolution : {'10m', '50m', '110m'}, default: '10m'
        Resolution of Natural Earth features.
    extent : array-like, optional
        Map extent as [lon_min, lon_max, lat_min, lat_max].
    figsize : tuple, default: (10, 5)
        Figure size (width, height) in inches.
    linewidth : float, default: 0.25
        Line width for coastlines, states, counties, and countries.
    return_fig : bool, default: False
        Whether to return both figure and axes objects.
    **kwargs : dict
        Additional arguments passed to plt.subplots().

    Returns
    -------
    matplotlib.axes.Axes or tuple
        If return_fig is False (default), returns just the axes.
        If return_fig is True, returns (fig, ax).
    """
    kwargs["figsize"] = figsize
    if "subplot_kw" not in kwargs:
        kwargs["subplot_kw"] = {}
    if crs is not None:
        kwargs["subplot_kw"]["projection"] = crs
    else:
        if "projection" not in kwargs["subplot_kw"]:
            kwargs["subplot_kw"]["projection"] = ccrs.PlateCarree()

    fig, ax = plt.subplots(**kwargs)

    if natural_earth:
        ax.add_feature(cfeature.OCEAN)
        ax.add_feature(cfeature.LAND)
        ax.add_feature(cfeature.LAKES)
        ax.add_feature(cfeature.RIVERS)

    if states:
        states_provinces = cfeature.NaturalEarthFeature(
            category="cultural",
            name="admin_1_states_provinces_lines",
            scale=resolution,
            facecolor="none",
            edgecolor="k",
        )

    if counties:
        counties = cfeature.NaturalEarthFeature(
            category="cultural",
            name="admin_2_counties",
            scale=resolution,
            facecolor="none",
            edgecolor="k",
        )

    if coastlines:
        ax.coastlines(resolution, linewidth=linewidth)

    if countries:
        ax.add_feature(cfeature.BORDERS, linewidth=linewidth)

    if states:
        ax.add_feature(states_provinces, linewidth=linewidth)

    if counties:
        ax.add_feature(counties, linewidth=linewidth)

    if extent is not None:
        ax.set_extent(extent)

    if return_fig:
        return fig, ax
    else:
        return ax
