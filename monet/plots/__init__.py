import warnings

from .colorbars import cmap_discretize, colorbar_index
from .mapgen import draw_map
from .plots import (
    kdeplot,
    make_spatial_contours,
    make_spatial_plot,
    normval,
    scatter,
    spatial,
    spatial_bias_scatter,
    taylordiagram,
    timeseries,
    wind_barbs,
    wind_quiver,
)

__all__ = (
    #
    "savefig",
    "sp_scatter_bias",
    #
    "cmap_discretize",
    "colorbar_index",
    #
    "mapgen",
    #
    "kdeplot",
    "make_spatial_contours",
    "make_spatial_plot",
    "normval",
    "scatter",
    "spatial",
    "spatial_bias_scatter",
    "taylordiagram",
    "timeseries",
    "wind_barbs",
    "wind_quiver",
)


# This is the driver for all verify objects


def _dynamic_fig_size(obj):
    """Try to determine a generic figure size based on the shape of obj.

    Parameters
    ----------
    obj : xarray.DataArray
        A 2D xarray DataArray with dimensions that could be
        'x'/'y', 'latitude'/'longitude', or 'lat'/'lon'.

    Returns
    -------
    tuple
        A tuple of (width, height) for the figure size.
    """
    if "x" in obj.dims:
        nx, ny = len(obj.x), len(obj.y)
        scale = float(ny) / float(nx)
    elif "latitude" in obj.dims:
        nx, ny = len(obj.longitude), len(obj.latitude)
        scale = float(ny) / float(nx)
    elif "lat" in obj.dims:
        nx, ny = len(obj.lon), len(obj.lat)
        scale = float(ny) / float(nx)
    figsize = (10, 10 * scale)
    return figsize


def savefig(fname, *, loc=1, decorate=True, logo=None, logo_height=None, **kwargs):
    """Save figure and add logo.

    Parameters
    ----------
    fname : str
        Output file name or path. Passed to ``plt.savefig``.
        Must include desired file extension (``.jpg`` or ``.png``).
    loc : int
        The location for the logo.

        * 1 -- bottom left (default)
        * 2 -- bottom right
        * 3 -- top right
        * 4 -- top left
    decorate : bool, default: True
        Whether to add the logo.
    logo : str, optional
        Path to the logo to be used.
        If not provided, the MONET logo is used.
    logo_height : float or int, optional
        Desired logo height in pixels.
        If not provided, the original logo image dimensions are used.
        Modify to scale the logo.
    **kwargs : dict
        Additional keyword arguments passed to the ``plt.savefig`` function.

    Returns
    -------
    None
    """
    from pathlib import Path

    import matplotlib.pyplot as plt
    from PIL import Image
    from pydecorate import DecoratorAGG

    parts = fname.split(".")
    if not len(parts) > 1:
        raise ValueError("`fname` must include a file extension, e.g. '.png'")
    ext = fname.split(".")[-1]

    # Save current figure
    plt.savefig(fname, **kwargs)

    # Add logo
    if decorate:
        if logo is None:
            logo = Path(__file__).parent / "../data/MONET-logo.png"
        add_logo_kwargs = {}
        if logo_height is not None:
            add_logo_kwargs["height"] = logo_height
        if ext.lower() not in {"png", "jpg", "jpeg"}:
            raise ValueError(f"only PNG and JPEG supported, but detected extension is {ext!r}")

        img = Image.open(fname)
        dc = DecoratorAGG(img)  # cursor starts top-left
        if loc == 1:
            dc.align_bottom()
        elif loc == 2:
            dc.align_bottom()
            dc.align_right()
        elif loc == 3:
            dc.align_right()
        elif loc == 4:
            pass
        else:
            raise ValueError(f"invalid `loc` {loc!r}")
        dc.add_logo(logo, **add_logo_kwargs)

        # PIL.Image will determine format from the filename extension
        img.save(fname)

        img.close()


def sp_scatter_bias(
    df,
    col1=None,
    col2=None,
    ax=None,
    outline=False,
    tight=True,
    global_map=True,
    map_kwargs={},
    cbar_kwargs={},
    val_max=None,
    val_min=None,
    **kwargs,
):
    """Create a scatter plot showing the bias between two columns.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the data to plot.
    col1 : str
        Name of the first column to compare (typically observed values).
    col2 : str
        Name of the second column to compare (typically model values).
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, a new map axis will be created.
    outline : bool, default: False
        Whether to show map outlines.
    tight : bool, default: True
        Whether to use tight layout.
    global_map : bool, default: True
        Whether to set global map limits.
    map_kwargs : dict, default: {}
        Additional keyword arguments for the map creation.
    cbar_kwargs : dict, default: {}
        Additional keyword arguments for the colorbar.
    val_max : float, optional
        Maximum value for colorbar. If None, it's determined from the data.
    val_min : float, optional
        Minimum value for colorbar. If None, it's determined from the data.
    **kwargs : dict
        Additional keyword arguments passed to the scatter plot.

    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the plot.
    """
    import matplotlib.pyplot as plt
    from scipy.stats import scoreatpercentile as score

    if ax is None:
        ax = draw_map(**map_kwargs)
    try:
        if col1 is None or col2 is None:
            print("User must specify col1 and col2 in the dataframe")
            raise ValueError
        else:
            dfnew = df[["latitude", "longitude", col1, col2]].dropna().copy(deep=True)
            dfnew["sp_diff"] = dfnew[col2] - dfnew[col1]
            top = score(dfnew["sp_diff"].abs(), per=95)
            if val_max is not None:
                top = val_max
            # x, y = df.longitude.values, df.latitude.values
            dfnew["sp_diff_size"] = dfnew["sp_diff"].abs() / top * 100.0
            dfnew.loc[dfnew["sp_diff_size"] > 300, "sp_diff_size"] = 300.0
            dfnew.plot.scatter(
                x="longitude",
                y="latitude",
                c=dfnew["sp_diff"],
                s=dfnew["sp_diff_size"],
                vmin=-1 * top,
                vmax=top,
                ax=ax,
                colorbar=True,
                **kwargs,
            )
            if not outline:
                _set_outline_patch_alpha(ax)
            if global_map:
                plt.xlim([-180, 180])
                plt.ylim([-90, 90])
            if tight:
                plt.tight_layout(pad=0)
            return ax
    except ValueError:
        exit


def _set_outline_patch_alpha(ax, alpha=0):
    """Set the alpha value for the outline patch of a cartopy axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        A cartopy GeoAxes instance.
    alpha : float, default: 0
        Alpha transparency value to set (0 = fully transparent, 1 = fully opaque).

    Returns
    -------
    None
    """
    for f in [
        lambda alpha: ax.axes.outline_patch.set_alpha(alpha),
        lambda alpha: ax.outline_patch.set_alpha(alpha),
        lambda alpha: ax.spines["geo"].set_alpha(alpha),
    ]:
        try:
            f(alpha)
        except AttributeError:
            continue
        else:
            break
    else:
        warnings.warn("unable to set outline_patch alpha", stacklevel=2)
