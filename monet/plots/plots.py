"""Plotting routines for MONET visualization."""

import functools

import matplotlib.pyplot as plt
import seaborn as sns

from . import taylordiagram as td
from .colorbars import colorbar_index

colors = ["#1e90ff", "#DA70D6", "#228B22", "#FA8072", "#FF1493"]


def _default_sns_context(f):
    @functools.wraps(f)
    def inner(*args, **kwargs):
        with sns.plotting_context("poster"), sns.color_palette(colors):
            return f(*args, **kwargs)

    return inner


@_default_sns_context
def make_spatial_plot(modelvar, m, dpi=None, plotargs={}, ncolors=15, discrete=False):
    """Create a spatial plot of model data.

    Parameters
    ----------
    modelvar : numpy.ndarray
        Model variable 2D array to plot.
    m : basemap.Basemap
        Basemap object for mapping.
    dpi : int, optional
        Dots per inch for the figure.
    plotargs : dict, default: {}
        Additional keyword arguments passed to the plotting function.
    ncolors : int, default: 15
        Number of colors for discrete colormap.
    discrete : bool, default: False
        Whether to use a discrete colormap.

    Returns
    -------
    tuple
        Tuple containing (figure, axis, colorbar, colormap, vmin, vmax).
    """
    f, ax = plt.subplots(1, 1, figsize=(11, 6), frameon=False)
    if "cmap" not in plotargs:
        plotargs["cmap"] = "viridis"
    if discrete and "vmin" in plotargs and "vmax" in plotargs:
        c, cmap = colorbar_index(
            ncolors, plotargs["cmap"], minval=plotargs["vmin"], maxval=plotargs["vmax"], basemap=m
        )
        plotargs["cmap"] = cmap
        m.imshow(modelvar, **plotargs)
        vmin, vmax = plotargs["vmin"], plotargs["vmax"]
    elif discrete:
        temp = m.imshow(modelvar, **plotargs)
        vmin, vmax = temp.get_clim()
        c, cmap = colorbar_index(ncolors, plotargs["cmap"], minval=vmin, maxval=vmax, basemap=m)
        plotargs["cmap"] = cmap
        m.imshow(modelvar, vmin=vmin, vmax=vmax, **plotargs)
    else:
        temp = m.imshow(modelvar, **plotargs)
        c = m.colorbar()
        vmin, vmax = temp.get_clim()
        cmap = plotargs["cmap"]
    m.drawstates()
    m.drawcoastlines(linewidth=0.3)
    m.drawcountries()
    return f, ax, c, cmap, vmin, vmax


@_default_sns_context
def spatial(modelvar, **kwargs):
    """Create a simple spatial plot using xarray's built-in plotting.

    Parameters
    ----------
    modelvar : xarray.DataArray
        Model variable to plot.
    **kwargs : dict
        Additional keyword arguments passed to the xarray plot function.

    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the plot.
    """
    if kwargs.get("ax") is None:
        f, ax = plt.subplots(1, 1, figsize=(11, 6), frameon=False)
        kwargs["ax"] = ax
    ax = modelvar.plot(**kwargs)
    plt.tight_layout()
    return ax


@_default_sns_context
def make_spatial_contours(
    modelvar,
    gridobj,
    date,
    m,
    dpi=None,
    savename="",
    discrete=True,
    ncolors=None,
    dtype="int",
    **kwargs
):
    """Create a contour plot of spatial data.

    Parameters
    ----------
    modelvar : numpy.ndarray
        Model variable 2D array to plot.
    gridobj : netCDF4.Dataset or similar
        Grid object containing LAT and LON variables.
    date : datetime.datetime
        Date/time for the title.
    m : basemap.Basemap
        Basemap object for mapping.
    dpi : int, optional
        Dots per inch for saving the figure.
    savename : str, default: ""
        Base name for saving the figure. If empty, figure is not saved.
    discrete : bool, default: True
        Whether to use a discrete colormap.
    ncolors : int, optional
        Number of discrete colors for colorbar.
    dtype : str, default: "int"
        Data type for colorbar labels ('int' or 'float').
    **kwargs : dict
        Additional keyword arguments passed to contourf.

    Returns
    -------
    matplotlib.colorbar.Colorbar
        The colorbar object.
    """
    plt.figure(figsize=(11, 6), frameon=False)
    lat = gridobj.variables["LAT"][0, 0, :, :].squeeze()
    lon = gridobj.variables["LON"][0, 0, :, :].squeeze()
    m.drawstates()
    m.drawcoastlines(linewidth=0.3)
    m.drawcountries()
    x, y = m(lon, lat)
    plt.axis("off")
    m.contourf(x, y, modelvar, **kwargs)
    cmap = kwargs["cmap"]
    levels = kwargs["levels"]
    if discrete:
        c, cmap = colorbar_index(
            ncolors, cmap, minval=levels[0], maxval=levels[-1], basemap=m, dtype=dtype
        )
    else:
        c = m.colorbar()
    titstring = date.strftime("%B %d %Y %H")
    plt.title(titstring)
    plt.tight_layout()
    if savename != "":
        plt.savefig(savename + date.strftime("%Y%m%d_%H.jpg"), dpi=dpi)
        plt.close()
    return c


@_default_sns_context
def wind_quiver(ws, wdir, gridobj, m, **kwargs):
    """Create a quiver plot for wind data.

    Parameters
    ----------
    ws : numpy.ndarray
        Wind speed 2D array.
    wdir : numpy.ndarray
        Wind direction 2D array (in degrees).
    gridobj : netCDF4.Dataset or similar
        Grid object containing LAT and LON variables.
    m : basemap.Basemap
        Basemap object for mapping.
    **kwargs : dict
        Additional keyword arguments passed to the quiver function.

    Returns
    -------
    matplotlib.quiver.Quiver
        The quiver object.
    """
    from . import tools

    lat = gridobj.variables["LAT"][0, 0, :, :].squeeze()
    lon = gridobj.variables["LON"][0, 0, :, :].squeeze()
    x, y = m(lon, lat)
    u, v = tools.wsdir2uv(ws, wdir)
    quiv = m.quiver(x[::15, ::15], y[::15, ::15], u[::15, ::15], v[::15, ::15], **kwargs)
    return quiv


@_default_sns_context
def wind_barbs(ws, wdir, gridobj, m, **kwargs):
    """Create a barbs plot for wind data.

    Parameters
    ----------
    ws : numpy.ndarray
        Wind speed 2D array.
    wdir : numpy.ndarray
        Wind direction 2D array (in degrees).
    gridobj : netCDF4.Dataset or similar
        Grid object containing LAT and LON variables.
    m : basemap.Basemap
        Basemap object for mapping.
    **kwargs : dict
        Additional keyword arguments passed to the barbs function.

    Returns
    -------
    None
    """
    import tools

    lat = gridobj.variables["LAT"][0, 0, :, :].squeeze()
    lon = gridobj.variables["LON"][0, 0, :, :].squeeze()
    x, y = m(lon, lat)
    u, v = tools.wsdir2uv(ws, wdir)
    m.barbs(x[::15, ::15], y[::15, ::15], u[::15, ::15], v[::15, ::15], **kwargs)


def normval(vmin, vmax, cmap):
    """Create a BoundaryNorm for a color map.

    Parameters
    ----------
    vmin : float
        Minimum value for the colorbar.
    vmax : float
        Maximum value for the colorbar.
    cmap : matplotlib.colors.Colormap
        Colormap to use.

    Returns
    -------
    matplotlib.colors.BoundaryNorm
        Boundary normalization object.
    """
    from matplotlib.colors import BoundaryNorm
    from numpy import arange

    bounds = arange(vmin, vmax + 5.0, 5.0)
    norm = BoundaryNorm(boundaries=bounds, ncolors=cmap.N)
    return norm


@_default_sns_context
def spatial_bias_scatter(
    df, m, date, vmin=None, vmax=None, savename="", ncolors=15, fact=1.5, cmap="RdBu_r"
):
    """Create a scatter plot showing spatial bias.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the data with CMAQ, Obs, longitude, latitude columns.
    m : basemap.Basemap
        Basemap object for mapping.
    date : str or datetime.datetime
        Date to filter the DataFrame.
    vmin : float, optional
        Minimum value for colorbar. If None, it's determined from the data.
    vmax : float, optional
        Maximum value for colorbar. If None, it's determined from the data.
    savename : str, default: ""
        Base name for saving the figure. If empty, figure is not saved.
    ncolors : int, default: 15
        Number of discrete colors for colorbar.
    fact : float, default: 1.5
        Factor to scale the point sizes.
    cmap : str, default: "RdBu_r"
        Colormap name to use.

    Returns
    -------
    tuple
        Tuple containing (figure, axis, colorbar).
    """
    from numpy import around
    from scipy.stats import scoreatpercentile as score

    f, ax = plt.subplots(figsize=(11, 6), frameon=False)
    ax.set_facecolor("white")
    diff = df.CMAQ - df.Obs
    top = around(score(diff.abs(), per=95))
    new = df[df.datetime == date]
    x, y = m(new.longitude.values, new.latitude.values)
    c, cmap = colorbar_index(ncolors, cmap, minval=top * -1, maxval=top, basemap=m)

    c.ax.tick_params(labelsize=13)
    colors = new.CMAQ - new.Obs
    ss = (new.CMAQ - new.Obs).abs() / top * 100.0
    ss[ss > 300] = 300.0
    plt.scatter(
        x,
        y,
        c=colors,
        s=ss,
        vmin=-1.0 * top,
        vmax=top,
        cmap=cmap,
        edgecolors="k",
        linewidths=0.25,
        alpha=0.7,
    )

    if savename != "":
        plt.savefig(savename + date + ".jpg", dpi=75.0)
        plt.close()
    return f, ax, c


@_default_sns_context
def timeseries(
    df,
    x="time",
    y="obs",
    ax=None,
    plotargs={},
    fillargs={"alpha": 0.2},
    title="",
    ylabel=None,
    label=None,
):
    """Create a time series plot with error shading.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the data to plot.
    x : str, default: "time"
        Column name to use for the x-axis.
    y : str, default: "obs"
        Column name to use for the y-axis.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, a new axis will be created.
    plotargs : dict, default: {}
        Additional keyword arguments for the line plot.
    fillargs : dict, default: {"alpha": 0.2}
        Additional keyword arguments for the fill_between function.
    title : str, default: ""
        Title for the plot.
    ylabel : str, optional
        Label for the y-axis. If None, uses the variable name and unit.
    label : str, optional
        Label for the line in the legend. If None, uses the y column name.

    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the plot.
    """
    with sns.axes_style("ticks"):
        if ax is None:
            f, ax = plt.subplots(figsize=(11, 6), frameon=False)
        df.index = df[x]
        m = df.groupby("time").mean()
        e = df.groupby("time").std()
        variable = df.variable[0]
        if df.columns.isin(["units"]).max():
            unit = df.units[0]
        else:
            unit = "None"
        upper = m[y] + e[y]
        lower = m[y] - e[y]
        lower.loc[lower < 0] = 0
        lower = lower.values
        if "alpha" not in fillargs:
            fillargs["alpha"] = 0.2
        if label is not None:
            m.rename(columns={y: label}, inplace=True)
        else:
            label = y
        m[label].plot(ax=ax, **plotargs)
        ax.fill_between(m[label].index, lower, upper, **fillargs)
        if ylabel is None:
            ax.set_ylabel(variable + " (" + unit + ")")
        else:
            ax.set_ylabel(label)
        ax.set_xlabel("")
        plt.legend()
        plt.title(title)
        plt.tight_layout()

    return ax


@_default_sns_context
def kdeplot(df, title=None, label=None, ax=None, **kwargs):
    """Create a kernel density estimation plot.

    Parameters
    ----------
    df : pandas.Series or numpy.ndarray
        Data to plot.
    title : str, optional
        Title for the plot.
    label : str, optional
        Label for the line in the legend.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, a new axis will be created.
    **kwargs : dict
        Additional keyword arguments passed to sns.kdeplot.

    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the plot.
    """
    with sns.axes_style("ticks"):
        if ax is None:
            f, ax = plt.subplots(figsize=(11, 6), frameon=False)
            sns.despine()
        ax = sns.kdeplot(df, ax=ax, label=label, **kwargs)

    return ax


@_default_sns_context
def scatter(df, x=None, y=None, title=None, label=None, ax=None, **kwargs):
    """Create a scatter plot with regression line.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the data to plot.
    x : str, optional
        Column name to use for the x-axis. Default is 'obs'.
    y : str, optional
        Column name to use for the y-axis. Default is 'model'.
    title : str, optional
        Title for the plot.
    label : str, optional
        Label for the data in the legend.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, a new axis will be created.
    **kwargs : dict
        Additional keyword arguments passed to sns.regplot.

    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the plot.
    """
    with sns.axes_style("ticks"):
        if ax is None:
            f, ax = plt.subplots(figsize=(8, 6), frameon=False)
        ax = sns.regplot(data=df, x=x, y=y, label=label, **kwargs)
        plt.title(title)

    return ax


@_default_sns_context
def taylordiagram(
    df,
    marker="o",
    col1="obs",
    col2="model",
    label1="OBS",
    label2="MODEL",
    scale=1.5,
    addon=False,
    dia=None,
):
    from numpy import corrcoef

    df = df.drop_duplicates().dropna(subset=[col1, col2])

    if not addon and dia is None:
        with sns.axes_style("ticks"):
            f = plt.figure(figsize=(12, 10))
            obsstd = df[col1].std()

            dia = td.TaylorDiagram(obsstd, scale=scale, fig=f, rect=111, label=label1)
            plt.grid(linewidth=1, alpha=0.5)
            cc = corrcoef(df[col1].values, df[col2].values)[0, 1]
            dia.add_sample(df[col2].std(), cc, marker=marker, zorder=9, ls=None, label=label2)
            contours = dia.add_contours(colors="0.5")
            plt.clabel(contours, inline=1, fontsize=10)
            plt.grid(alpha=0.5)
            plt.legend(fontsize="small", loc="best")

    elif not addon and dia is not None:
        print("Do you want to add this on? if so please turn the addon keyword to True")
    elif addon and dia is None:
        print("Please pass the previous Taylor Diagram Instance with dia keyword...")
    else:
        cc = corrcoef(df.Obs.values, df.CMAQ.values)[0, 1]
        dia.add_sample(df.CMAQ.std(), cc, marker=marker, zorder=9, ls=None, label=label1)
        plt.legend(fontsize="small", loc="best")
        plt.tight_layout()
    return dia