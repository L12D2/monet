try:
    from pyresample.geometry import AreaDefinition, SwathDefinition
    from pyresample.kd_tree import XArrayResamplerNN  # noqa: F401

    has_pyresample = True
except ImportError:
    print("PyResample not installed.  Some functionality will be lost")
    has_pyresample = False
try:
    import xesmf  # noqa: F401

    has_xesmf = True
except ImportError:
    has_xesmf = False


def _ensure_swathdef_compatability(defn):
    """Ensures the SwathDefinition is compatible with XArrayResamplerNN.

    Parameters
    ----------
    defn : pyresample.geometry.SwathDefinition
        A SwathDefinition instance to check and potentially modify.

    Returns
    -------
    pyresample.geometry.SwathDefinition
        The original or modified SwathDefinition with xarray.DataArray attributes.
    """
    import xarray as xr

    if isinstance(defn.lons, xr.DataArray):
        return defn  # do nothing
    else:
        defn.lons = xr.DataArray(defn.lons, dims=["y", "x"]).chunk()
        defn.lats = xr.DataArray(defn.lons, dims=["y", "x"]).chunk()
        return defn


def _check_swath_or_area(defn):
    """Checks for SwathDefinition or AreaDefinition compatibility.

    If defn is a SwathDefinition, ensures compatibility with XArrayResamplerNN.
    If defn is an AreaDefinition, returns it unchanged.

    Parameters
    ----------
    defn : pyresample.geometry.SwathDefinition or pyresample.geometry.AreaDefinition
        The definition to check and potentially modify.

    Returns
    -------
    pyresample.geometry.SwathDefinition or pyresample.geometry.AreaDefinition
        The checked and potentially modified definition, compatible with resampling.

    Raises
    ------
    RuntimeError
        If the input is neither a SwathDefinition nor an AreaDefinition.
    """
    try:
        if isinstance(defn, SwathDefinition):
            newswath = _ensure_swathdef_compatability(defn)
        elif isinstance(defn, AreaDefinition):
            newswath = defn
        else:
            raise RuntimeError
    except RuntimeError:
        print("grid definition must be a pyresample SwathDefinition or AreaDefinition")
        return
    return newswath


def _reformat_resampled_data(orig, new, target_grid):
    """Reformats the resampled data array with appropriate coordinates and attributes.

    Parameters
    ----------
    orig : xarray.DataArray
        Original input DataArray before resampling.
    new : xarray.DataArray
        Resampled DataArray that needs reformatting.
    target_grid : pyresample.geometry
        Target SwathDefinition or AreaDefinition with coordinate information.

    Returns
    -------
    xarray.DataArray
        Reformatted DataArray with proper coordinates, name, and attributes.
    """
    target_lon, target_lat = target_grid.get_lonlats_dask()
    new.name = orig.name
    new["latitude"] = (("y", "x"), target_lat)
    new["longitude"] = (("y", "x"), target_lon)
    new.attrs["area"] = target_grid
    return new


def resample_stratify(da, levels, vertical, axis=1):
    """Vertically interpolate data to specified levels.

    Parameters
    ----------
    da : xarray.DataArray
        DataArray containing the data to interpolate.
    levels : array-like
        Target vertical levels to interpolate to.
    vertical : xarray.DataArray
        DataArray containing the vertical coordinate values.
    axis : int, default: 1
        Axis along which to perform the interpolation.

    Returns
    -------
    xarray.DataArray
        Interpolated data at the specified levels.
    """
    import stratify
    import xarray as xr

    result = stratify.interpolate(levels, vertical.chunk().data, da.chunk().data, axis=axis)
    dims = da.dims
    out = xr.DataArray(result, dims=dims, name=da.name)
    out.attrs = da.attrs.copy()
    if len(da.coords) > 0:
        for vn in da.coords:
            if vn != "z" and "z" not in da[vn].dims:
                out[vn] = da[vn].copy()
    return out


def resample_xesmf(source_da, target_da, cleanup=False, **kwargs):
    """Resample data using xESMF regridding.

    Parameters
    ----------
    source_da : xarray.DataArray or xarray.Dataset
        Source data to be regridded.
    target_da : xarray.DataArray or xarray.Dataset
        Target grid definition.
    cleanup : bool, default: False
        Whether to clean up the weight file after regridding.
    **kwargs : dict
        Additional keyword arguments passed to xesmf.Regridder.

    Returns
    -------
    xarray.DataArray or xarray.Dataset
        Regridded data on the target grid.

    Notes
    -----
    Requires xESMF to be installed.
    """
    if has_xesmf:
        import xarray as xr
        import xesmf as xe

        regridder = xe.Regridder(source_da, target_da, **kwargs)
        if cleanup:
            regridder.clean_weight_file()
        if isinstance(source_da, xr.Dataset):
            das = {}
            for name, i in source_da.data_vars.items():
                das[name] = regridder(i)
            ds = xr.Dataset(das)
            ds.attrs = source_da.attrs
            return ds
        else:
            da = regridder(source_da)
            if da.name is None:
                da.name = source_da.name
            return da
