"MONET Accessor"

import numpy as np
import pandas as pd
import xarray as xr

from .plots import _set_outline_patch_alpha

try:
    import xesmf  # noqa: F401

    has_xesmf = True
except ImportError:
    has_xesmf = False
try:
    import pyresample as pr
    from pyresample.utils import wrap_longitudes

    has_pyresample = True
except ImportError:
    has_pyresample = False

    def wrap_longitudes(lons):
        """For longitudes that may be in [0, 360) format, return in [-180, 180) format.

        Parameters
        ----------
        lons : array-like
            Longitude values to wrap.

        Returns
        -------
        array-like
            Wrapped longitude values in range [-180, 180).
        """
        return (lons + 180) % 360 - 180


def _rename_latlon(ds):
    """Rename latitude/longitude variants to ``lat``/``lon``,
    returning a new xarray object.

    Parameters
    ----------
    ds : xarray.DataArray or xarray.Dataset
        Dataset with latitude/longitude coordinates to rename.

    Returns
    -------
    xarray.DataArray or xarray.Dataset
        Dataset with renamed coordinates.
    """
    if "latitude" in ds.coords:
        return ds.rename({"latitude": "lat", "longitude": "lon"})
    elif "Latitude" in ds.coords:
        return ds.rename({"Latitude": "lat", "Longitude": "lon"})
    elif "Lat" in ds.coords:
        return ds.rename({"Lat": "lat", "Lon": "lon"})
    else:
        return ds


def _monet_to_latlon(da):
    """Convert from MONET-style coordinates to standard lat/lon coordinates.

    Parameters
    ----------
    da : xarray.DataArray
        DataArray with MONET-style coordinates.

    Returns
    -------
    xarray.DataArray or xarray.Dataset
        DataArray or Dataset with standard lat/lon coordinates.
    """
    if isinstance(da, xr.DataArray):
        dset = da.to_dataset()
    dset["x"] = da.longitude[0, :].values
    dset["y"] = da.latitude[:, 0].values
    dset = dset.drop_vars(["latitude", "longitude"])
    dset = dset.set_coords(["x", "y"])
    dset = dset.rename({"x": "lon", "y": "lat"})
    if isinstance(da, xr.DataArray):
        return dset[da.name]
    else:
        return dset


def _dataset_to_monet(
    dset,
    lat_name="latitude",
    lon_name="longitude",
    latlon2d=None,
    lon180=None,
):
    """Rename xarray DataArray or Dataset coordinate variables for use with monet functions,
    returning a new xarray object.

    Parameters
    ----------
    dset : xarray.DataArray or xarray.Dataset
        A given data obj to be renamed for monet.
    lat_name : str, default: "latitude"
        Name of the latitude array.
    lon_name : str, default: "longitude"
        Name of the longitude array.
    latlon2d : bool, optional
        Whether the latitude and longitude data is two-dimensional.
        If unset (``None``), guess based on dim count.
    lon180 : bool, optional
        Whether the longitude values are in the range [-180, 180) already.
        If true, longitude wrapping/normalization,
        which can introduce small floating point errors, will be skipped.
        If unset (``None``), compute min/max to determine.

    Returns
    -------
    xarray.DataArray or xarray.Dataset
        The input with coordinates renamed for MONET compatibility.

    Raises
    ------
    TypeError
        If dset is not an xarray DataArray or Dataset.
    """
    if not isinstance(dset, (xr.DataArray, xr.Dataset)):
        raise TypeError("dset must be an xarray.DataArray or xarray.Dataset")

    if "grid_xt" in dset.dims:  # GFS v16 file
        if isinstance(dset, xr.DataArray):
            dset = _dataarray_coards_to_netcdf(dset, lat_name="grid_yt", lon_name="grid_xt")
        elif isinstance(dset, xr.Dataset):
            dset = _dataarray_coards_to_netcdf(dset, lat_name="grid_yt", lon_name="grid_xt")

    if "south_north" in dset.dims:  # WRF WPS file
        dset = dset.rename(dict(south_north="y", west_east="x"))
        if isinstance(dset, xr.Dataset):
            if "XLAT_M" in dset.data_vars:
                dset["XLAT_M"] = dset.XLAT_M.squeeze()
                dset["XLONG_M"] = dset.XLONG_M.squeeze()
                dset = dset.set_coords(["XLAT_M", "XLONG_M"])
            elif "XLAT" in dset.data_vars:
                dset["XLAT"] = dset.XLAT.squeeze()
                dset["XLONG"] = dset.XLONG.squeeze()
                dset = dset.set_coords(["XLAT", "XLONG"])
        elif isinstance(dset, xr.DataArray):
            if "XLAT_M" in dset.coords:
                dset["XLAT_M"] = dset.XLAT_M.squeeze()
                dset["XLONG_M"] = dset.XLONG_M.squeeze()
            elif "XLAT" in dset.coords:
                dset["XLAT"] = dset.XLAT.squeeze()
                dset["XLONG"] = dset.XLONG.squeeze()

    # Rename lat/lon coordinates to 'latitude'/'longitude'
    dset = _rename_to_monet_latlon(dset)  # common cases
    if (isinstance(dset, xr.Dataset) and not {"latitude", "longitude"} <= set(dset.variables)) or (
        isinstance(dset, xr.DataArray) and not {"latitude", "longitude"} <= set(dset.coords)
    ):
        dset = dset.rename({lat_name: "latitude", lon_name: "longitude"})

    # Maybe wrap longitudes
    if lon180 is None:
        lon180 = dset["longitude"].min() >= -180 and dset["longitude"].max() < 180
    if not lon180:
        dset["longitude"] = wrap_longitudes(dset["longitude"])

    # lat & lon are not coordinate variables in unstructured grid, so we're done
    if dset.attrs.get("mio_has_unstructured_grid", False):
        return dset

    # Maybe convert 1-D lat/lon coords to 2-D
    if latlon2d is None:
        latlon2d = dset["latitude"].ndim >= 2
    if not latlon2d:
        if isinstance(dset, xr.DataArray):
            dset = _dataarray_coards_to_netcdf(dset, lat_name="latitude", lon_name="longitude")
        elif isinstance(dset, xr.Dataset):
            dset = _coards_to_netcdf(dset, lat_name="latitude", lon_name="longitude")

    return dset


def _rename_to_monet_latlon(ds):
    """Rename latitude/longitude variants to ``latitude``/``longitude``,
    returning a new xarray object.

    Parameters
    ----------
    ds : xarray.DataArray or xarray.Dataset
        Dataset with latitude/longitude coordinates to rename.

    Returns
    -------
    xarray.DataArray or xarray.Dataset
        Dataset with renamed coordinates.
    """
    # To consider unstructured grid
    if ds.attrs.get("mio_has_unstructured_grid", False):
        check_list = ds.data_vars
    else:
        check_list = ds.coords

    if "lat" in check_list:
        return ds.rename({"lat": "latitude", "lon": "longitude"})
    elif "Latitude" in check_list:
        return ds.rename({"Latitude": "latitude", "Longitude": "longitude"})
    elif "Lat" in check_list:
        return ds.rename({"Lat": "latitude", "Lon": "longitude"})
    elif "XLAT_M" in check_list:
        return ds.rename({"XLAT_M": "latitude", "XLONG_M": "longitude"})
    elif "XLAT" in check_list:
        return ds.rename({"XLAT": "latitude", "XLONG": "longitude"})
    else:
        return ds.copy()


def _coards_to_netcdf(dset, lat_name="lat", lon_name="lon"):
    """Convert 1-D lat/lon coords to x/y convention, with
    lat/lon as 2-D variables with (y, x) dimensions,
    returning a new xarray object.

    Parameters
    ----------
    dset : xarray.Dataset
        Dataset with 1D lat/lon coordinates.
    lat_name : str, default: "lat"
        Name of the latitude coordinate.
    lon_name : str, default: "lon"
        Name of the longitude coordinate.

    Returns
    -------
    xarray.Dataset
        Dataset with 2D lat/lon coordinates.
    """
    from numpy import arange, meshgrid

    lon = dset[lon_name]
    lat = dset[lat_name]
    lons, lats = meshgrid(lon, lat)
    x = arange(len(lon))
    y = arange(len(lat))
    dset = dset.rename({lon_name: "x", lat_name: "y"})
    dset.coords["longitude"] = (("y", "x"), lons)
    dset.coords["latitude"] = (("y", "x"), lats)
    dset["x"] = x
    dset["y"] = y
    dset = dset.set_coords(["latitude", "longitude"])
    return dset


def _dataarray_coards_to_netcdf(dset, lat_name="lat", lon_name="lon"):
    """Convert 1-D lat/lon coords to x/y convention, with
    lat/lon as 2-D variables with (y, x) dimensions,
    returning a new xarray object.

    Parameters
    ----------
    dset : xarray.DataArray
        DataArray with 1D lat/lon coordinates.
    lat_name : str, default: "lat"
        Name of the latitude coordinate.
    lon_name : str, default: "lon"
        Name of the longitude coordinate.

    Returns
    -------
    xarray.DataArray
        DataArray with 2D lat/lon coordinates.
    """
    from numpy import arange, meshgrid

    lon = dset[lon_name]
    lat = dset[lat_name]
    lons, lats = meshgrid(lon, lat)
    x = arange(len(lon))
    y = arange(len(lat))
    dset = dset.rename({lon_name: "x", lat_name: "y"})
    dset.coords["latitude"] = (("y", "x"), lats)
    dset.coords["longitude"] = (("y", "x"), lons)
    dset["x"] = x
    dset["y"] = y
    return dset


@pd.api.extensions.register_dataframe_accessor("monet")
class MONETAccessorPandas:
    """Pandas DataFrame accessor for MONET functionality.

    This accessor adds MONET-specific methods to pandas DataFrames.
    """

    def __init__(self, pandas_obj):
        """Initialize the accessor.

        Parameters
        ----------
        pandas_obj : pandas.DataFrame
            The pandas DataFrame this accessor will work with.
        """
        self._validate(pandas_obj)
        self._obj = pandas_obj

    @staticmethod
    def _validate(obj):
        """Verify there is a column ``'latitude'`` and a column ``'longitude'``.

        Parameters
        ----------
        obj : pandas.DataFrame
            Object to validate.

        Raises
        ------
        AttributeError
            If obj does not have 'latitude' and 'longitude' columns.
        """
        if "latitude" not in obj.columns or "longitude" not in obj.columns:
            raise AttributeError("Must have 'latitude' and 'longitude'.")

    @property
    def center(self):
        """The geographic center point of this DataFrame.

        .. note::

           Currently just the lon and lat mean values,
           not necessarily representative of the geographic center.

        Returns
        -------
        tuple
            (lon, lat)
        """
        lat = self._obj.latitude
        lon = self._obj.longitude
        return (float(lon.mean()), float(lat.mean()))
        # TODO: won't necessarily represent geographic center

    def to_ascii2nc_df(
        self,
        grib_code=126,
        height_msl=0.0,
        column="aod_550nm",
        message_type="ADPUPA",
        pressure=1000.0,
        qc=None,
        height_agl=None,
    ):
        """Convert DataFrame to MET ASCII2NC format.

        Parameters
        ----------
        grib_code : int, default: 126
            GRIB code for the variable.
        height_msl : float or str, default: 0.0
            Height above mean sea level (meters).
        column : str, default: "aod_550nm"
            Column name from which to get observation values.
        message_type : str, default: "ADPUPA"
            Message type for MET.
        pressure : float or str, default: 1000.0
            Pressure level (mb).
        qc : str, optional
            Quality control value. If None, defaults to 0.
        height_agl : float or str, optional
            Height above ground level (meters). If None, defaults to 0.

        Returns
        -------
        pandas.DataFrame
            DataFrame formatted for MET ASCII2NC.
        """
        df = self._obj
        df["ascii2nc_time"] = df.time.dt.strftime("%Y%m%d_%H%M%S")
        df["ascii2nc_gribcode"] = int(grib_code)
        if isinstance(height_msl, str):
            df["ascii2nc_elevation"] = df[height_msl]
        else:
            df["ascii2nc_elevation"] = height_msl
        df["ascii2nc_message"] = message_type
        if isinstance(pressure, str):
            df["ascii2nc_pressure"] = df[pressure]
        else:
            df["ascii2nc_pressure"] = pressure
        df["ascii2nc_value"] = df[column]
        if qc is None:
            df["ascii2nc_qc"] = "0"
            df.loc[df["ascii2nc_value"].isnull(), "ascii2nc_qc"] = "1"
        else:
            df["ascii2nc_qc"] = "0"
        if height_agl is None:
            df["ascii2nc_height_agl"] = df["ascii2nc_elevation"]
        elif isinstance(height_agl, str):
            df["ascii2nc_height_agl"] = df[height_agl]
        else:
            df["ascii2nc_height_agl"] = height_agl
        out = df[
            [
                "ascii2nc_message",
                "siteid",
                "ascii2nc_time",
                "latitude",
                "longitude",
                "ascii2nc_elevation",
                "ascii2nc_gribcode",
                "ascii2nc_pressure",
                "ascii2nc_height_agl",
                "ascii2nc_qc",
                "ascii2nc_value",
            ]
        ]
        out = out.rename(
            dict(
                ascii2nc_message="typ",
                siteid="sid",
                ascii2nc_time="vld",
                latitude="lat",
                longitude="lon",
                ascii2nc_elevation="elv",
                ascii2nc_gribcode="var",
                ascii2nc_pressure="lvl",
                ascii2nc_height_agl="lvl",
                ascii2nc_qc="qc",
                ascii2nc_value="obs",
            ),
            axis=1,
        )
        out = out.astype(dict(typ=str, sid=str, vld=str, var=str, qc=str))
        return out

    def to_ascii2nc_list(self, **kwargs):
        """Convert DataFrame to MET ASCII2NC list format.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments passed to to_ascii2nc_df().

        Returns
        -------
        list of list
            List of lists formatted for MET ASCII2NC.
        """
        out = self.to_ascii2nc_df(**kwargs)
        return out.values.tolist()

    def rename_for_monet(self, df=None):
        """Rename latitude and longitude columns in the DataFrame.

        Parameters
        ----------
        df : pandas.DataFrame, optional
            To use instead of self.

        Returns
        -------
        pandas.DataFrame
            DataFrame with renamed latitude/longitude columns.
        """
        if df is None:
            df = self._obj
        if "lat" in df.columns:
            df = df.rename({"lat": "latitude", "lon": "longitude"})
        elif "Latitude" in df.columns:
            df = df.rename({"Latitude": "latitude", "Longitude": "longitude"})
        elif "Lat" in df.columns:
            df = df.rename({"Lat": "latitude", "Lon": "longitude"})
        elif "LAT" in df.columns:
            df = df.rename({"LAT": "latitude", "LON": "longitude"})
        return df

    def get_sparse_SwathDefinition(self):
        """Creates a ``pyreample.geometry.SwathDefinition`` for a single point.

        Returns
        -------
        pyreample.geometry.SwathDefinition
            SwathDefinition object for data points.
        """
        df = self.rename_for_monet(self._obj)
        if has_pyresample:
            from .util.interp_util import nearest_point_swathdefinition as npsd

            return npsd(latitude=df.latitude.values, longitude=df.longitude.values)

    def _df_to_da(self, d=None):  # TODO: should be `to_ds` or `to_xarray`
        """Convert DataFrame to xarray.

        Parameters
        ----------
        d : pandas.DataFrame, optional
            To use instead of self.

        Returns
        -------
        xarray.Dataset
        """
        index_name = "index"
        if d is None:
            d = self._obj
        if d.index.name is not None:
            index_name = d.index.name
        ds = d.to_xarray().rename({index_name: "x"}).expand_dims("y")
        if "time" in ds.data_vars.keys():
            ds["time"] = ds.time.squeeze()  # it is only 1D
        if "latitude" in ds.data_vars.keys():
            ds = ds.set_coords(["latitude", "longitude"])
        return ds

    def remap_nearest(
        self,
        df,
        radius_of_influence=1e5,
        combine=False,
    ):
        """Remap data using nearest neighbor interpolation.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame to remap.
        radius_of_influence : float, default: 1e5
            Search radius in meters.
        combine : bool, default: False
            Whether to combine the remapped data with the original data.

        Returns
        -------
        pandas.DataFrame
            Remapped DataFrame.
        """
        source_data = self.rename_for_monet(df)
        target_data = self.rename_for_monet(self._obj)
        # make fake index
        if has_pyresample:
            source_data = self._make_fake_index_var(source_data)
            source_data_da = self._df_to_da(source_data)
            target_data_da = self._df_to_da(target_data)
            source = source_data_da.monet._get_CoordinateDefinition(source_data_da)
            target = target_data_da.monet._get_CoordinateDefinition(target_data_da)
            res = pr.kd_tree.XArrayResamplerNN(
                source, target, radius_of_influence=radius_of_influence
            )
            res.get_neighbour_info()
            # interpolate just the make_fake_index variable
            # print(ds1)
            r = res.get_sample_from_neighbour_info(source_data_da.monet_fake_index)
            r.name = "monet_fake_index"
            # r = ds2.monet.remap_nearest(
            #     ds1, radius_of_influence=radius_of_influence)
            # now merge back from original DataFrame
            q = r.compute()
            v = q.squeeze().to_dataframe()
            result = v.merge(source_data, how="left", on="monet_fake_index").drop(
                "monet_fake_index", axis=1
            )
            if combine:
                columns_to_use = result.columns.difference(target_data.columns)
                return pd.merge(
                    target_data,
                    result[columns_to_use],
                    left_index=True,
                    right_index=True,
                    how="outer",
                )
            else:
                return result

        else:
            print("pyresample unavailable. Try `import pyresample` and check the failure message.")

    def cftime_to_datetime64(self, col=None):
        """Convert cftime column to numpy datetime64.

        Parameters
        ----------
        col : str, optional
            Name of the column to convert. If None, tries to detect the time column.

        Returns
        -------
        pandas.DataFrame
            DataFrame with converted time column.
        """
        df = self._obj

        def cf_to_dt64(x):
            return pd.to_datetime(x.strftime("%Y-%m-%d %H:%M:%S"))

        if col is None:  # assume 'time' is the column name to transform
            col = "time"
        df[col] = df[col].apply(cf_to_dt64)
        return df

    def _make_fake_index_var(self, df):
        """Create a fake index variable for the DataFrame.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame to create index for.

        Returns
        -------
        pandas.DataFrame
            DataFrame with fake index column added.
        """
        from numpy import arange

        # column = df.columns[0]
        fake_index = arange(len(df))
        column_name = "monet_fake_index"
        r = pd.Series(fake_index.astype(float), index=df.index)
        r.name = column_name
        df[column_name] = r
        return df


@xr.register_dataarray_accessor("monet")
class MONETAccessor:
    """DataArray accessor for MONET functionality."""

    def __init__(self, xray_obj):
        """Initialize the accessor.

        Parameters
        ----------
        xray_obj : xarray.DataArray
            The DataArray this accessor will work with.
        """
        self._obj = xray_obj

    def wrap_longitudes(self, lon_name="longitude"):
        """Wrap longitude values to [-180, 180).

        Parameters
        ----------
        lon_name : str, default: "longitude"
            Name of the longitude coordinate.

        Returns
        -------
        xarray.DataArray
            DataArray with wrapped longitudes.
        """
        dset = self._obj
        dset[lon_name] = (dset[lon_name] + 180) % 360 - 180
        return dset

    def tidy(self, lon_name="longitude"):
        """Apply tidying operations to the data.

        Parameters
        ----------
        lon_name : str, default: "longitude"
            Name of the longitude coordinate.

        Returns
        -------
        xarray.DataArray
            Tidied DataArray.
        """
        d = self._obj
        wd = d.monet.wrap_longitudes(lon_name=lon_name)
        wdl = wd.sortby(wd[lon_name])
        return wdl

    def is_land(self, return_xarray=False):
        """Check if points are on land.

        Parameters
        ----------
        return_xarray : bool, default: False
            If True, return results as xarray. Otherwise, return numpy array.

        Returns
        -------
        xarray.DataArray or numpy.ndarray
            Boolean array where True indicates land.
        """
        try:
            import global_land_mask as glm
        except ImportError:
            print("Please install global_land_mask from pypi")
        da = _dataset_to_monet(self._obj)
        island = glm.is_land(da.latitude.values, da.longitude.values)
        if return_xarray:
            return da.where(island)
        else:
            return island

    def is_ocean(self, return_xarray=False):
        """Check if points are on ocean.

        Parameters
        ----------
        return_xarray : bool, default: False
            If True, return results as xarray. Otherwise, return numpy array.

        Returns
        -------
        xarray.DataArray or numpy.ndarray
            Boolean array where True indicates ocean.
        """
        try:
            import global_land_mask as glm
        except ImportError:
            print("Please install global_land_mask from pypi")
        da = _dataset_to_monet(self._obj)
        isocean = glm.is_ocean(da.latitude.values, da.longitude.values)
        if return_xarray:
            return da.where(isocean)
        else:
            return isocean

    def cftime_to_datetime64(self, name=None):
        """Convert cftime coordinates to numpy datetime64.

        Parameters
        ----------
        name : str, optional
            Name of the coordinate to convert. If None, tries to detect the time coordinate.

        Returns
        -------
        xarray.DataArray
            DataArray with converted time coordinate.
        """
        from numpy import vectorize

        da = self._obj

        def cf_to_dt64(x):
            return pd.to_datetime(x.strftime("%Y-%m-%d %H:%M:%S"))

        if name is None:  # assume 'time' is the column name to transform
            name = "time"
        if isinstance(da[name].to_index(), xr.CFTimeIndex):
            da[name] = xr.apply_ufunc(vectorize(cf_to_dt64), da[name])
        return da

    def structure_for_monet(self, lat_name="lat", lon_name="lon", return_obj=True):
        """Structure the DataArray for use with MONET functions.

        Parameters
        ----------
        lat_name : str, default: "lat"
            Name of the latitude coordinate.
        lon_name : str, default: "lon"
            Name of the longitude coordinate.
        return_obj : bool, default: True
            Whether to return the restructured object.

        Returns
        -------
        xarray.DataArray or None
            Restructured DataArray if return_obj is True, otherwise None.
        """
        if return_obj:
            return _dataset_to_monet(self._obj, lat_name=lat_name, lon_name=lon_name)
        else:
            self._obj = _dataset_to_monet(self._obj, lat_name=lat_name, lon_name=lon_name)

    def stratify(self, levels, vertical, axis=1):
        """Vertically interpolate data to specified levels.

        Parameters
        ----------
        levels : array-like
            Target vertical levels.
        vertical : xarray.DataArray
            Vertical coordinate values.
        axis : int, default: 1
            Axis along which to interpolate.

        Returns
        -------
        xarray.DataArray
            Vertically interpolated data.
        """
        from .util.resample import resample_stratify

        if isinstance(vertical, str):
            vertical = self._obj[vertical]

        out = resample_stratify(self._obj, levels, vertical, axis=axis)
        return out

    def window(self, lat_min=None, lon_min=None, lat_max=None, lon_max=None, rectilinear=False):
        """Extract a spatial window from the data.

        Parameters
        ----------
        lat_min : float, optional
            Minimum latitude.
        lon_min : float, optional
            Minimum longitude.
        lat_max : float, optional
            Maximum latitude.
        lon_max : float, optional
            Maximum longitude.
        rectilinear : bool, default: False
            Whether the grid is rectilinear.

        Returns
        -------
        xarray.DataArray
            Windowed DataArray.
        """
        try:
            from numpy import concatenate
            from pyresample import utils

            has_pyresample = True
        except ImportError:
            has_pyresample = False
        try:
            if rectilinear:
                dset = _dataset_to_monet(self._obj)
                lat = dset.latitude.isel(x=0).values
                lon = dset.longitude.isel(y=0).values
                dset["x"] = lon
                dset["y"] = lat
                # dset = dset.drop(['latitude', 'longitude'])
                # check if latitude is in the correct order
                if dset.latitude.isel(x=0).values[0] > dset.latitude.isel(x=0).values[-1]:
                    lat_min_copy = lat_min
                    lat_min = lat_max
                    lat_max = lat_min_copy
                d = dset.sel(x=slice(lon_min, lon_max), y=slice(lat_min, lat_max))
                return d
            elif has_pyresample:
                dset = _dataset_to_monet(self._obj)
                lons, lats = utils.check_and_wrap(dset.longitude.values, dset.latitude.values)
                x_ll, y_ll = dset.monet.nearest_ij(lat=float(lat_min), lon=float(lon_min))
                x_ur, y_ur = dset.monet.nearest_ij(lat=float(lat_max), lon=float(lon_max))
                if x_ur < x_ll:
                    x1 = dset.x.where(dset.x >= x_ll, drop=True).values
                    x2 = dset.x.where(dset.x <= x_ur, drop=True).values
                    xrange = concatenate([x1, x2]).astype(int)
                    dset["longitude"][:] = utils.wrap_longitudes(dset.longitude.values)
                else:
                    xrange = slice(x_ll, x_ur)
                if y_ur < y_ll:
                    y1 = dset.y.where(dset.y >= y_ll, drop=True).values
                    y2 = dset.y.where(dset.y <= y_ur, drop=True).values
                    yrange = concatenate([y1, y2]).astype(int)
                else:
                    yrange = slice(y_ll, y_ur)
                return dset.isel(x=xrange, y=yrange)
            else:
                raise ImportError
        except ImportError:
            print(
                """If this is a rectilinear grid and you don't have pyresample
                  please add the rectilinear=True to the call.  Otherwise the window
                  functionality is unavailable without pyresample"""
            )

    def interp_constant_lat(self, lat=None, lat_name="latitude", lon_name="longitude", **kwargs):
        """Interpolate data to a constant latitude.

        Parameters
        ----------
        lat : float, optional
            Latitude value to interpolate to.
        lat_name : str, default: "latitude"
            Name of the latitude coordinate.
        lon_name : str, default: "longitude"
            Name of the longitude coordinate.
        **kwargs : dict
            Additional keyword arguments for interpolation.

        Returns
        -------
        xarray.DataArray
            Interpolated DataArray.
        """
        from numpy import asarray, linspace, ones

        if has_xesmf:
            from .util.interp_util import constant_1d_xesmf
            from .util.resample import resample_xesmf

        try:
            if lat is None:
                raise RuntimeError
        except RuntimeError:
            print("Must enter lat value")
        d1 = _dataset_to_monet(self._obj, lat_name=lat_name, lon_name=lon_name)
        longitude = linspace(d1.longitude.min(), d1.longitude.max(), len(d1.x))
        latitude = ones(longitude.shape) * asarray(lat)
        if has_pyresample:
            d2 = xr.DataArray(
                ones((len(longitude), len(longitude))),
                dims=["lon", "lat"],
                coords=[longitude, latitude],
            )
            d2 = _dataset_to_monet(d2)
            result = d2.monet.remap_nearest(d1)
            return result.isel(y=0)
        elif has_xesmf:
            output = constant_1d_xesmf(latitude=latitude, longitude=longitude)
            out = resample_xesmf(self._obj, output, **kwargs)
            return _rename_latlon(out)

    def interp_constant_lon(self, lon=None, **kwargs):
        """Interpolate data to a constant longitude.

        Parameters
        ----------
        lon : float, optional
            Longitude value to interpolate to.
        **kwargs : dict
            Additional keyword arguments for interpolation.

        Returns
        -------
        xarray.DataArray
            Interpolated DataArray.
        """
        if has_xesmf:
            from .util.interp_util import constant_1d_xesmf
            from .util.resample import resample_xesmf
        from numpy import asarray, linspace, ones

        try:
            if lon is None:
                raise RuntimeError
        except RuntimeError:
            print("Must enter lon value")
        d1 = _dataset_to_monet(self._obj)
        latitude = linspace(d1.latitude.min(), d1.latitude.max(), len(d1.y))
        longitude = ones(latitude.shape) * asarray(lon)
        if has_pyresample:
            if has_pyresample:
                d2 = xr.DataArray(
                    ones((len(longitude), len(longitude))),
                    dims=["lon", "lat"],
                    coords=[longitude, latitude],
                )
                d2 = _dataset_to_monet(d2)
                result = d2.monet.remap_nearest(d1)
                return result.isel(x=0)
            elif has_xesmf:
                output = constant_1d_xesmf(latitude=latitude, longitude=longitude)
                out = resample_xesmf(self._obj, output, **kwargs)
                return _rename_latlon(out)

    def nearest_ij(self, lat=None, lon=None, **kwargs):
        """Find the nearest grid indices to given lat/lon point(s).

        Parameters
        ----------
        lat : float or array-like, optional
            Latitude value(s).
        lon : float or array-like, optional
            Longitude value(s).
        **kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        tuple
            (i, j) indices of nearest point(s).
        """
        try:
            from pyresample import utils

            from .util.interp_util import lonlat_to_swathdefinition as llsd
            from .util.interp_util import nearest_point_swathdefinition as npsd

            has_pyresample = True
        except ImportError:
            has_pyresample = False
            print("requires pyresample to be installed")

        try:
            if lat is None or lon is None:
                raise RuntimeError
        except RuntimeError:
            print("Must provide latitude and longitude")

        if has_pyresample:
            dset = _dataset_to_monet(self._obj)
            lons, lats = utils.check_and_wrap(dset.longitude.values, dset.latitude.values)
            swath = llsd(longitude=lons, latitude=lats)
            pswath = npsd(longitude=float(lon), latitude=float(lat))
            row, col = utils.generate_nearest_neighbour_linesample_arrays(swath, pswath, float(1e6))
            y, x = row[0][0], col[0][0]
            return x, y

    def nearest_latlon(self, lat=None, lon=None, cleanup=True, esmf=False, **kwargs):
        """Extract data at nearest lat/lon point(s).

        Parameters
        ----------
        lat : float or array-like, optional
            Latitude value(s).
        lon : float or array-like, optional
            Longitude value(s).
        cleanup : bool, default: True
            Whether to clean up temporary files after regridding.
        esmf : bool, default: False
            Whether to use ESMF for regridding.
        **kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        xarray.DataArray
            DataArray at nearest point(s).
        """
        try:
            from pyresample import utils

            from .util.interp_util import lonlat_to_swathdefinition as llsd
            from .util.interp_util import nearest_point_swathdefinition as npsd

            has_pyresample = True
        except ImportError:
            has_pyresample = False

        from .util.interp_util import lonlat_to_xesmf
        from .util.resample import resample_xesmf

        try:
            if lat is None or lon is None:
                raise RuntimeError
        except RuntimeError:
            print("Must provide latitude and longitude")

        d = _dataset_to_monet(self._obj)
        if has_pyresample:
            lons, lats = utils.check_and_wrap(d.longitude.values, d.latitude.values)
            swath = self._get_CoordinateDefinition(d)
            pswath = npsd(longitude=float(lon), latitude=float(lat))
            row, col = utils.generate_nearest_neighbour_linesample_arrays(swath, pswath, **kwargs)
            y, x = row[0][0], col[0][0]
            return d.isel(x=x, y=y)
        elif has_xesmf:
            kwargs = self._check_kwargs_and_set_defaults(**kwargs)
            self._obj = _rename_latlon(self._obj)
            target = lonlat_to_xesmf(longitude=lon, latitude=lat)
            output = resample_xesmf(self._obj, target, **kwargs)
            if cleanup:
                output = resample_xesmf(self._obj, target, cleanup=True, **kwargs)
            return _rename_latlon(output.squeeze())

    @staticmethod
    def _check_kwargs_and_set_defaults(**kwargs):
        """
        Parameters
        ----------
        kwargs : dict
            Regrid/remap kwargs dict.

        Returns
        -------
        dict
            With defaults added if not already set.
            Note modified in place.
        """
        if "reuse_weights" not in kwargs:
            kwargs["reuse_weights"] = False
        if "method" not in kwargs:
            kwargs["method"] = "bilinear"
        if "periodic" not in kwargs:
            kwargs["periodic"] = False
        if "filename" not in kwargs:
            kwargs["filename"] = "monet_xesmf_regrid_file.nc"
        return kwargs

    def quick_imshow(self, map_kws=None, roll_dateline=False, **kwargs):
        """Create a quick imshow plot of the data.

        Parameters
        ----------
        map_kws : dict, optional
            Keyword arguments for map creation.
        roll_dateline : bool, default: False
            Whether to roll the dateline for proper global visualization.
        **kwargs : dict
            Additional keyword arguments for imshow.

        Returns
        -------
        matplotlib.axes.Axes
            The plot axes.
        """
        import cartopy.crs as ccrs
        import matplotlib.pyplot as plt
        import seaborn as sns
        from cartopy.mpl.geoaxes import GeoAxes

        from .plots import _dynamic_fig_size
        from .plots.mapgen import draw_map

        if map_kws is None:
            map_kws = {}

        da = _dataset_to_monet(self._obj)
        da = _monet_to_latlon(da)
        crs_p = ccrs.PlateCarree()
        if "crs" not in map_kws:
            map_kws["crs"] = crs_p
        if "figsize" in kwargs:
            map_kws["figsize"] = kwargs["figsize"]
            kwargs.pop("figsize", None)
        else:
            figsize = _dynamic_fig_size(da)
            map_kws["figsize"] = figsize
        if "transform" not in kwargs:
            transform = crs_p
        else:
            transform = kwargs["transform"]
            kwargs.pop("transform", None)
        with sns.plotting_context("notebook", font_scale=1.2):
            if "ax" not in kwargs:
                ax = draw_map(**map_kws)
            else:
                ax = kwargs.pop("ax", None)
                if not isinstance(ax, GeoAxes):
                    raise TypeError("`ax` should be a Cartopy GeoAxes instance")
            _set_outline_patch_alpha(ax)
            if roll_dateline:
                _ = (
                    da.squeeze()
                    .roll(lon=int(len(da.lon) / 2), roll_coords=True)
                    .plot.imshow(ax=ax, transform=transform, **kwargs)
                )
            else:
                _ = da.squeeze().plot.imshow(ax=ax, transform=transform, **kwargs)
            plt.tight_layout()

        return ax

    def quick_map(self, map_kws=None, roll_dateline=False, **kwargs):
        """Create a quick map plot of the data.

        Parameters
        ----------
        map_kws : dict, optional
            Keyword arguments for map creation.
        roll_dateline : bool, default: False
            Whether to roll the dateline for proper global visualization.
        **kwargs : dict
            Additional keyword arguments for plotting.

        Returns
        -------
        matplotlib.axes.Axes
            The plot axes.
        """
        import cartopy.crs as ccrs
        import matplotlib.pyplot as plt
        import seaborn as sns
        from cartopy.mpl.geoaxes import GeoAxes

        from .plots import _dynamic_fig_size
        from .plots.mapgen import draw_map

        if map_kws is None:
            map_kws = {}

        da = _dataset_to_monet(self._obj)
        crs_p = ccrs.PlateCarree()
        if "crs" not in map_kws:
            map_kws["crs"] = crs_p
        if "figsize" in kwargs:
            map_kws["figsize"] = kwargs["figsize"]
            kwargs.pop("figsize", None)
        else:
            figsize = _dynamic_fig_size(da)
            map_kws["figsize"] = figsize
        transform = kwargs.pop("transform", crs_p)
        with sns.plotting_context("notebook"):
            if "ax" not in kwargs:
                ax = draw_map(**map_kws)
            else:
                ax = kwargs.pop("ax", None)
                if not isinstance(ax, GeoAxes):
                    raise TypeError("`ax` should be a Cartopy GeoAxes instance")
            _set_outline_patch_alpha(ax)
            if roll_dateline:
                _ = da.roll(x=int(len(da.x) / 2), roll_coords=True).plot(
                    x="longitude", y="latitude", ax=ax, transform=transform, **kwargs
                )
            else:
                _ = da.plot(x="longitude", y="latitude", ax=ax, transform=transform, **kwargs)
            plt.tight_layout()

        return ax

    def quick_contourf(self, map_kws=None, roll_dateline=False, **kwargs):
        """Create a quick filled contour plot of the data.

        Parameters
        ----------
        map_kws : dict, optional
            Keyword arguments for map creation.
        roll_dateline : bool, default: False
            Whether to roll the dateline for proper global visualization.
        **kwargs : dict
            Additional keyword arguments for contourf.

        Returns
        -------
        matplotlib.axes.Axes
            The plot axes.
        """
        import cartopy.crs as ccrs
        import matplotlib.pyplot as plt
        import seaborn as sns
        from cartopy.mpl.geoaxes import GeoAxes

        from monet.plots import _dynamic_fig_size
        from monet.plots.mapgen import draw_map

        if map_kws is None:
            map_kws = {}

        da = _dataset_to_monet(self._obj)
        dlon = da.longitude.diff("x")
        if not ((dlon >= 0).all() or (dlon <= 0).all()):  # monotonic
            da["longitude"] = da.longitude % 360  # unwrap longitudes
        crs_p = ccrs.PlateCarree()
        if "crs" not in map_kws:
            map_kws["crs"] = crs_p
        if "figsize" in kwargs:
            map_kws["figsize"] = kwargs["figsize"]
            kwargs.pop("figsize", None)
        else:
            figsize = _dynamic_fig_size(da)
            map_kws["figsize"] = figsize
        if "transform" not in kwargs:
            transform = crs_p
        else:
            transform = kwargs["transform"]
            kwargs.pop("transform", None)
        with sns.plotting_context("notebook"):
            if "ax" not in kwargs:
                ax = draw_map(**map_kws)
            else:
                ax = kwargs.pop("ax", None)
                if not isinstance(ax, GeoAxes):
                    raise TypeError("`ax` should be a Cartopy GeoAxes instance")
            _set_outline_patch_alpha(ax)
            if roll_dateline:
                _ = da.roll(x=int(len(da.x) / 2), roll_coords=True).plot.contourf(
                    x="longitude", y="latitude", ax=ax, transform=transform, **kwargs
                )
            else:
                _ = da.plot.contourf(
                    x="longitude", y="latitude", ax=ax, transform=transform, **kwargs
                )
            plt.tight_layout()

        return ax

    def _tight_layout(self):
        """Apply tight layout to the current figure.

        Returns
        -------
        None
        """
        from matplotlib.pyplot import subplots_adjust

        subplots_adjust(0, 0, 1, 1)

    def _check_swath_def(self, defn):
        """Check if a SwathDefinition is valid.

        Parameters
        ----------
        defn : object
            Object to check if it's a valid SwathDefinition.

        Returns
        -------
        pyresample.geometry.SwathDefinition
            Validated SwathDefinition.
        """
        from pyresample.geometry import SwathDefinition

        if isinstance(defn, SwathDefinition):
            return True
        else:
            return False

    def _get_CoordinateDefinition(self, data=None):
        """Get a CoordinateDefinition from DataArray.

        Parameters
        ----------
        data : xarray.DataArray, optional
            DataArray to get coordinates from.

        Returns
        -------
        pyresample.geometry.CoordinateDefinition
            CoordinateDefinition for the data.
        """
        from pyresample import geometry as geo

        if data is not None:
            g = geo.CoordinateDefinition(lats=data.latitude, lons=data.longitude)
        else:
            g = geo.CoordinateDefinition(lats=self._obj.latitude, lons=self._obj.longitude)
        return g

    def remap_nearest(self, data, radius_of_influence=1e6, **kwargs):
        """Remap data using nearest neighbor interpolation.

        Parameters
        ----------
        data : xarray.DataArray or xarray.Dataset
            Data to remap.
        radius_of_influence : float, default: 1e6
            Search radius in meters.
        **kwargs : dict
            Additional keyword arguments for regridding.

        Returns
        -------
        xarray.DataArray or xarray.Dataset
            Remapped data.
        """
        from pyresample import kd_tree

        source_data = _dataset_to_monet(data)
        target_data = _dataset_to_monet(self._obj)
        source = self._get_CoordinateDefinition(source_data)
        target = self._get_CoordinateDefinition(target_data)
        r = kd_tree.XArrayResamplerNN(
            source, target, radius_of_influence=radius_of_influence, **kwargs
        )
        r.get_neighbour_info()
        if isinstance(source_data, xr.DataArray):
            result = r.get_sample_from_neighbour_info(source_data)
            result.name = source_data.name
            result["latitude"] = target_data.latitude
            result["longitude"] = target_data.longitude

        elif isinstance(source_data, xr.Dataset):
            results = {}
            for i in source_data.data_vars.keys():
                results[i] = r.get_sample_from_neighbour_info(source_data[i])
            result = xr.Dataset(results)
            if bool(source_data.attrs):
                result.attrs = source_data.attrs
            result.coords["latitude"] = target_data.latitude
            result.coords["longitude"] = target_data.longitude

        return result

    def remap_xesmf(self, data, **kwargs):
        """Remap data using xESMF regridding.

        Parameters
        ----------
        data : xarray.DataArray or xarray.Dataset
            Data to remap.
        **kwargs : dict
            Keyword arguments for xESMF regridding.

        Returns
        -------
        xarray.DataArray or xarray.Dataset
            Remapped data.
        """
        kwargs["method"] = kwargs.get("method", "bilinear")
        if has_xesmf:
            from .util import resample

            target = _rename_latlon(self._obj)
            source = _rename_latlon(data)
            out = resample.resample_xesmf(source, target, **kwargs)
            return _rename_to_monet_latlon(out)

        else:
            print("xesmf unavailable. Try `import xesmf` and check the failure message.")

    def combine_point(self, data, suffix=None, pyresample=True, **kwargs):
        """Combine point data with this DataArray.

        Parameters
        ----------
        data : pandas.DataFrame
            Point data to combine.
        suffix : str, optional
            Suffix to add to variable names. Default is '_new'.
        pyresample : bool, default: True
            Whether to use pyresample for remapping.
        **kwargs : dict
            Additional keyword arguments for regridding.

        Returns
        -------
        pandas.DataFrame
            Combined data.
        """
        if has_pyresample:
            from .util.combinetool import combine_da_to_df
        if has_xesmf:
            from .util.combinetool import combine_da_to_df_xesmf
        da = _dataset_to_monet(self._obj)
        if isinstance(data, pd.DataFrame):
            if has_pyresample and pyresample:
                return combine_da_to_df(da, data, **kwargs)
            else:  # xesmf resample
                return combine_da_to_df_xesmf(da, data, suffix=suffix, **kwargs)
        else:
            print("`data` must be a pandas.DataFrame")


@xr.register_dataset_accessor("monet")
class MONETAccessorDataset:
    """Dataset accessor for MONET functionality."""

    def __init__(self, xray_obj):
        """Initialize the accessor.

        Parameters
        ----------
        xray_obj : xarray.Dataset
            The Dataset this accessor will work with.
        """
        self._obj = xray_obj

    def is_land(self, return_xarray=False):
        """Check if points are on land.

        Parameters
        ----------
        return_xarray : bool, default: False
            If True, return results as xarray. Otherwise, return numpy array.

        Returns
        -------
        xarray.DataArray or numpy.ndarray
            Boolean array where True indicates land.
        """
        import global_land_mask as glm

        da = _dataset_to_monet(self._obj)
        island = glm.is_land(da.latitude.values, da.longitude.values)
        if return_xarray:
            return da.where(island)
        else:
            return island

    def is_ocean(self, return_xarray=False):
        """Check if points are on ocean.

        Parameters
        ----------
        return_xarray : bool, default: False
            If True, return results as xarray. Otherwise, return numpy array.

        Returns
        -------
        xarray.DataArray or numpy.ndarray
            Boolean array where True indicates ocean.
        """
        import global_land_mask as glm

        da = _dataset_to_monet(self._obj)
        isocean = glm.is_ocean(da.latitude.values, da.longitude.values)
        if return_xarray:
            return da.where(isocean)
        else:
            return isocean

    def cftime_to_datetime64(self, name=None):
        """Convert cftime coordinates to numpy datetime64.

        Parameters
        ----------
        name : str, optional
            Name of the coordinate to convert. If None, tries to detect the time coordinate.

        Returns
        -------
        xarray.Dataset
            Dataset with converted time coordinate.
        """
        from numpy import vectorize

        ds = self._obj

        def cf_to_dt64(x):
            return pd.to_datetime(x.strftime("%Y-%m-%d %H:%M:%S"))

        if name is None:  # assume 'time' is the column name to transform
            name = "time"
        if isinstance(ds[name].to_index(), xr.CFTimeIndex):
            ds[name] = xr.apply_ufunc(vectorize(cf_to_dt64), ds[name])
        return ds

    def remap_xesmf(self, data, **kwargs):
        """Remap data using xESMF regridding.

        Parameters
        ----------
        data : xarray.DataArray or xarray.Dataset
            Data to remap.
        **kwargs : dict
            Keyword arguments for xESMF regridding.

        Returns
        -------
        xarray.Dataset
            Remapped dataset.
        """
        if has_xesmf:
            try:
                if isinstance(data, xr.DataArray):
                    data = _rename_latlon(data)
                    self._remap_xesmf_dataarray(data, **kwargs)
                elif isinstance(data, xr.Dataset):
                    data = _rename_latlon(data)
                    self._remap_xesmf_dataset(data, **kwargs)
                else:
                    raise TypeError
            except TypeError:
                print("data must be an xarray.DataArray or xarray.Dataset")

        else:
            print("xesmf unavailable. Try `import xesmf` and check the failure message.")

    def _remap_xesmf_dataset(self, dset, filename="monet_xesmf_regrid_file.nc", **kwargs):
        """Remap dataset using xESMF.

        Parameters
        ----------
        dset : xarray.Dataset
            Dataset to remap.
        filename : str, default: "monet_xesmf_regrid_file.nc"
            Name of temporary file for regridding weights.
        **kwargs : dict
            Keyword arguments for xESMF regridding.

        Returns
        -------
        xarray.Dataset
            Remapped dataset.
        """
        skip_keys = ["lat", "lon", "time", "TFLAG"]
        vars = pd.Series(list(dset.variables))
        loop_vars = vars.loc[~vars.isin(skip_keys)]
        dataarray = dset[loop_vars[0]]
        da = self._remap_xesmf_dataarray(dataarray, filename=filename, **kwargs)
        self._obj[da.name] = da
        das = {}
        das[da.name] = da
        for i in loop_vars[1:]:
            dataarray = dset[i]
            tmp = self._remap_xesmf_dataarray(
                dataarray, filename=filename, reuse_weights=True, **kwargs
            )
            das[tmp.name] = tmp.copy()
        return xr.Dataset(das)

    def _remap_xesmf_dataarray(
        self, dataarray, method="bilinear", filename="monet_xesmf_regrid_file.nc", **kwargs
    ):
        """Remap DataArray using xESMF.

        Parameters
        ----------
        dataarray : xarray.DataArray
            DataArray to remap.
        method : str, default: "bilinear"
            Regridding method.
        filename : str, default: "monet_xesmf_regrid_file.nc"
            Name of temporary file for regridding weights.
        **kwargs : dict
            Keyword arguments for xESMF regridding.

        Returns
        -------
        xarray.DataArray
            Remapped DataArray.
        """
        from .util import resample

        target = self._obj
        out = resample.resample_xesmf(dataarray, target, method=method, filename=filename, **kwargs)
        if out.name in self._obj.variables:
            out.name = out.name + "_y"
        self._obj[out.name] = out
        return out

    def _get_CoordinateDefinition(self, data=None):
        """Get a CoordinateDefinition from Dataset.

        Parameters
        ----------
        data : xarray.Dataset, optional
            Dataset to get coordinates from.

        Returns
        -------
        pyresample.geometry.CoordinateDefinition
            CoordinateDefinition for the data.
        """
        from pyresample import geometry as geo

        if data is not None:
            g = geo.CoordinateDefinition(lats=data.latitude, lons=data.longitude)
        else:
            g = geo.CoordinateDefinition(lats=self._obj.latitude, lons=self._obj.longitude)
        return g

    def remap_nearest(self, data, radius_of_influence=1e6, **kwargs):
        """Remap data using nearest neighbor interpolation.

        Parameters
        ----------
        data : xarray.DataArray or xarray.Dataset
            Data to remap.
        radius_of_influence : float, default: 1e6
            Search radius in meters.
        **kwargs : dict
            Additional keyword arguments for regridding.

        Returns
        -------
        xarray.Dataset
            Remapped dataset.
        """
        from pyresample import kd_tree

        source_data = _dataset_to_monet(data)
        target_data = _dataset_to_monet(self._obj)
        source = self._get_CoordinateDefinition(source_data)
        target = self._get_CoordinateDefinition(target_data)
        r = kd_tree.XArrayResamplerNN(
            source, target, radius_of_influence=radius_of_influence, **kwargs
        )
        r.get_neighbour_info()
        if isinstance(source_data, xr.DataArray):
            result = r.get_sample_from_neighbour_info(source_data)
            result.name = source_data.name
            result["latitude"] = target_data.latitude
            result["longitude"] = target_data.longitude

        elif isinstance(source_data, xr.Dataset):
            results = {}
            for i in source_data.data_vars.keys():
                results[i] = r.get_sample_from_neighbour_info(source_data[i])
            result = xr.Dataset(results)
            if bool(source_data.attrs):
                result.attrs = source_data.attrs
            result.coords["latitude"] = target_data.latitude
            result.coords["longitude"] = target_data.longitude

        return result

    def remap_nearest_unstructured(self, data):
        """Remap unstructured grid data using nearest neighbor interpolation.

        Parameters
        ----------
        data : xarray.DataArray or xarray.Dataset
            Unstructured grid data to remap.

        Returns
        -------
        xarray.Dataset
            Remapped dataset.
        """
        try:
            check_error = False
            if isinstance(data, xr.DataArray) or isinstance(data, xr.Dataset):
                check_error = False
            else:
                check_error = True
            if check_error:
                raise TypeError
        except TypeError:
            print("data must be either an xarray.DataArray or xarray.Dataset")

        model_data = data
        obs_data = self._obj

        site_indices = []
        site_latitudes = obs_data["latitude"].values[0, :]
        site_longitudes = obs_data["longitude"].values[0, :]
        model_latitudes = model_data["latitude"].values
        model_longitudes = model_data["longitude"].values

        for siteii in np.arange(len(obs_data["siteid"][0])):
            site_indices.append(
                np.argmin(
                    np.abs(site_latitudes[siteii] - model_latitudes)
                    + np.abs(site_longitudes[siteii] - model_longitudes)
                )
            )

        dict_data = {}
        for dvar in model_data.data_vars:
            if dvar in ["latitude", "longitude"]:
                continue
            else:
                dict_data[dvar] = (
                    ["time", "z", "y", "x"],
                    model_data[dvar][:, 0, np.array(site_indices)].values.reshape(
                        len(model_data["time"]), 1, 1, len(site_indices)
                    ),
                )

        dict_coords = {
            "time": (["time"], model_data["time"].values),
            "x": (["x"], np.arange(len(site_indices))),
            "longitude": (
                ["y", "x"],
                model_longitudes[np.array(site_indices)].reshape(1, len(site_indices)),
            ),
            "latitude": (
                ["y", "x"],
                model_latitudes[np.array(site_indices)].reshape(1, len(site_indices)),
            ),
        }

        result = xr.Dataset(data_vars=dict_data, coords=dict_coords)

        return result

    def nearest_ij(self, lat=None, lon=None, **kwargs):
        """Find the nearest grid indices to given lat/lon point(s).

        Parameters
        ----------
        lat : float or array-like, optional
            Latitude value(s).
        lon : float or array-like, optional
            Longitude value(s).
        **kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        tuple
            (i, j) indices of nearest point(s).
        """
        try:
            from pyresample import utils

            has_pyresample = True
        except ImportError:
            has_pyresample = False
            print("requires pyresample to be installed")
        from .util.interp_util import lonlat_to_swathdefinition as llsd
        from .util.interp_util import nearest_point_swathdefinition as npsd

        try:
            if lat is None or lon is None:
                raise RuntimeError
        except RuntimeError:
            print("Must provide latitude and longitude")

        if has_pyresample:
            dset = _dataset_to_monet(self._obj)
            lons, lats = utils.check_and_wrap(dset.longitude.values, dset.latitude.values)
            swath = llsd(longitude=lons, latitude=lats)
            pswath = npsd(longitude=float(lon), latitude=float(lat))
            row, col = utils.generate_nearest_neighbour_linesample_arrays(swath, pswath, float(1e6))
            y, x = row[0][0], col[0][0]
            return x, y

    def nearest_latlon(self, lat=None, lon=None, cleanup=True, esmf=False, **kwargs):
        """Extract data at nearest lat/lon point(s).

        Parameters
        ----------
        lat : float or array-like, optional
            Latitude value(s).
        lon : float or array-like, optional
            Longitude value(s).
        cleanup : bool, default: True
            Whether to clean up temporary files after regridding.
        esmf : bool, default: False
            Whether to use ESMF for regridding.
        **kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        xarray.Dataset
            Dataset at nearest point(s).
        """
        try:
            from pyresample import utils

            from .util.interp_util import lonlat_to_swathdefinition as llsd
            from .util.interp_util import nearest_point_swathdefinition as npsd

            has_pyresample = True
        except ImportError:
            has_pyresample = False

        from .util.interp_util import lonlat_to_xesmf
        from .util.resample import resample_xesmf

        try:
            if lat is None or lon is None:
                raise RuntimeError
        except RuntimeError:
            print("Must provide latitude and longitude")

        if has_pyresample:
            dset = _dataset_to_monet(self._obj)
            lons, lats = utils.check_and_wrap(dset.longitude.values, dset.latitude.values)
            swath = llsd(longitude=lons, latitude=lats)
            pswath = npsd(longitude=float(lon), latitude=float(lat))
            row, col = utils.generate_nearest_neighbour_linesample_arrays(swath, pswath, float(1e6))
            y, x = row[0][0], col[0][0]
            return dset.isel(x=x).isel(y=y)
        elif has_xesmf:
            kwargs = self._check_kwargs_and_set_defaults(**kwargs)
            self._obj = _rename_latlon(self._obj)
            target = lonlat_to_xesmf(longitude=lon, latitude=lat)
            output = resample_xesmf(self._obj, target, **kwargs)
            if cleanup:
                output = resample_xesmf(self._obj, target, cleanup=True, **kwargs)
            return _rename_latlon(output.squeeze())

    def interp_constant_lat(self, lat=None, lat_name="latitude", lon_name="longitude", **kwargs):
        """Interpolate data to a constant latitude.

        Parameters
        ----------
        lat : float, optional
            Latitude value to interpolate to.
        lat_name : str, default: "latitude"
            Name of the latitude coordinate.
        lon_name : str, default: "longitude"
            Name of the longitude coordinate.
        **kwargs : dict
            Additional keyword arguments for interpolation.

        Returns
        -------
        xarray.Dataset
            Interpolated dataset.
        """
        from numpy import asarray, linspace, ones

        if has_xesmf:
            from .util.interp_util import constant_1d_xesmf
            from .util.resample import resample_xesmf

        try:
            if lat is None:
                raise RuntimeError
        except RuntimeError:
            print("Must enter lat value")
        d1 = _dataset_to_monet(self._obj, lat_name=lat_name, lon_name=lon_name)
        longitude = linspace(d1.longitude.min(), d1.longitude.max(), len(d1.x))
        latitude = ones(longitude.shape) * asarray(lat)
        if has_pyresample:
            d2 = xr.DataArray(
                ones((len(longitude), len(longitude))),
                dims=["lon", "lat"],
                coords=[longitude, latitude],
            )
            d2 = _dataset_to_monet(d2)
            result = d2.monet.remap_nearest(d1)
            return result.isel(y=0)
        elif has_xesmf:
            output = constant_1d_xesmf(latitude=latitude, longitude=longitude)
            out = resample_xesmf(self._obj, output, **kwargs)
            return _rename_latlon(out)

    def interp_constant_lon(self, lon=None, **kwargs):
        """Interpolate data to a constant longitude.

        Parameters
        ----------
        lon : float, optional
            Longitude value to interpolate to.
        **kwargs : dict
            Additional keyword arguments for interpolation.

        Returns
        -------
        xarray.Dataset
            Interpolated dataset.
        """
        if has_xesmf:
            from .util.interp_util import constant_1d_xesmf
            from .util.resample import resample_xesmf
        from numpy import asarray, linspace, ones

        try:
            if lon is None:
                raise RuntimeError
        except RuntimeError:
            print("Must enter lon value")
        d1 = _dataset_to_monet(self._obj)
        latitude = linspace(d1.latitude.min(), d1.latitude.max(), len(d1.y))
        longitude = ones(latitude.shape) * asarray(lon)
        if has_pyresample:
            if has_pyresample:
                d2 = xr.DataArray(
                    ones((len(longitude), len(longitude))),
                    dims=["lon", "lat"],
                    coords=[longitude, latitude],
                )
                d2 = _dataset_to_monet(d2)
                result = d2.monet.remap_nearest(d1)
                return result.isel(x=0)
            elif has_xesmf:
                output = constant_1d_xesmf(latitude=latitude, longitude=longitude)
                out = resample_xesmf(self._obj, output, **kwargs)
                return _rename_latlon(out)

    def stratify(self, levels, vertical, axis=1):
        """Vertically interpolate data to specified levels.

        Parameters
        ----------
        levels : array-like
            Target vertical levels.
        vertical : xarray.DataArray
            Vertical coordinate values.
        axis : int, default: 1
            Axis along which to interpolate.

        Returns
        -------
        xarray.Dataset
            Vertically interpolated dataset.
        """
        if isinstance(vertical, str):
            vertical = self._obj[vertical]
        vertical_shape = vertical.shape
        loop_vars = [
            vn
            for vn in self._obj.variables
            if "z" in self._obj[vn].dims
            and vn != vertical.name
            and len(self._obj[vn].shape) >= len(vertical_shape)
            and self._obj[vn].shape[-len(vertical_shape) :] == vertical_shape
        ]
        orig = self._obj[loop_vars[0]].monet.stratify(levels, vertical, axis=axis)
        dset = orig.to_dataset(name=loop_vars[0])
        dset.attrs = self._obj.attrs.copy()
        for vn in loop_vars[1:]:
            dset[vn] = self._obj[vn].monet.stratify(levels, vertical, axis=axis)
        return dset

    def window(self, lat_min, lon_min, lat_max, lon_max):
        """Extract a spatial window from the data.

        Parameters
        ----------
        lat_min : float
            Minimum latitude.
        lon_min : float
            Minimum longitude.
        lat_max : float
            Maximum latitude.
        lon_max : float
            Maximum longitude.

        Returns
        -------
        xarray.Dataset
            Windowed dataset.
        """
        try:
            from numpy import concatenate
            from pyresample import utils

            from .util.interp_util import lonlat_to_swathdefinition as llsd
            from .util.interp_util import nearest_point_swathdefinition as npsd

            has_pyresample = True
        except ImportError:
            has_pyresample = False
        try:
            if has_pyresample:
                lons, lats = utils.check_and_wrap(
                    self._obj.longitude.values, self._obj.latitude.values
                )
                swath = llsd(longitude=lons, latitude=lats)
                pswath_ll = npsd(longitude=float(lon_min), latitude=float(lat_min))
                pswath_ur = npsd(longitude=float(lon_max), latitude=float(lat_max))
                row, col = utils.generate_nearest_neighbour_linesample_arrays(
                    swath, pswath_ll, float(1e6)
                )
                y_ll, x_ll = row[0][0], col[0][0]
                row, col = utils.generate_nearest_neighbour_linesample_arrays(
                    swath, pswath_ur, float(1e6)
                )
                y_ur, x_ur = row[0][0], col[0][0]
                if x_ur < x_ll:
                    x1 = self._obj.x.where(self._obj.x >= x_ll, drop=True).values
                    x2 = self._obj.x.where(self._obj.x <= x_ur, drop=True).values
                    xrange = concatenate([x1, x2]).astype(int)
                    self._obj["longitude"][:] = utils.wrap_longitudes(self._obj.longitude.values)
                else:
                    xrange = slice(x_ll, x_ur)
                if y_ur < y_ll:
                    y1 = self._obj.y.where(self._obj.y >= y_ll, drop=True).values
                    y2 = self._obj.y.where(self._obj.y <= y_ur, drop=True).values
                    yrange = concatenate([y1, y2]).astype(int)
                else:
                    yrange = slice(y_ll, y_ur)
                return self._obj.sel(x=xrange, y=yrange)
            else:
                raise ImportError
        except ImportError:
            print("Window functionality is unavailable without pyresample")

    def combine_point(self, data, suffix=None, pyresample=True, **kwargs):
        """Combine point data with this Dataset.

        Parameters
        ----------
        data : pandas.DataFrame
            Point data to combine.
        suffix : str, optional
            Suffix to add to variable names. Default is '_new'.
        pyresample : bool, default: True
            Whether to use pyresample for remapping.
        **kwargs : dict
            Additional keyword arguments for regridding.

        Returns
        -------
        pandas.DataFrame
            Combined data.
        """
        if has_pyresample:
            from .util.combinetool import combine_da_to_df
        if has_xesmf:
            from .util.combinetool import combine_da_to_df_xesmf
        da = _dataset_to_monet(self._obj)
        if isinstance(data, pd.DataFrame):
            if has_pyresample and pyresample:
                return combine_da_to_df(da, data, **kwargs)
            else:  # xesmf resample
                return combine_da_to_df_xesmf(da, data, suffix=suffix, **kwargs)
        else:
            print("`data` must be a pandas.DataFrame")

    def wrap_longitudes(self, lon_name="longitude"):
        """Wrap longitude values to [-180, 180).

        Parameters
        ----------
        lon_name : str, default: "longitude"
            Name of the longitude coordinate.

        Returns
        -------
        xarray.Dataset
            Dataset with wrapped longitudes.
        """
        dset = self._obj
        dset[lon_name] = (dset[lon_name] + 180) % 360 - 180
        return dset

    def tidy(self, lon_name="longitude"):
        """Apply tidying operations to the data.

        Parameters
        ----------
        lon_name : str, default: "longitude"
            Name of the longitude coordinate.

        Returns
        -------
        xarray.Dataset
            Tidied dataset.
        """
        d = self._obj
        wd = d.monet.wrap_longitudes(lon_name=lon_name)
        wdl = wd.sortby(wd[lon_name])
        return wdl
