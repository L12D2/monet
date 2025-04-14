Get in Touch
------------

Ask questions, suggest features or view source code `on GitHub`_.

If an issue arises please post on the
`GitHub issues <https://github.com/noaa-oar-arl/monet/issues>`__.


API
---

.. module:: monet

.. py:currentmodule:: None

Top-level functions
~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: api/
   :recursive:

   monet.dataset_to_monet
   monet.rename_to_monet_latlon
   monet.rename_latlon


Modules
~~~~~~~

.. autosummary::
   :toctree: api/
   :recursive:

   monet.met_funcs
   monet.plots
   monet.util

Plotting Functions
~~~~~~~~~~~~~~~~~~

.. currentmodule:: monet.plots

.. autosummary::
   :toctree: api/
   :template: autosummary/accessor_function.rst

   cmap_discretize
   colorbar_index
   kdeplot
   make_spatial_contours
   make_spatial_plot
   normval
   savefig
   scatter
   sp_scatter_bias
   spatial
   spatial_bias_scatter
   taylordiagram
   timeseries
   wind_barbs
   wind_quiver

.. toctree::
   :hidden:

   api/monet.plots.taylordiagram

.. toctree::
   :hidden:

   api/generated/monet.plots.cmap_discretize
   api/generated/monet.plots.colorbar_index
   api/generated/monet.plots.kdeplot
   api/generated/monet.plots.make_spatial_contours
   api/generated/monet.plots.make_spatial_plot
   api/generated/monet.plots.normval
   api/generated/monet.plots.savefig
   api/generated/monet.plots.scatter
   api/generated/monet.plots.sp_scatter_bias
   api/generated/monet.plots.spatial
   api/generated/monet.plots.spatial_bias_scatter
   api/generated/monet.plots.taylordiagram
   api/generated/monet.plots.timeseries
   api/generated/monet.plots.wind_barbs
   api/generated/monet.plots.wind_quiver

.. _xarray-accessors:

DataArray Accessor
~~~~~~~~~~~~~~~~~~

.. currentmodule:: xarray

.. autosummary::
   :toctree: api/
   :template: autosummary/accessor_method.rst

   DataArray.monet.wrap_longitudes
   DataArray.monet.tidy
   DataArray.monet.is_land
   DataArray.monet.is_ocean
   DataArray.monet.cftime_to_datetime64
   DataArray.monet.structure_for_monet
   DataArray.monet.stratify
   DataArray.monet.window
   DataArray.monet.interp_constant_lat
   DataArray.monet.interp_constant_lon
   DataArray.monet.nearest_ij
   DataArray.monet.nearest_latlon
   DataArray.monet.quick_imshow
   DataArray.monet.quick_map
   DataArray.monet.quick_contourf
   DataArray.monet.remap_nearest
   DataArray.monet.remap_xesmf
   DataArray.monet.combine_point


Dataset Accessor
~~~~~~~~~~~~~~~~

.. currentmodule:: xarray

.. autosummary::
   :toctree: api/
   :template: autosummary/accessor_method.rst

   Dataset.monet.wrap_longitudes
   Dataset.monet.tidy
   Dataset.monet.is_land
   Dataset.monet.is_ocean
   Dataset.monet.cftime_to_datetime64
   Dataset.monet.stratify
   Dataset.monet.window
   Dataset.monet.interp_constant_lat
   Dataset.monet.interp_constant_lon
   Dataset.monet.nearest_ij
   Dataset.monet.nearest_latlon
   Dataset.monet.remap_nearest
   Dataset.monet.remap_nearest_unstructured
   Dataset.monet.remap_xesmf
   Dataset.monet.combine_point


.. _pandas-accessors:

DataFrame Accessor
~~~~~~~~~~~~~~~~~~

.. currentmodule:: pandas

.. autosummary::
   :toctree: api/
   :template: autosummary/accessor_method.rst

   DataFrame.monet.to_ascii2nc_df
   DataFrame.monet.to_ascii2nc_list
   DataFrame.monet.rename_for_monet
   DataFrame.monet.get_sparse_SwathDefinition
   DataFrame.monet.remap_nearest
   DataFrame.monet.cftime_to_datetime64

.. autosummary::
   :toctree: api/
   :template: autosummary/accessor_attribute.rst

   DataFrame.monet.center


.. _on GitHub: https://github.com/noaa-oar-arl/monet
