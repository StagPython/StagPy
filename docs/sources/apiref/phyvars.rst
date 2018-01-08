phyvars
=======

.. automodule:: stagpy.phyvars

   .. class:: Varf

      :class:`collections.namedtuple` whose instances hold metadata of
      scalar fields. It defines the following fields:

      - **description** (*str* or *func*): short description of the variable if
        it is output by StagYY, function to compute it otherwise.
      - **shortname** (*str*): used to label axis on plot.
      - **popts** (*dict*): keyword arguments fed to
        :meth:`matplotlib.axes.Axes.pcolormesh` in
        :func:`stagpy.field.plot_scalar`.

   .. data:: FIELD
      :annotation: = {fieldvar: Varf()}

      Dictionary of scalar fields output by StagYY. Keys are the variable
      names, values are :class:`Varf` instances.

   .. data:: FIELD_EXTRA
      :annotation: = {fieldvar: Varf()}

      Dictionary of scalar fields that StagPy can compute. Keys are the
      variable names, values are :class:`Varf` instances.

   .. class:: Varr

      :class:`collections.namedtuple` whose instances hold metadata of
      radial profiles. It defines the following fields:

      - **description** (*str* or *func*): short description of the variable if
        it is output by StagYY, function to compute it otherwise.
      - **shortname** (*str*): used to label axis on plot.

   .. data:: RPROF
      :annotation: = {rprofvar: Varr()}

      Dictionary of radial profiles output by StagYY. Keys are the variable
      names, values are :class:`Varr` instances.

   .. data:: RPROF_EXTRA
      :annotation: = {rprofvar: Varr()}

      Dictionary of radial profiles that StagPy can compute. Keys are the
      variable names, values are :class:`Vart` instances.

   .. class:: Vart

      :class:`collections.namedtuple` whose instances hold metadata of
      time series. It defines the following fields:

      - **description** (*str* or *func*): short description of the variable if
        it is output by StagYY, function to compute it otherwise.
      - **shortname** (*str*): used to label axis on plot.

   .. data:: TIME
      :annotation: = {timevar: Vart()}

      Dictionary of time series output by StagYY. Keys are the variable names,
      values are :class:`Vart` instances.

   .. data:: TIME_EXTRA
      :annotation: = {timevar: Vart()}

      Dictionary of time series that StagPy can compute. Keys are the variable
      names, values are :class:`Vart` instances.

   .. class:: Varp

      :class:`collections.namedtuple` whose instances hold metadata of
      plate variables. It defines the following fields:

      - **description** (*str*): short description of the variable.

   .. data:: PLATES
      :annotation: = {platevar: Varp()}

      Dictionary of plate variables output by StagYY. Keys are the variable
      names, values are :class:`Varp` instances.
