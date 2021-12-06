phyvars
=======

.. automodule:: stagpy.phyvars

   .. data:: FIELD
      :annotation: = {fieldvar: Varf()}

      Dictionary of scalar fields output by StagYY. Keys are the variable
      names, values are :class:`Varf` instances.

   .. data:: FIELD_EXTRA
      :annotation: = {fieldvar: Varf()}

      Dictionary of scalar fields that StagPy can compute. Keys are the
      variable names, values are :class:`Varf` instances.

   .. data:: SFIELD
      :annotation: = {fieldvar: Varf()}

      Dictionary of surface scalar fields output by StagYY. Keys are the
      variable names, values are :class:`Varf` instances.

   .. data:: RPROF
      :annotation: = {rprofvar: Varr()}

      Dictionary of radial profiles output by StagYY. Keys are the variable
      names, values are :class:`Varr` instances.

   .. data:: RPROF_EXTRA
      :annotation: = {rprofvar: Varr()}

      Dictionary of radial profiles that StagPy can compute. Keys are the
      variable names, values are :class:`Vart` instances.

   .. data:: REFSTATE
      :annotation: = {refstatevar: Varr()}

      Dictionary of radial profiles of the reference state. Keys are the
      variable names, values are :class:`Varr` instances.

   .. data:: TIME
      :annotation: = {timevar: Vart()}

      Dictionary of time series output by StagYY. Keys are the variable names,
      values are :class:`Vart` instances.

   .. data:: TIME_EXTRA
      :annotation: = {timevar: Vart()}

      Dictionary of time series that StagPy can compute. Keys are the variable
      names, values are :class:`Vart` instances.

   .. data:: SCALES
      :annotation: = {dimstr: func}

      Dictionary mapping dimension strings (**dim** fields in ``Var*``) to
      functions which are themselves mapping from
      :class:`~stagpy.stagyydata.StagyyData` to the scale for that dimension.
