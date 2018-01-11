parfile
=======

.. automodule:: stagpy.parfile
   :members:

   .. data:: PAR_DFLT_FILE
      :annotation: = config.CONFIG_DIR / 'par'

      Path of default par file used to fill missing entries by :func:`readpar`.

   .. data:: PAR_DEFAULT

      Defaut value of all the StagYY input parameters used to fill missing
      entries by :func:`readpar`. It is a :class:`f90nml.namelist.Namelist`.
