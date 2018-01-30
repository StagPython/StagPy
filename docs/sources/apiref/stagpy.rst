stagpy
======

.. automodule:: stagpy
   :members:

   .. data:: __version__
      :annotation: = 'x.y.z[.devN+gHASH[.dYYYYMMDD]]'

      Full version of StagPy. ``'x.y.z'`` is the stable version number. For
      developers, the ``'.devN+gHASH'`` part indicates the number ``N`` of
      commits since the last stable version and the last commit hash. If you
      made modifications to the code that are not committed yet, the date of
      the last modification appears in the ``'.dYYYYMMDD'`` segment.

   .. data:: conf

      Global :class:`loam.manager.ConfigurationManager` instance, holding
      configuration options.
