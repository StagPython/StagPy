config
======

.. automodule:: stagpy.config
   :members:

   .. data:: HOME_DIR
      :annotation: = pathlib.Path.home()

      Home directory.

   .. data:: CONFIG_DIR
      :annotation: = HOME_DIR / '.config' / 'stagpy'

      StagPy configuration directory.

   .. data:: CONFIG_FILE
      :annotation: = CONFIG_DIR / 'config.toml'

      Path of global configuration file.

   .. data:: CONFIG_LOCAL
      :annotation: = pathlib.Path('.stagpy.toml')

      Path of local configuration file.
