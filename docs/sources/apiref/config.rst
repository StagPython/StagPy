config
======

.. automodule:: stagpy.config
   :members:
   :exclude-members: Conf

   .. data:: CONFIG_DIR
      :annotation: = pathlib.Path(os.path.expanduser(~)) / '.config' / 'stagpy'

      StagPy configuration directory.

   .. data:: CONFIG_FILE
      :annotation: = config.CONFIG_DIR / 'config'

      Path of default configuration file.

   .. class:: Conf

      :class:`collections.namedtuple` whose instances hold metadata of
      configuration options. It defines the following fields:

      - **default**: the default value of the configuration option.
      - **cmd_arg** (*bool*): whether the option is a command line argument.
      - **shortname** (*str*): short version of the command line argument.
      - **kwargs** (*dict*): keyword arguments fed to
        :meth:`argparse.ArgumentParser.add_argument` during the construction
        of the command line arguments parser.
      - **conf_arg** (*bool*): whether the option can be set in the config file.
      - **help_string** (*str*): short description of the option.

   .. data:: CONF_DEF
      :annotation: = {'section': {'option': Conf()}}

      Hold metadata and structure of StagPy configuration options. Options are
      grouped by *sections* whose names are the first level of keys of
      :data:`CONF_DEF`. The *option names* themselves constitute the second level
      of keys. Finally, values are :class:`Conf` instances holding the metadata
      of each option.
