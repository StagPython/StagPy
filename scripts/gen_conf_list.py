"""Generate list of configuration options."""

import importlib
from dataclasses import fields
from pathlib import Path
from typing import TextIO

import mkdocs_gen_files

import stagpy.config


def print_config_list(fd: TextIO) -> None:
    importlib.reload(stagpy.config)
    from stagpy.config import Config

    print("Run the following to create a local config file `.stagpy.toml`", file=fd)
    print('```sh title="shell"', file=fd)
    print("stagpy config --create", file=fd)
    print("```", file=fd)
    print(file=fd)
    print("Available options are described hereafter.", file=fd)

    stagpy_conf = Config.default_()
    for sec_fld in fields(stagpy_conf):
        sec_name = sec_fld.name
        print(file=fd)
        print(f"## {sec_name}", file=fd)
        print(file=fd)
        print("Name | Description | CLI, config file?", file=fd)
        print("---|---|---", file=fd)
        section = getattr(stagpy_conf, sec_name)
        for fld in fields(section):
            opt = fld.name
            entry = section.meta_(opt).entry
            if entry.in_cli and entry.in_file:
                c_f = "both"
            elif entry.in_cli:
                c_f = "CLI"
            else:
                c_f = "config file"
            sec_class = section.__class__.__name__
            ident = f"stagpy.config.{sec_class}.{opt}"
            print(f"[{opt}][{ident}] | {entry.doc} | {c_f}", file=fd)


full_doc_path = Path("user-guide") / "config-opts.md"

with mkdocs_gen_files.open(full_doc_path, "w") as fd:
    print("List of configuration options", file=fd)
    print("===", file=fd)
    print_config_list(fd)
