Releasing process
=================

This section is intended for maintainers of the project. It describes how new
versions of StagPy are released on PyPI.

The version number lives in `pyproject.toml`. Releases are tagged, with the
annotations of the tag containing release notes.
Please make sure to always provide a patch version number (i.e. use a version
number with *three* levels such as `1.0.0` instead of `1.0`).

```sh title="shell"
just release X.Y.Z
```

This will update the version number to `X.Y.Z` and create an annotated tag for
you. Once you've written the release notes, check the created commit. If
satisfactory, `git push --follow-tags` will trigger publication on PyPI via a
GitHub Action.
