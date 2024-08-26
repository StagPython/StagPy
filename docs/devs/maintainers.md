Releasing process
=================

This section is intended for maintainers of the project. It describes how new
versions of StagPy are released on PyPI.

The version number lives in `pyproject.toml`. Releases are tagged, with the
annotations of the tag containing release notes.
Please make sure to always provide a patch version number (i.e. use a version
number with *three* levels such as `1.0.0` instead of `1.0`).

```sh title="shell"
# edit version number in pyproject.toml
git add pyproject.toml
git commit -m "release vX.Y.Z"
git tag -a vX.Y.Z
git push --follow-tags
```

Releasing on PyPI is then performed automatically by a GitHub action workflow.
