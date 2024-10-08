Contributing
============

The development of StagPy is made using the Git version control system. The
first three chapters of the [Git book](https://git-scm.com/book/en/v2) should
give you all the necessary basic knowledge to use Git for this project.

If you want to contribute to development of StagPy, create an account on
[GitHub](https://github.com/) and fork the [StagPy
repository](https://github.com/StagPython/StagPy).

To get a local copy of your fork of StagPy, clone it (here using [the SSH
protocol](https://docs.github.com/en/authentication/connecting-to-github-with-ssh):

```sh title="shell"
git clone git@github.com:YOUR_USER_NAME/StagPy.git
cd StagPy
```

Then add a remote (here called `upstream`) pointing to the main StagPy
repository:

```sh title="shell"
git remote add upstream git@github.com:StagPython/StagPy.git
```

To sync your fork with the main repository, you can run the following:

```sh title="shell"
git switch master
git pull upstream master
git push origin
```

To add your own modifications, create a new branch from the tip of master:

```sh title="shell"
git switch -c branch-name master
```

where `branch-name` is the desired branch name.  Modify the code as desired,
commit it, and push it on your fork:

```sh title="shell"
git push -u origin branch-name
```

You can then create a PR from your fork on GitHub to have your changes
incorporated in the main repository and made available to other users.

Testing
-------

StagPy uses [uv to manage dev environments](https://docs.astral.sh/uv/).
It also uses [just to run commands](https://just.systems/man/en/).

Launching `just` from anywhere in the repository will automatically run checks
and tests in a virtual environment. Before submitting modifications to the
code, please make sure they pass the tests by running `just`.

Documentation
-------------

The StagPy documentation is built with [MkDocs](https://www.mkdocs.org/). To
build it locally, run:

```sh title="shell"
just mkdocs serve
```
