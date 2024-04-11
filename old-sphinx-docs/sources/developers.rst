Contributing
============

The development of StagPy is made using the Git version control system. The
first three chapters of the `Git book`__ should give you all the necessary
basic knowledge to use Git for this project.

.. __: https://git-scm.com/book/en/v2

If you want to contribute to development of StagPy, create an account on
GitHub_ and fork the `StagPy repository`__.

.. _GitHub: https://github.com/
.. __: https://github.com/StagPython/StagPy

To get a local copy of your fork of StagPy, clone it (here using `the SSH
protocol`__)::

    % git clone git@github.com:YOUR_USER_NAME/StagPy.git
    % cd StagPy

.. __: https://help.github.com/articles/connecting-to-github-with-ssh/

Then add a remote (here called ``upstream``) pointing to the main StagPy
repository::

    % git remote add upstream git@github.com:StagPython/StagPy.git

To sync your fork with the main repository, you can run the following::

    % git switch master
    % git pull upstream master
    % git push origin

To add your own modifications, create a new branch from the tip of master::

    % git switch -c branch-name master

where ``branch-name`` is the desired branch name.  Modify the code as desired,
commit it, and push it on your fork::

    % git push -u origin branch-name

You can then create a PR from your fork on GitHub to have your changes
incorporated in the main repository and made available to other users.

Testing
-------

StagPy uses tox_ for code testing.  Make sure it is installed and up to date on
your system::

    % python3 -m pip install -U tox

.. _tox: https://tox.readthedocs.io

Launching ``tox`` in the root of the repository will automatically run the
tests in a virtual environment. Before submitting modifications to the code,
please make sure they pass the tests by running ``tox``.

Documentation
-------------

The StagPy documentation is built with Sphinx_. To build it locally, install
the needed packages::

    % python3 -m pip install -r docs/requirements.txt

.. _Sphinx: https://www.sphinx-doc.org

Then, in the ``docs`` directory, run::

    % make html

Open the produced file ``_build/html/index.html`` in your navigator to browse
your local version of the documentation.
