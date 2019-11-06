import pytest


@pytest.fixture(scope='session')
def repo_dir():
    import pathlib
    return pathlib.Path(__file__).parent.parent.resolve()


@pytest.fixture(scope='session',
                params=['ra-100000', 'annulus'])
def example_dir(request, repo_dir):
    return repo_dir / 'Examples' / request.param


@pytest.fixture(scope='module')
def sdat(example_dir):
    import stagpy.stagyydata
    return stagpy.stagyydata.StagyyData(example_dir)


@pytest.fixture(scope='module')
def step(sdat):
    return sdat.snaps[-1]
