from pathlib import Path

from pytest import FixtureRequest, fixture

from stagpy.stagyydata import StagyyData, Step


@fixture(scope="session")
def repo_dir() -> Path:
    return Path(__file__).parent.parent.resolve()


@fixture(
    scope="session",
    params=[
        "ra-100000",
        "annulus",
        "blankenbach/serial",
        "blankenbach/parallel",
    ],
)
def example_dir(request: FixtureRequest, repo_dir: Path) -> Path:
    return repo_dir / "Examples" / request.param


@fixture(
    scope="session",
    params=[
        "ra-100000",
        "annulus",
    ],
)
def example_legacy_path(request: FixtureRequest, repo_dir: Path) -> Path:
    return repo_dir / "Examples" / request.param


@fixture(
    scope="session",
    params=[
        "blankenbach/serial",
        "blankenbach/parallel",
    ],
)
def example_h5_path(request: FixtureRequest, repo_dir: Path) -> Path:
    return repo_dir / "Examples" / request.param


@fixture(scope="module")
def sdat(example_dir: Path) -> StagyyData:
    return StagyyData(example_dir)


@fixture(scope="module")
def sdat_legacy(example_legacy_path: Path) -> StagyyData:
    return StagyyData(example_legacy_path)


@fixture(scope="module")
def sdat_h5(example_h5_path: Path) -> StagyyData:
    return StagyyData(example_h5_path)


@fixture(scope="module")
def step(sdat: StagyyData) -> Step:
    return sdat.snaps[-1]
