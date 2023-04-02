from __future__ import annotations

import re
import subprocess
from pathlib import Path

from pytest import FixtureRequest, fixture


@fixture(params=[("ra-100000", 5, 1000), ("annulus", 100, 3000)])
def dir_isnap(request: FixtureRequest, repo_dir: Path) -> tuple[Path, int, int]:
    return (
        repo_dir / "Examples" / request.param[0],
        request.param[1],
        request.param[2],
    )


@fixture(
    params=[
        ("stagpy field", ["stagpy_T_stream{:05d}.pdf"]),
        ("stagpy field -o=T.v3", ["stagpy_T_v3{:05d}.pdf"]),
        ("stagpy field -o=T-v3", ["stagpy_T{:05d}.pdf", "stagpy_v3{:05d}.pdf"]),
    ]
)
def all_cmd_field(
    request: FixtureRequest, dir_isnap: tuple[Path, int, int]
) -> tuple[str, list[str]]:
    cmd = request.param[0]
    cmd += " -p={}".format(dir_isnap[0])
    expected_files = []
    for expfile in request.param[1]:
        expected_files.append(expfile.format(dir_isnap[1]))
    return cmd, expected_files


@fixture(
    params=[
        ("stagpy rprof", ["stagpy_rprof_Tmean_{}.pdf"]),
        ("stagpy rprof -o=Tmean,vzabs", ["stagpy_rprof_Tmean_vzabs_{}.pdf"]),
    ]
)
def all_cmd_rprof(
    request: FixtureRequest, dir_isnap: tuple[Path, int, int]
) -> tuple[str, list[str]]:
    cmd = request.param[0]
    cmd += " -p={}".format(dir_isnap[0])
    expected_files = []
    for expfile in request.param[1]:
        expected_files.append(expfile.format(dir_isnap[2]))
    return cmd, expected_files


@fixture(
    params=[
        ("stagpy time -o Tmean.Nutop", ["stagpy_time_Tmean_Nutop.pdf"]),
    ]
)
def all_cmd_time(
    request: FixtureRequest, dir_isnap: tuple[str, int, int]
) -> tuple[str, list[str]]:
    cmd = request.param[0]
    cmd += " -p={}".format(dir_isnap[0])
    return cmd, request.param[1]


@fixture(
    params=[
        (
            "stagpy plates -o v2.dv2 -continents --field T",
            [
                "stagpy_plates_surf_v2_dv2_{:05d}.pdf",
                "stagpy_plates_T{:05d}.pdf",
                "stagpy_plates_trenches_snaps[-1,].dat",
            ],
        ),
    ]
)
def all_cmd_plates(
    request: FixtureRequest, dir_isnap: tuple[str, int, int]
) -> tuple[str, list[str]]:
    cmd = request.param[0]
    cmd += " -p={}".format(dir_isnap[0])
    expected_files = []
    for expfile in request.param[1]:
        expected_files.append(expfile.format(dir_isnap[1]))
    return cmd, expected_files


def helper_test_cli(all_cmd: tuple[str, list[str]], tmp: Path) -> None:
    subprocess.run(all_cmd[0] + " -n={}/stagpy".format(tmp), shell=True)
    produced_files = sorted(tmp.iterdir())
    expected_files = [tmp / expfile for expfile in sorted(all_cmd[1])]
    assert produced_files == expected_files


def test_field_cli(all_cmd_field: tuple[str, list[str]], tmp_path: Path) -> None:
    helper_test_cli(all_cmd_field, tmp_path)


def test_rprof_cli(all_cmd_rprof: tuple[str, list[str]], tmp_path: Path) -> None:
    helper_test_cli(all_cmd_rprof, tmp_path)


def test_time_cli(all_cmd_time: tuple[str, list[str]], tmp_path: Path) -> None:
    helper_test_cli(all_cmd_time, tmp_path)


def test_plates_cli(all_cmd_plates: tuple[str, list[str]], tmp_path: Path) -> None:
    helper_test_cli(all_cmd_plates, tmp_path)


def test_err_cli() -> None:
    subp = subprocess.run("stagpy field", shell=True, stderr=subprocess.PIPE)
    reg = re.compile(rb"^Oops!.*\nPlease.*\n\nNoParFileError.*$")
    assert reg.match(subp.stderr)
