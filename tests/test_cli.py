import pytest
import subprocess


@pytest.fixture(params=[('ra-100000', 5, 1000), ('annulus', 100, 3000)])
def dir_isnap(request, repo_dir):
    return (repo_dir / 'Examples' / request.param[0],
            request.param[1], request.param[2])


@pytest.fixture(params=[
    ('stagpy field', ['stagpy_T_stream{:05d}.pdf']),
    ('stagpy field -o=T.v3', ['stagpy_T_v3{:05d}.pdf']),
    ('stagpy field -o=T-v3', ['stagpy_T{:05d}.pdf',
                              'stagpy_v3{:05d}.pdf']),
])
def all_cmd_field(request, dir_isnap):
    cmd = request.param[0]
    cmd += ' -p={}'.format(dir_isnap[0])
    expected_files = []
    for expfile in request.param[1]:
        expected_files.append(expfile.format(dir_isnap[1]))
    return cmd, expected_files


@pytest.fixture(params=[
    ('stagpy rprof', ['stagpy_rprof_Tmean_{}.pdf']),
    ('stagpy rprof -o=Tmean,vzabs', ['stagpy_rprof_Tmean_vzabs_{}.pdf']),
])
def all_cmd_rprof(request, dir_isnap):
    cmd = request.param[0]
    cmd += ' -p={}'.format(dir_isnap[0])
    expected_files = []
    for expfile in request.param[1]:
        expected_files.append(expfile.format(dir_isnap[2]))
    return cmd, expected_files


@pytest.fixture(params=[
    ('stagpy time -o Tmean.Nutop', ['stagpy_time_Tmean_Nutop.pdf']),
])
def all_cmd_time(request, dir_isnap):
    cmd = request.param[0]
    cmd += ' -p={}'.format(dir_isnap[0])
    return cmd, request.param[1]


def helper_test_cli(all_cmd, tmp):
    subprocess.run(all_cmd[0] + ' -n={}/stagpy'.format(tmp), shell=True)
    produced_files = tmp.listdir(sort=True)
    expected_files = [tmp / expfile for expfile in all_cmd[1]]
    assert produced_files == expected_files


def test_field_cli(all_cmd_field, tmpdir):
    helper_test_cli(all_cmd_field, tmpdir)


def test_rprof_cli(all_cmd_rprof, tmpdir):
    helper_test_cli(all_cmd_rprof, tmpdir)


def test_time_cli(all_cmd_time, tmpdir):
    helper_test_cli(all_cmd_time, tmpdir)
