import pytest
import subprocess


@pytest.fixture(params=[('ra-100000', 5), ('annulus', 100)])
def dir_isnap(request, repo_dir):
    return repo_dir / 'Examples' / request.param[0], request.param[1]


@pytest.fixture(params=[
    ('stagpy field -o=T', ['stagpy_T{:05d}.pdf']),
    ('stagpy field -o=T,v3', ['stagpy_T{:05d}.pdf',
                              'stagpy_v3{:05d}.pdf']),
])
def cmd_line(request):
    return request.param


@pytest.fixture
def all_cmd(cmd_line, dir_isnap):
    cmd = cmd_line[0]
    cmd += ' -p={}'.format(dir_isnap[0])
    expected_files = []
    for expfile in cmd_line[1]:
        expected_files.append(expfile.format(dir_isnap[1]))
    return cmd, expected_files


def test_cli(all_cmd, tmpdir):
    subprocess.run(all_cmd[0] + ' -n={}/stagpy'.format(tmpdir), shell=True)
    produced_files = tmpdir.listdir(sort=True)
    expected_files = [tmpdir / expfile for expfile in all_cmd[1]]
    assert produced_files == expected_files
