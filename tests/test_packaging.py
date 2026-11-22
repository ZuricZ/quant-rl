"""The package is the Hub environment: every install route gets all of it."""

import pathlib
import subprocess
import tarfile
import zipfile

ROOT = pathlib.Path(__file__).parents[1]
PACKAGES = {"quantenv", "quant_research", "quant_research_notools"}


def test_wheel_built_from_the_sdist_carries_every_package(tmp_path):
    # verifiers installs a toolset from the sdist of its project, and a private
    # Hub install builds from the pushed folder; both build outside this
    # checkout, so nothing the package needs may live outside the project.
    subprocess.run(
        ["uv", "build", "--sdist", "--out-dir", str(tmp_path / "sdist"), str(ROOT)],
        check=True,
    )
    (sdist,) = (tmp_path / "sdist").glob("*.tar.gz")
    with tarfile.open(sdist) as archive:
        archive.extractall(tmp_path / "src", filter="data")
    (source,) = (tmp_path / "src").iterdir()
    subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(tmp_path / "wheel"), str(source)],
        check=True,
    )
    (wheel,) = (tmp_path / "wheel").glob("*.whl")
    names = zipfile.ZipFile(wheel).namelist()
    assert {n.split("/")[0] for n in names if ".dist-info/" not in n} == PACKAGES
    assert "quant_research/grade.py" in names
