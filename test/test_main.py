import os
import pytest
from clusttraj.main import main


def test_main(tmp_path):
    main(
        [
            "test/ref/testtraj.xyz",
            "--min-rmsd",
            "1.0",
            "-np",
            "1",
            "-i",
            "test/ref/test_distmat.npy",
            "-oc",
            os.path.join(tmp_path, "clusters.dat"),
        ]
    )

    assert os.path.exists(os.path.join(tmp_path, "clusters.dat"))
    assert os.path.exists(os.path.join(tmp_path, "clusters.out"))


def test_main_nclusters(tmp_path):
    main(
        [
            "test/ref/testtraj.xyz",
            "--n-clusters",
            "2",
            "-np",
            "1",
            "-i",
            "test/ref/test_distmat.npy",
            "-oc",
            os.path.join(tmp_path, "clusters.dat"),
        ]
    )

    assert os.path.exists(os.path.join(tmp_path, "clusters.dat"))
    assert os.path.exists(os.path.join(tmp_path, "clusters.out"))


def test_main_nclusters_too_many_clusters(tmp_path):
    with pytest.raises(SystemExit, match="Cannot request 4 clusters"):
        main(
            [
                "test/ref/testtraj.xyz",
                "--n-clusters",
                "4",
                "-np",
                "1",
                "-i",
                "test/ref/test_distmat.npy",
                "-oc",
                os.path.join(tmp_path, "clusters.dat"),
            ]
        )


def test_main_save_medoids(tmp_path):
    main(
        [
            "test/ref/testtraj.xyz",
            "--n-clusters",
            "2",
            "-np",
            "1",
            "-i",
            "test/ref/test_distmat.npy",
            "-oc",
            os.path.join(tmp_path, "clusters.dat"),
            "-mc",
            "xyz",
        ]
    )

    assert os.path.exists(os.path.join(tmp_path, "clusters_medoids.xyz"))
