import os
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
