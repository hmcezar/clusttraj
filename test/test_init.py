import subprocess
import sys


def test_import_clusttraj_does_not_eagerly_import_main():
    """Verify that importing clusttraj does not eagerly import clusttraj.main."""
    script = (
        "import clusttraj; "
        "import sys; "
        "print(sys.modules.get('clusttraj.main', 'NOT_FOUND'))"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "NOT_FOUND"
