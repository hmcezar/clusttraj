__version__ = "1.2.0"

__all__ = [
    "main",
]


def main(args=None):
    """Run the clusttraj command-line entry point."""
    from .main import main as _main

    return _main(args)
