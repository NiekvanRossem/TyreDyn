from pathlib import Path

def _find_project_root(marker: str = "example_tyres"):
    """Small script that find the project root folder path, as well as commonly used folders. Written by ChatGPT."""

    path = Path(__file__).resolve()
    for parent in path.parents:
        if (parent / marker).exists():
            return parent
    return Path.cwd()

# commonly used folders to be imported elsewhere
PROJECT_ROOT = _find_project_root()
TYRE_DIR = PROJECT_ROOT / "example_tyres"