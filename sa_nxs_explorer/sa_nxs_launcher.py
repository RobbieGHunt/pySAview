"""Direct launcher script for SuperADAM NeXus Explorer."""

from pathlib import Path
import sys

# Ensure root workspace is in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from sa_nxs_explorer.app import main

if __name__ == "__main__":
    sys.exit(main())
