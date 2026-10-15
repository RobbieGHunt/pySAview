"""Source measurement files are read-only; derived outputs are separate."""
from pathlib import Path


def validate_output_path(destination, source_paths=()):
    """Reject NeXus outputs and aliases of source files before opening for write."""
    target = Path(destination)
    if target.suffix.lower() in {".nxs", ".nexus", ".h5", ".hdf5", ".hdf"}:
        raise ValueError("Measurement files are read-only. Choose a separate output file with the appropriate export extension.")
    resolved = target.resolve()
    for source in source_paths:
        original = Path(source)
        if resolved == original.resolve() or (target.exists() and original.exists() and target.samefile(original)):
            raise ValueError("Cannot overwrite a source measurement file. Choose a separate output file.")
    return target
