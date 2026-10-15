# Project invariants

- Original measurement files are read-only. Never open a source NeXus/HDF5 file in a writable mode or overwrite it with transformed data, exports, ROI definitions, or session state.
- Keep transformations in memory. Save derived data, ROI definitions, and sessions to separate files. All output features must use the source-path validation guard before writing.
- Each scan owns its ROI independently. Do not automatically inherit an ROI from a previous scan. Explicit reuse creates an independent working definition and must preserve the original file ROI.
- Preserve normalization and uncertainty conventions documented in docs/normalization.md. Display transformations must not modify numerical results or source data.
- Use the project .venv Python and run relevant regression tests.
