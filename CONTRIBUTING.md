# Contributing

- `pip install -e packages/vidlabel-spec -e packages/vidlabel pytest`, then `pytest`.
- The spec (`vidlabel-spec`) is the contract. Changing a column or metadata field means bumping `SPEC_VERSION`,
  updating `docs/SPEC.md`, and adding a test. Custom data never needs a spec change: put it in `extra`.
- New mask codecs: add a value to `MaskCodec`, a module next to `masks.py`, and a lossless round-trip test.
- Converters to and from other formats are welcome as separate adapters. They only need to produce ordinary annotation rows
  (origin is a column, never a separate structure).
