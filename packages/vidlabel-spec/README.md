# vidlabel-spec

The specification of the `vidlabel` video annotation format, written as
[pydantic](https://docs.pydantic.dev) models and exportable as JSON Schema so other languages can implement it.
It has almost no dependencies. Install it on its own just to validate metadata; install `vidlabel` to read and write datasets.

```python
from vidlabel_spec import VideoMetadata, Manifest, ANNOTATION_COLUMNS, json_schema
```
