"""`python -m vidlabel_spec [out.json]` writes the JSON Schema."""
import sys

from ._schema import json_schema, write_json_schema

if len(sys.argv) > 1:
    write_json_schema(sys.argv[1])
else:
    import json

    print(json.dumps(json_schema(), indent=2))
