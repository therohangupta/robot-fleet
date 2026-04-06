from typing import Optional
from pathlib import Path

import yaml
import jsonschema


class YAMLValidator:
    """Validator for YAML files against a schema."""

    def __init__(self, schema_path: Optional[str] = None):
        self.schema: Optional[dict] = None
        if schema_path is None:
            schema_path = str(Path(__file__).parent / "schema.yaml")
        self._load_schema(schema_path)

    def _load_schema(self, schema_path: str) -> None:
        schema_file = Path(schema_path)
        if not schema_file.exists():
            raise FileNotFoundError(f"Schema file not found: {schema_file}")

        with schema_file.open() as f:
            self.schema = yaml.safe_load(f)

    def validate(self, yaml_data: dict) -> dict:
        if not self.schema:
            raise RuntimeError("Schema not loaded")

        try:
            jsonschema.validate(yaml_data, self.schema)
            return yaml_data
        except jsonschema.exceptions.ValidationError as e:
            raise ValueError(f"Invalid YAML: {str(e)}")

    def validate_file(self, yaml_path: str) -> dict:
        yaml_file = Path(yaml_path)
        if not yaml_file.exists():
            raise FileNotFoundError(f"YAML file not found: {yaml_file}")

        with yaml_file.open() as f:
            yaml_data = yaml.safe_load(f)
            return self.validate(yaml_data)

