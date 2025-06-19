from typing import Dict, Optional
import yaml
from pathlib import Path
import jsonschema

class YAMLValidator:
    """Validator for YAML files against a schema"""
    
    def __init__(self, schema_path: Optional[str] = None):
        self.schema: Optional[dict] = None
        if schema_path is None:
            schema_path = str(Path(__file__).parent / "schema.yaml")
        self._load_schema(schema_path)
        
    def _load_schema(self, schema_path: str) -> None:
        """Load the YAML schema file"""
        schema_file = Path(schema_path)
        if not schema_file.exists():
            raise FileNotFoundError(f"Schema file not found: {schema_file}")
            
        with schema_file.open() as f:
            self.schema = yaml.safe_load(f)
            
    def validate(self, yaml_data: dict) -> dict:
        """Validate YAML data against the schema and return the validated data"""
        if not self.schema:
            raise RuntimeError("Schema not loaded")
            
        try:
            jsonschema.validate(yaml_data, self.schema)
            return yaml_data
        except jsonschema.exceptions.ValidationError as e:
            raise ValueError(f"Invalid YAML: {str(e)}")
            
    def validate_file(self, yaml_path: str) -> dict:
        """Validate a YAML file against the schema and return the validated data"""
        yaml_file = Path(yaml_path)
        if not yaml_file.exists():
            raise FileNotFoundError(f"YAML file not found: {yaml_file}")
            
        with yaml_file.open() as f:
            yaml_data = yaml.safe_load(f)
            return self.validate(yaml_data)

    @staticmethod
    def get_container_config(yaml_data: dict) -> dict:
        """Get container configuration from yaml data"""
        return yaml_data["container"]

    @staticmethod
    def get_deployment_config(yaml_data: dict) -> dict:
        """Get deployment configuration from yaml data"""
        return yaml_data["deployment"]

    @staticmethod
    def get_task_server_config(yaml_data: dict) -> dict:
        """Get taskServer configuration from yaml data"""
        return yaml_data["taskServer"]

    @staticmethod
    def get_capabilities(yaml_data: dict) -> list:
        """Get list of capabilities from yaml data"""
        return yaml_data["capabilities"]

    @staticmethod
    def get_metadata(yaml_data: dict) -> dict:
        """Get metadata from yaml data"""
        return yaml_data["metadata"] 