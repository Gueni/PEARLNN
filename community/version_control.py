import hashlib
import json
from datetime import datetime
from pathlib import Path

class VersionControl:
    """Manage model versions and updates"""
    
    def __init__(self, models_dir):
        self.models_dir = Path(models_dir)
        self.versions_file = self.models_dir / "model_versions.json"
        self._initialize_versions()
    
    def _initialize_versions(self):
        """Initialize versions file if it doesn't exist"""
        if not self.versions_file.exists():
            with open(self.versions_file, 'w', encoding="UTF-8") as f:
                json.dump({"versions": {}}, f, indent=2)
    
    def register_model_version(self, component_type, model_path, metadata=None):
        """Register a new model version"""
        if metadata is None:
            metadata = {}
        
        model_hash = self._calculate_model_hash(model_path)
        version_info = {
            "hash": model_hash,
            "timestamp": datetime.now().isoformat(),
            "file_size": Path(model_path).stat().st_size,
            "metadata": metadata,
            "component_type": component_type
        }
        
        # Load existing versions
        with open(self.versions_file, 'r', encoding="UTF-8") as f:
            versions_data = json.load(f)
        
        # Add new version
        if component_type not in versions_data["versions"]:
            versions_data["versions"][component_type] = []
        
        versions_data["versions"][component_type].append(version_info)
        
        # Keep only recent versions
        versions_data["versions"][component_type] = versions_data["versions"][component_type][-10:]
        
        # Save updated versions
        with open(self.versions_file, 'w', encoding="UTF-8") as f:
            json.dump(versions_data, f, indent=2)
        
        return version_info
    
    def get_latest_version(self, component_type):
        """Get latest version for component type"""
        with open(self.versions_file, 'r', encoding="UTF-8") as f:
            versions_data = json.load(f)
        
        if component_type in versions_data["versions"] and versions_data["versions"][component_type]:
            return versions_data["versions"][component_type][-1]
        return None
    
    def check_for_updates(self, component_type, current_hash):
        """Check if newer version is available"""
        latest_version = self.get_latest_version(component_type)
        if latest_version and latest_version["hash"] != current_hash:
            return latest_version
        return None
    
    def _calculate_model_hash(self, model_path):
        """Calculate hash of model file"""
        sha256_hash = hashlib.sha256()
        
        with open(model_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        
        return sha256_hash.hexdigest()