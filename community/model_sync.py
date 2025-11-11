
import hashlib
from pathlib import Path
from datetime import datetime

from community.ipfs_manager import IPFSManager
from community.gist_manager import GistManager
from community.contribution_tracker import ContributionTracker

class ModelSync:
    """Synchronize models with the PEARLNN community"""
    
    def __init__(self, base_dir="."):
        self.base_dir = Path(base_dir)
        self.community_dir = self.base_dir / "community"
        self.community_dir.mkdir(exist_ok=True)
        
        self.ipfs_manager = IPFSManager()
        self.gist_manager = GistManager()
        self.contribution_tracker = ContributionTracker(self.community_dir)
    
    def download_latest_models(self, component_types=None):
        """Download latest community models"""
        if component_types is None:
            component_types = ["mosfet", "opamp", "capacitor", "bjt", "inductor", "diode"]
        
        downloaded = {}
        
        for component in component_types:
            try:
                model_path = self._download_component_model(component)
                if model_path:
                    downloaded[component] = model_path
                    print(f"✅ Downloaded {component} model")
            except Exception as e:
                print(f"❌ Failed to download {component} model: {e}")
        
        return downloaded
    
    def upload_model(self, model_path, component_type, metadata=None):
        """Upload improved model to community"""
        if metadata is None:
            metadata = {}
        
        # Add metadata
        metadata.update({
            'component_type': component_type,
            'upload_time': datetime.now().isoformat(),
            'model_hash': self._calculate_file_hash(model_path),
            'file_size': Path(model_path).stat().st_size
        })
        
        # Upload to multiple services for redundancy
        results = {}
        
        try:
            # Upload to IPFS
            ipfs_hash = self.ipfs_manager.upload_file(model_path)
            results['ipfs'] = ipfs_hash
        except Exception as e:
            results['ipfs'] = f"Failed: {e}"
        
        try:
            # Upload metadata to Gist
            gist_url = self.gist_manager.upload_metadata(metadata)
            results['gist'] = gist_url
        except Exception as e:
            results['gist'] = f"Failed: {e}"
        
        # Track contribution
        self.contribution_tracker.record_contribution(component_type, metadata, results)
        
        return results
    
    def _download_component_model(self, component_type):
        """Download model for specific component type"""
        # Try IPFS first
        try:
            latest_hash = self._get_latest_model_hash(component_type)
            if latest_hash:
                model_path = self.community_dir / f"{component_type}_model.pt"
                self.ipfs_manager.download_file(latest_hash, model_path)
                return model_path
        except Exception as e:
            print(f"IPFS download failed for {component_type}: {e}")
        
        return None
    
    def _get_latest_model_hash(self, component_type):
        """Get latest model hash from community index"""
        # This would query a community endpoint
        # For now, return None (will be implemented with actual community)
        return None
    
    def _calculate_file_hash(self, file_path):
        """Calculate SHA256 hash of file"""
        sha256_hash = hashlib.sha256()
        
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        
        return sha256_hash.hexdigest()
    
    def get_community_stats(self):
        """Get community statistics"""
        return self.contribution_tracker.get_stats()