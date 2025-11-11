
#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import json
from pathlib import Path
from datetime import datetime

class ContributionTracker:
    """Track community contributions and statistics"""
    
    def __init__(self, community_dir):
        self.community_dir = Path(community_dir)
        self.contributions_file = self.community_dir / "contributions.json"
        self.stats_file = self.community_dir / "stats.json"
        
        self._initialize_files()
    
    def _initialize_files(self):
        """Initialize tracking files if they don't exist"""
        if not self.contributions_file.exists():
            with open(self.contributions_file, 'w' , encoding="UTF-8") as f:
                json.dump({"contributions": []}, f, indent=2)
        
        if not self.stats_file.exists():
            with open(self.stats_file, 'w', encoding="UTF-8") as f:
                json.dump({
                    "total_contributions": 0,
                    "components_contributed": {},
                    "last_contribution": None,
                    "total_models_downloaded": 0
                }, f, indent=2)
    
    def record_contribution(self, component_type, metadata, upload_results):
        """Record a new community contribution"""
        contribution = {
            "timestamp": datetime.now().isoformat(),
            "component_type": component_type,
            "metadata": metadata,
            "upload_results": upload_results,
            "contribution_id": self._generate_contribution_id()
        }
        
        # Load existing contributions
        with open(self.contributions_file, 'r', encoding="UTF-8") as f:
            data = json.load(f)
        
        # Add new contribution
        data["contributions"].append(contribution)
        
        # Save updated contributions
        with open(self.contributions_file, 'w', encoding="UTF-8") as f:
            json.dump(data, f, indent=2)
        
        # Update statistics
        self._update_stats(component_type)
    
    def _generate_contribution_id(self):
        """Generate unique contribution ID"""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        return f"contribution_{timestamp}"
    
    def _update_stats(self, component_type):
        """Update community statistics"""
        with open(self.stats_file, 'r', encoding="UTF-8") as f:
            stats = json.load(f)
        
        stats["total_contributions"] += 1
        stats["last_contribution"] = datetime.now().isoformat()
        
        # Update component counts
        if component_type in stats["components_contributed"]:
            stats["components_contributed"][component_type] += 1
        else:
            stats["components_contributed"][component_type] = 1
        
        # Save updated stats
        with open(self.stats_file, 'w', encoding="UTF-8") as f:
            json.dump(stats, f, indent=2)
    
    def get_stats(self):
        """Get community statistics"""
        with open(self.stats_file, 'r', encoding="UTF-8") as f:
            return json.load(f)
    
    def get_recent_contributions(self, limit=10):
        """Get recent contributions"""
        with open(self.contributions_file, 'r', encoding="UTF-8") as f:
            data = json.load(f)
        
        contributions = data["contributions"]
        return contributions[-limit:] if limit else contributions