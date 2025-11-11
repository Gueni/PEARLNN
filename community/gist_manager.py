import requests
import json

class GistManager:
    """Manage GitHub Gists for metadata storage"""
    
    def __init__(self):
        self.gist_url = "https://api.github.com/gists"
    
    def upload_metadata(self, metadata):
        """Upload metadata to GitHub Gist"""
        gist_data = {
            "description": "PEARLNN Model Metadata",
            "public": True,
            "files": {
                "metadata.json": {
                    "content": json.dumps(metadata, indent=2)
                }
            }
        }
        
        response = requests.post(self.gist_url, json=gist_data, timeout=30)
        
        if response.status_code == 201:
            result = response.json()
            return result['html_url']
        else:
            raise Exception(f"Gist creation failed: {response.status_code}")
    
    def download_metadata(self, gist_id):
        """Download metadata from Gist"""
        url = f"https://api.github.com/gists/{gist_id}"
        response = requests.get(url, timeout=30)
        
        if response.status_code == 200:
            gist_data = response.json()
            metadata_file = gist_data['files']['metadata.json']
            metadata_content = metadata_file['content']
            return json.loads(metadata_content)
        else:
            raise Exception(f"Gist download failed: {response.status_code}")