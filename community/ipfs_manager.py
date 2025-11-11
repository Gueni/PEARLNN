
import requests

class IPFSManager:
    """Manage IPFS operations for model storage"""
    
    def __init__(self):
        self.gateways = [
            "https://ipfs.io/ipfs/",
            "https://gateway.pinata.cloud/ipfs/",
            "https://cloudflare-ipfs.com/ipfs/",
            "https://dweb.link/ipfs/"
        ]
    
    def upload_file(self, file_path):
        """Upload file to IPFS using public gateways"""
        # Try Infura's public IPFS API
        try:
            with open(file_path, 'rb') as f:
                files = {'file': f}
                response = requests.post(
                    'https://ipfs.infura.io:5001/api/v0/add',
                    files=files,
                    timeout=60
                )
            
            if response.status_code == 200:
                result = response.json()
                return result['Hash']
        except Exception as e:
            print(f"Infura IPFS upload failed: {e}")
        
        # Fallback: Try Web3.Storage
        try:
            return self._upload_web3_storage(file_path)
        except Exception as e:
            print(f"Web3.Storage upload failed: {e}")
        
        raise Exception("All IPFS upload methods failed")
    
    def _upload_web3_storage(self, file_path):
        """Upload using Web3.Storage free tier"""
        url = "https://api.web3.storage/upload"
        
        with open(file_path, 'rb') as f:
            headers = {
                'Authorization': 'Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJkaWQ6ZXRocjoweEQ4MjJmN0U4OTVhQzVCOTg1ODM0MTg4MTczODZCMjA2OEU0N0U0RkIiLCJpc3MiOiJ3ZWIzLXN0b3JhZ2UiLCJpYXQiOjE2OTkyNTU0MDE4MjAsIm5hbWUiOiJwZWFybG5uIn0.DummyTokenForNow',  # Would be real token
                'X-Client': 'PEARLNN'
            }
            response = requests.post(url, headers=headers, files={'file': f}, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            return result['cid']
        
        raise Exception(f"Web3.Storage error: {response.status_code}")
    
    def download_file(self, ipfs_hash, destination_path):
        """Download file from IPFS"""
        for gateway in self.gateways:
            try:
                url = f"{gateway}{ipfs_hash}"
                response = requests.get(url, timeout=30)
                
                if response.status_code == 200:
                    with open(destination_path, 'wb') as f:
                        f.write(response.content)
                    return destination_path
            except Exception as e:
                print(f"Gateway {gateway} failed: {e}")
                continue
        
        raise Exception("All IPFS gateways failed")
    
    def pin_file(self, ipfs_hash):
        """Pin file to keep it available"""
        # This would pin the file using a pinning service
        # For now, just return success
        return True