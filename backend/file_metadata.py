#!/usr/bin/env python3
"""
File Metadata Module

Manages metadata for uploaded files including original filenames,
extracted names, and other information needed for dynamic filename generation.
"""

import os
import json
import logging
from typing import Optional, Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)

class FileMetadataManager:
    """Manages file metadata storage and retrieval"""
    
    def __init__(self, metadata_dir: str = "metadata"):
        self.metadata_dir = metadata_dir
        self.ensure_directory()
    
    def ensure_directory(self):
        """Ensure metadata directory exists"""
        if not os.path.exists(self.metadata_dir):
            os.makedirs(self.metadata_dir)
    
    def store_metadata(self, file_id: str, metadata: Dict[str, Any]):
        """Store metadata for a file"""
        try:
            metadata_path = os.path.join(self.metadata_dir, f"{file_id}.json")
            
            # Add timestamp
            metadata['created_at'] = datetime.now().isoformat()
            metadata['updated_at'] = datetime.now().isoformat()
            
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Stored metadata for file_id: {file_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing metadata for {file_id}: {e}")
            return False
    
    def get_metadata(self, file_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve metadata for a file"""
        try:
            metadata_path = os.path.join(self.metadata_dir, f"{file_id}.json")
            
            if not os.path.exists(metadata_path):
                return None
            
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error retrieving metadata for {file_id}: {e}")
            return None
    
    def update_metadata(self, file_id: str, updates: Dict[str, Any]):
        """Update existing metadata"""
        try:
            existing_metadata = self.get_metadata(file_id) or {}
            existing_metadata.update(updates)
            existing_metadata['updated_at'] = datetime.now().isoformat()
            
            return self.store_metadata(file_id, existing_metadata)
            
        except Exception as e:
            logger.error(f"Error updating metadata for {file_id}: {e}")
            return False
    
    def get_original_filename(self, file_id: str) -> Optional[str]:
        """Get original filename for a file"""
        metadata = self.get_metadata(file_id)
        return metadata.get('original_filename') if metadata else None
    
    def get_extracted_name(self, file_id: str) -> Optional[str]:
        """Get extracted bearer name for a file"""
        metadata = self.get_metadata(file_id)
        return metadata.get('extracted_name') if metadata else None
    
    def cleanup_old_metadata(self, max_age_days: int = 7):
        """Clean up old metadata files"""
        try:
            current_time = datetime.now()
            
            for filename in os.listdir(self.metadata_dir):
                if not filename.endswith('.json'):
                    continue
                
                file_path = os.path.join(self.metadata_dir, filename)
                file_modified = datetime.fromtimestamp(os.path.getmtime(file_path))
                
                if (current_time - file_modified).days > max_age_days:
                    os.remove(file_path)
                    logger.info(f"Cleaned up old metadata: {filename}")
                    
        except Exception as e:
            logger.error(f"Error cleaning up metadata: {e}")

# Global instance
metadata_manager = FileMetadataManager()

def store_file_metadata(file_id: str, original_filename: str, extracted_name: Optional[str] = None):
    """Convenience function to store file metadata"""
    metadata = {
        'file_id': file_id,
        'original_filename': original_filename,
        'extracted_name': extracted_name
    }
    return metadata_manager.store_metadata(file_id, metadata)

def get_file_metadata(file_id: str) -> Optional[Dict[str, Any]]:
    """Convenience function to get file metadata"""
    return metadata_manager.get_metadata(file_id)

def update_extracted_name(file_id: str, extracted_name: str):
    """Convenience function to update extracted name"""
    return metadata_manager.update_metadata(file_id, {'extracted_name': extracted_name}) 