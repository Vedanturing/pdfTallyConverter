#!/usr/bin/env python3
"""
Parser Configuration

Configuration settings for the hybrid bank statement parser
"""

import os
from typing import Dict, Any

class ParserConfig:
    """Configuration class for parser settings"""
    
    def __init__(self):
        # Hybrid parser settings
        self.ENABLE_HYBRID_PARSER = os.getenv('ENABLE_HYBRID_PARSER', 'true').lower() == 'true'
        self.ENABLE_ADVANCED_TABLE_DETECTION = os.getenv('ENABLE_ADVANCED_TABLE_DETECTION', 'true').lower() == 'true'
        
        # OCR Engine preferences (in order of preference)
        self.OCR_ENGINE_PRIORITY = [
            'paddleocr',    # Best for table structure
            'easyocr',      # Good fallback
            'tesseract',    # Traditional OCR
            'simple'        # Basic CV fallback
        ]
        
        # Table detection model preferences
        self.TABLE_DETECTION_MODELS = [
            'table-transformer',  # Microsoft Table Transformer
            'contour',           # OpenCV contour detection
            'line',              # Line-based detection
        ]
        
        # Parsing confidence thresholds
        self.MIN_TABLE_CONFIDENCE = 0.5
        self.MIN_CELL_CONFIDENCE = 0.3
        self.MIN_OCR_CONFIDENCE = 0.2
        
        # Bank statement patterns and keywords
        self.BANK_KEYWORDS = [
            'bank statement', 'account statement', 'transaction history',
            'monthly statement', 'savings account', 'current account'
        ]
        
        # Performance settings
        self.MAX_PAGES_TO_PROCESS = 10
        self.IMAGE_SCALE_FACTOR = 2.0  # Scale factor for PDF to image conversion
        self.PARALLEL_PROCESSING = False  # Disable for stability
        
        # Debugging and logging
        self.DEBUG_MODE = os.getenv('PARSER_DEBUG', 'false').lower() == 'true'
        self.SAVE_DEBUG_IMAGES = self.DEBUG_MODE
        self.DEBUG_OUTPUT_DIR = os.path.join(os.getcwd(), 'debug_output')
        
        # Fallback settings
        self.FALLBACK_TO_ORIGINAL_PARSER = True
        self.STRICT_VALIDATION = False  # If True, reject results that don't meet quality standards
        
    def get_settings(self) -> Dict[str, Any]:
        """Get all configuration settings as a dictionary"""
        return {
            'hybrid_parser_enabled': self.ENABLE_HYBRID_PARSER,
            'advanced_detection_enabled': self.ENABLE_ADVANCED_TABLE_DETECTION,
            'ocr_priority': self.OCR_ENGINE_PRIORITY,
            'table_detection_models': self.TABLE_DETECTION_MODELS,
            'confidence_thresholds': {
                'table': self.MIN_TABLE_CONFIDENCE,
                'cell': self.MIN_CELL_CONFIDENCE,
                'ocr': self.MIN_OCR_CONFIDENCE
            },
            'performance': {
                'max_pages': self.MAX_PAGES_TO_PROCESS,
                'scale_factor': self.IMAGE_SCALE_FACTOR,
                'parallel': self.PARALLEL_PROCESSING
            },
            'debug': {
                'enabled': self.DEBUG_MODE,
                'save_images': self.SAVE_DEBUG_IMAGES,
                'output_dir': self.DEBUG_OUTPUT_DIR
            }
        }
    
    def update_setting(self, key: str, value: Any):
        """Update a configuration setting"""
        if hasattr(self, key.upper()):
            setattr(self, key.upper(), value)
        else:
            raise ValueError(f"Unknown configuration key: {key}")

# Global configuration instance
config = ParserConfig()

# Environment-based feature flags
def is_hybrid_parser_enabled() -> bool:
    """Check if hybrid parser is enabled"""
    return config.ENABLE_HYBRID_PARSER

def is_advanced_detection_enabled() -> bool:
    """Check if advanced table detection is enabled"""
    return config.ENABLE_ADVANCED_TABLE_DETECTION

def get_preferred_ocr_engines() -> list:
    """Get preferred OCR engines in order"""
    return config.OCR_ENGINE_PRIORITY

def get_confidence_thresholds() -> Dict[str, float]:
    """Get confidence thresholds for different components"""
    return {
        'table': config.MIN_TABLE_CONFIDENCE,
        'cell': config.MIN_CELL_CONFIDENCE,
        'ocr': config.MIN_OCR_CONFIDENCE
    } 