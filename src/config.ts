export const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export const SUPPORTED_FILE_TYPES = {
  'application/pdf': ['.pdf'],
  'image/*': ['.png', '.jpg', '.jpeg']
};

export const MAX_FILE_SIZE = 10 * 1024 * 1024; // 10MB

export const MAX_FILES = 3;

export const VALIDATION_LEVELS = {
  CRITICAL: 'critical',
  WARNING: 'warning',
  INFO: 'info'
} as const;

export const FILE_EXTENSIONS = {
  PDF: '.pdf',
  JPEG: '.jpeg',
  JPG: '.jpg',
  PNG: '.png'
} as const;