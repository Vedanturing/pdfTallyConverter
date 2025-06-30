import { useState, useCallback } from 'react';
import axios from 'axios';
import { API_URL } from '../config';

interface FileMetadata {
  original_filename: string | null;
  extracted_name: string | null;
  has_extracted_name: boolean;
}

interface DynamicFilenameResponse {
  filename: string;
  metadata: FileMetadata;
}

export const useDynamicFilename = () => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const generateFilename = useCallback(async (
    fileId: string,
    format: string = 'xlsx',
    language: string = 'en'
  ): Promise<string> => {
    setLoading(true);
    setError(null);

    try {
      const response = await axios.get<DynamicFilenameResponse>(
        `${API_URL}/api/filename/${fileId}`,
        {
          params: { format, language }
        }
      );

      return response.data.filename;
    } catch (err: any) {
      const errorMessage = err.response?.data?.detail || 'Failed to generate dynamic filename';
      setError(errorMessage);
      
      // Return fallback filename
      const timestamp = new Date().toISOString()
        .slice(0, 19)
        .replace(/[T:]/g, '_')
        .replace(/-/g, '-');
      
      return `export_${fileId.slice(0, 8)}_${timestamp}.${format}`;
    } finally {
      setLoading(false);
    }
  }, []);

  const getFileMetadata = useCallback(async (
    fileId: string
  ): Promise<FileMetadata | null> => {
    try {
      const response = await axios.get<DynamicFilenameResponse>(
        `${API_URL}/api/filename/${fileId}`
      );

      return response.data.metadata;
    } catch (err) {
      console.warn('Failed to fetch file metadata:', err);
      return null;
    }
  }, []);

  return {
    generateFilename,
    getFileMetadata,
    loading,
    error
  };
}; 