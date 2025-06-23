import { FileMetadata } from '../types/file';

const API_BASE_URL = 'http://localhost:3001';

export async function uploadFile(file: File, onProgress?: (progress: number) => void): Promise<FileMetadata> {
  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch(`${API_BASE_URL}/convert`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Upload failed');
  }

  return response.json();
}

export async function getFileInfo(fileId: string): Promise<FileMetadata> {
  const response = await fetch(`${API_BASE_URL}/files/${fileId}`);
  
  if (!response.ok) {
    throw new Error('Failed to fetch file info');
  }

  return response.json();
}

export async function listFiles(): Promise<FileMetadata[]> {
  const response = await fetch(`${API_BASE_URL}/files`);
  
  if (!response.ok) {
    throw new Error('Failed to fetch files');
  }

  const data = await response.json();
  return data.files;
} 