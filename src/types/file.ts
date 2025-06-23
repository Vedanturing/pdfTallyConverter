export interface FileMetadata {
  id: string;
  original_name: string;
  file_size: number;
  mime_type: string;
  type: 'pdf' | 'image';
  num_pages?: number;
  width?: number;
  height?: number;
  format?: string;
}

export interface UploadFile {
  id: string;
  file: File;
  status: 'pending' | 'uploading' | 'uploaded' | 'error';
  progress: number;
  error?: string;
  metadata?: FileMetadata;
  uploadedAt?: Date;
} 