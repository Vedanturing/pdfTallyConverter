import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import axios from 'axios';
import { toast } from 'react-hot-toast';
import { useNavigate } from 'react-router-dom';
import { CloudArrowUpIcon, DocumentTextIcon } from '@heroicons/react/24/outline';
import { API_URL } from '../config';

interface UploadResponse {
  status: string;
  message: string;
  file_id: string;
}

const FileUploader: React.FC = () => {
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [currentFile, setCurrentFile] = useState<File | null>(null);
  const navigate = useNavigate();

  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    if (acceptedFiles.length === 0) return;
    
    const file = acceptedFiles[0];
    console.log('File selected:', { name: file.name, type: file.type, size: file.size }); // Debug log
    
    setCurrentFile(file);
    setUploading(true);
    setUploadProgress(0);

    const formData = new FormData();
    formData.append('file', file);

    const loadingToast = toast.loading('Uploading file...');

    try {
      console.log('Making upload request to:', `${API_URL}/upload`); // Debug log
      
      const response = await axios.post<UploadResponse>(`${API_URL}/upload`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
          'Accept': 'application/json',
        },
        withCredentials: true,
        onUploadProgress: (progressEvent) => {
          if (progressEvent.total) {
            const progress = Math.round(
              (progressEvent.loaded * 100) / progressEvent.total
            );
            setUploadProgress(progress);
          }
        }
      });

      console.log('Upload response:', response.data); // Debug log

      if (response.data.status === 'success') {
        toast.success('File uploaded successfully!');
        // Wait for a moment to ensure the file is processed
        await new Promise(resolve => setTimeout(resolve, 500));
        // Navigate to preview page with file details
        navigate('/preview', { 
          state: { 
            fileId: response.data.file_id,
            fileName: file.name,
            currentStep: 1 // Set to Preview step
          }
        });
      } else {
        throw new Error('No file ID received from server');
      }
    } catch (error) {
      console.error('Upload error:', error);
      let errorMessage = 'Failed to upload file';
      
      if (axios.isAxiosError(error)) {
        if (error.response) {
          // Server responded with an error
          errorMessage = error.response.data?.message || error.response.data?.detail || errorMessage;
        } else if (error.request) {
          // Request was made but no response received
          errorMessage = 'Server is not responding. Please try again later.';
        }
      }
      
      toast.error(errorMessage, {
        id: loadingToast
      });
    } finally {
      setUploading(false);
      setUploadProgress(0);
    }
  }, [navigate]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf'],
      'image/*': ['.png', '.jpg', '.jpeg']
    },
    multiple: false,
    maxSize: 10 * 1024 * 1024 // 10MB
  });

  return (
    <div className="max-w-xl mx-auto">
      <div
        {...getRootProps()}
        className={`border-2 border-dashed rounded-lg p-12 transition-colors ${
          isDragActive ? 'border-blue-500 bg-blue-50' : 'border-gray-300 hover:border-blue-400'
        }`}
      >
        <input {...getInputProps()} />
        <div className="text-center">
          {uploading ? (
            <div className="space-y-3">
              <CloudArrowUpIcon className="mx-auto h-12 w-12 text-blue-500 animate-bounce" />
              <div className="w-full bg-gray-200 rounded-full h-2.5">
                <div
                  className="bg-blue-600 h-2.5 rounded-full transition-all duration-300"
                  style={{ width: `${uploadProgress}%` }}
                />
              </div>
              <p className="text-sm text-gray-600">Uploading... {uploadProgress}%</p>
            </div>
          ) : currentFile ? (
            <div className="space-y-3">
              <DocumentTextIcon className="mx-auto h-12 w-12 text-blue-500" />
              <p className="text-sm font-medium text-gray-900">{currentFile.name}</p>
              <p className="text-xs text-gray-500">
                {(currentFile.size / 1024 / 1024).toFixed(2)} MB
              </p>
            </div>
          ) : (
            <div className="space-y-3">
              <CloudArrowUpIcon className={`mx-auto h-12 w-12 ${
                isDragActive ? 'text-blue-500' : 'text-gray-400'
              }`} />
              <p className="text-base text-gray-600">
                {isDragActive
                  ? 'Drop the file here...'
                  : 'Drag & drop your file here, or click to browse'}
              </p>
              <p className="text-sm text-gray-500">
                Supported formats: PDF, PNG, JPG, JPEG (max. 10MB)
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default FileUploader; 