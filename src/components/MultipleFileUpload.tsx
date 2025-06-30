import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { 
  DocumentArrowUpIcon, 
  DocumentArrowDownIcon,
  CheckCircleIcon,
  XCircleIcon,
  ClockIcon,
  ArrowPathIcon,
  ArchiveBoxIcon
} from '@heroicons/react/24/outline';
import axios from 'axios';
import { toast } from 'react-hot-toast';
import { API_URL } from '../config';

interface FileResult {
  filename: string;
  status: 'uploaded' | 'converted' | 'error' | 'password_required' | 'converting';
  file_id?: string;
  error?: string;
  size?: number;
  type?: string;
  headers?: string[];
  row_count?: number;
  download_links?: {
    json: string;
    xlsx: string;
    csv: string;
  };
}

interface MultipleFileUploadProps {
  onComplete?: (results: FileResult[]) => void;
}

const MultipleFileUpload: React.FC<MultipleFileUploadProps> = ({ onComplete }) => {
  const [files, setFiles] = useState<FileResult[]>([]);
  const [isUploading, setIsUploading] = useState(false);
  const [isConverting, setIsConverting] = useState(false);
  const [password, setPassword] = useState('');
  const [showPasswordInput, setShowPasswordInput] = useState(false);

  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    if (acceptedFiles.length === 0) return;
    
    if (acceptedFiles.length > 10) {
      toast.error('Maximum 10 files allowed');
      return;
    }

    setIsUploading(true);
    const toastId = toast.loading(`Uploading ${acceptedFiles.length} files...`);

    try {
      const formData = new FormData();
      acceptedFiles.forEach(file => {
        formData.append('files', file);
      });
      
      if (password) {
        formData.append('password', password);
      }

      const response = await axios.post(`${API_URL}/upload-multiple`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      const results = response.data.results as FileResult[];
      setFiles(results);
      
      // Check if any files need password
      const needsPassword = results.some(f => f.status === 'password_required');
      if (needsPassword && !password) {
        setShowPasswordInput(true);
        toast.error('Some PDF files are password protected', { id: toastId });
      } else {
        toast.success(`Uploaded ${results.filter(f => f.status === 'uploaded').length} files successfully`, { id: toastId });
        
        // Auto-convert uploaded files
        await convertFiles(results.filter(f => f.status === 'uploaded').map(f => f.file_id!));
      }
    } catch (error: any) {
      console.error('Upload error:', error);
      toast.error('Upload failed: ' + (error.response?.data?.detail || error.message), { id: toastId });
    } finally {
      setIsUploading(false);
    }
  }, [password]);

  const convertFiles = async (fileIds: string[]) => {
    if (fileIds.length === 0) return;

    setIsConverting(true);
    const toastId = toast.loading(`Converting ${fileIds.length} files...`);

    try {
      const response = await axios.post(`${API_URL}/convert-multiple`, {
        file_ids: fileIds
      });

      const results = response.data.results as FileResult[];
      
      // Update file statuses
      setFiles(prevFiles => 
        prevFiles.map(file => {
          const updated = results.find(r => r.file_id === file.file_id);
          return updated ? { ...file, ...updated } : file;
        })
      );

      const converted = results.filter(r => r.status === 'converted').length;
      const errors = results.filter(r => r.status === 'error').length;
      
      if (converted > 0) {
        toast.success(`Converted ${converted} files successfully`, { id: toastId });
      }
      if (errors > 0) {
        toast.error(`${errors} files failed to convert`, { id: toastId });
      }

      if (onComplete) {
        onComplete(results);
      }
    } catch (error: any) {
      console.error('Conversion error:', error);
      toast.error('Conversion failed: ' + (error.response?.data?.detail || error.message), { id: toastId });
    } finally {
      setIsConverting(false);
    }
  };

  const retryWithPassword = async () => {
    if (!password.trim()) {
      toast.error('Please enter a password');
      return;
    }

    const passwordRequiredFiles = files.filter(f => f.status === 'password_required');
    if (passwordRequiredFiles.length === 0) return;

    setIsUploading(true);
    const toastId = toast.loading('Retrying with password...');

    try {
      // Re-upload password-protected files
      const formData = new FormData();
      // Note: We'd need to store the original File objects to re-upload
      // For now, we'll show a message to re-upload
      toast.error('Please re-upload the files with the password', { id: toastId });
      setShowPasswordInput(false);
    } catch (error: any) {
      toast.error('Password retry failed', { id: toastId });
    } finally {
      setIsUploading(false);
    }
  };

  const downloadAll = async (format: 'xlsx' | 'csv' = 'xlsx') => {
    const convertedFiles = files.filter(f => f.status === 'converted' && f.file_id);
    if (convertedFiles.length === 0) {
      toast.error('No converted files to download');
      return;
    }

    const toastId = toast.loading(`Preparing ${format.toUpperCase()} download...`);

    try {
      const response = await axios.post(`${API_URL}/download-multiple`, {
        file_ids: convertedFiles.map(f => f.file_id),
        format: format
      }, {
        responseType: 'blob'
      });

      const blob = new Blob([response.data], { type: 'application/zip' });
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      
      const timestamp = new Date().toISOString().slice(0, 16).replace(/[T:]/g, '_');
      link.setAttribute('download', `converted_files_${timestamp}.zip`);
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.URL.revokeObjectURL(url);

      toast.success(`Downloaded ${convertedFiles.length} files as ZIP`, { id: toastId });
    } catch (error: any) {
      console.error('Download error:', error);
      toast.error('Download failed', { id: toastId });
    }
  };

  const downloadSingle = async (file: FileResult, format: 'xlsx' | 'csv' = 'xlsx') => {
    if (!file.file_id) return;

    try {
      const response = await axios.get(`${API_URL}/api/download/${file.file_id}/${format}`, {
        responseType: 'blob'
      });

      const blob = new Blob([response.data]);
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      
      // Get filename from headers
      let filename = `${file.filename.replace(/\.[^/.]+$/, '')}.${format}`;
      const contentDisposition = response.headers['content-disposition'];
      if (contentDisposition) {
        const filenameMatch = contentDisposition.match(/filename[^;=\n]*=((['"]).*?\2|[^;\n]*)/);
        if (filenameMatch) {
          filename = filenameMatch[1].replace(/['"]/g, '');
        }
      }
      
      link.setAttribute('download', filename);
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.URL.revokeObjectURL(url);

      toast.success(`Downloaded ${filename}`);
    } catch (error: any) {
      console.error('Download error:', error);
      toast.error(`Failed to download ${file.filename}`);
    }
  };

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf'],
      'image/*': ['.png', '.jpg', '.jpeg'],
      'text/xml': ['.xml'],
      'text/plain': ['.txt']
    },
    maxFiles: 10,
    disabled: isUploading || isConverting
  });

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'uploaded':
      case 'converted':
        return <CheckCircleIcon className="h-5 w-5 text-green-500" />;
      case 'error':
      case 'password_required':
        return <XCircleIcon className="h-5 w-5 text-red-500" />;
      case 'converting':
        return <ArrowPathIcon className="h-5 w-5 text-blue-500 animate-spin" />;
      default:
        return <ClockIcon className="h-5 w-5 text-gray-400" />;
    }
  };

  const getStatusText = (file: FileResult) => {
    switch (file.status) {
      case 'uploaded':
        return 'Uploaded ✓';
      case 'converted':
        return `Converted ✓ (${file.row_count} rows)`;
      case 'error':
        return `Error: ${file.error}`;
      case 'password_required':
        return 'Password Required';
      case 'converting':
        return 'Converting...';
      default:
        return 'Pending';
    }
  };

  const convertedCount = files.filter(f => f.status === 'converted').length;
  const errorCount = files.filter(f => f.status === 'error').length;
  const uploadedCount = files.filter(f => f.status === 'uploaded').length;

  return (
    <div className="max-w-6xl mx-auto p-6 space-y-6">
      <div className="text-center">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-gray-100 mb-2">
          Multiple File Upload & Conversion
        </h2>
        <p className="text-gray-600 dark:text-gray-400">
          Upload multiple PDFs, images, or data files for batch processing
        </p>
      </div>

      {/* Upload Area */}
      <div
        {...getRootProps()}
        className={`
          border-2 border-dashed rounded-lg p-8 text-center cursor-pointer
          transition-colors duration-200 ease-in-out
          ${isDragActive 
            ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20' 
            : 'border-gray-300 dark:border-gray-600 hover:border-blue-500'
          }
          ${(isUploading || isConverting) ? 'opacity-50 cursor-not-allowed' : ''}
        `}
      >
        <input {...getInputProps()} />
        <DocumentArrowUpIcon className="mx-auto h-12 w-12 text-gray-400 mb-4" />
        <div className="space-y-2">
          <p className="text-lg font-medium text-gray-900 dark:text-gray-100">
            {isDragActive ? 'Drop your files here' : 'Drag & drop multiple files here'}
          </p>
          <p className="text-sm text-gray-500 dark:text-gray-400">
            or click to browse (PDF, PNG, JPG, XML, TXT) • Max 10 files • 100MB each
          </p>
          {(isUploading || isConverting) && (
            <p className="text-sm text-blue-600 dark:text-blue-400">
              {isUploading ? 'Uploading...' : 'Converting...'}
            </p>
          )}
        </div>
      </div>

      {/* Password Input */}
      {showPasswordInput && (
        <div className="bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded-lg p-4">
          <h3 className="text-lg font-medium text-yellow-800 dark:text-yellow-200 mb-2">
            Password Required
          </h3>
          <p className="text-sm text-yellow-700 dark:text-yellow-300 mb-3">
            Some PDF files are password protected. Please enter the password:
          </p>
          <div className="flex space-x-3">
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="Enter PDF password"
              className="flex-1 px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
            <button
              onClick={retryWithPassword}
              disabled={isUploading}
              className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50"
            >
              {isUploading ? 'Retrying...' : 'Retry'}
            </button>
          </div>
        </div>
      )}

      {/* Summary Stats */}
      {files.length > 0 && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="bg-blue-50 dark:bg-blue-900/20 p-4 rounded-lg">
            <div className="text-2xl font-bold text-blue-900 dark:text-blue-100">{files.length}</div>
            <div className="text-sm text-blue-700 dark:text-blue-300">Total Files</div>
          </div>
          <div className="bg-green-50 dark:bg-green-900/20 p-4 rounded-lg">
            <div className="text-2xl font-bold text-green-900 dark:text-green-100">{convertedCount}</div>
            <div className="text-sm text-green-700 dark:text-green-300">Converted</div>
          </div>
          <div className="bg-yellow-50 dark:bg-yellow-900/20 p-4 rounded-lg">
            <div className="text-2xl font-bold text-yellow-900 dark:text-yellow-100">{uploadedCount}</div>
            <div className="text-sm text-yellow-700 dark:text-yellow-300">Processing</div>
          </div>
          <div className="bg-red-50 dark:bg-red-900/20 p-4 rounded-lg">
            <div className="text-2xl font-bold text-red-900 dark:text-red-100">{errorCount}</div>
            <div className="text-sm text-red-700 dark:text-red-300">Errors</div>
          </div>
        </div>
      )}

      {/* Bulk Actions */}
      {convertedCount > 0 && (
        <div className="flex justify-center space-x-4">
          <button
            onClick={() => downloadAll('xlsx')}
            className="flex items-center px-6 py-3 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors"
          >
            <ArchiveBoxIcon className="h-5 w-5 mr-2" />
            Download All as Excel ZIP
          </button>
          <button
            onClick={() => downloadAll('csv')}
            className="flex items-center px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
          >
            <ArchiveBoxIcon className="h-5 w-5 mr-2" />
            Download All as CSV ZIP
          </button>
        </div>
      )}

      {/* File List */}
      {files.length > 0 && (
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow overflow-hidden">
          <div className="px-6 py-4 border-b border-gray-200 dark:border-gray-700">
            <h3 className="text-lg font-medium text-gray-900 dark:text-gray-100">
              File Processing Status
            </h3>
          </div>
          <div className="divide-y divide-gray-200 dark:divide-gray-700">
            {files.map((file, index) => (
              <div key={index} className="px-6 py-4 flex items-center justify-between">
                <div className="flex items-center space-x-3 flex-1">
                  {getStatusIcon(file.status)}
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium text-gray-900 dark:text-gray-100 truncate">
                      {file.filename}
                    </p>
                    <p className="text-sm text-gray-500 dark:text-gray-400">
                      {getStatusText(file)}
                      {file.size && ` • ${(file.size / 1024 / 1024).toFixed(1)}MB`}
                    </p>
                  </div>
                </div>
                
                {file.status === 'converted' && (
                  <div className="flex items-center space-x-2">
                    <button
                      onClick={() => downloadSingle(file, 'xlsx')}
                      className="px-3 py-1 text-xs bg-green-600 text-white rounded hover:bg-green-700"
                    >
                      Excel
                    </button>
                    <button
                      onClick={() => downloadSingle(file, 'csv')}
                      className="px-3 py-1 text-xs bg-blue-600 text-white rounded hover:bg-blue-700"
                    >
                      CSV
                    </button>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default MultipleFileUpload; 