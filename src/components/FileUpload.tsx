import { useCallback, useState, useEffect } from 'react';
import { useDropzone } from 'react-dropzone';
import axios from 'axios';
import toast, { Toaster } from 'react-hot-toast';
import { API_URL } from '../config';
import { useNavigate } from 'react-router-dom';
import { initPdfWorker, cleanupPdfWorker, pdfjsLib } from '../utils/pdfjs-config';

// Initialize PDF.js worker
initPdfWorker();

interface FilePreview {
  file: File;
  preview: string;
  type: 'pdf' | 'image';
  numPages?: number;
  currentPage: number;
}

const FileUpload: React.FC = () => {
  const [filePreview, setFilePreview] = useState<FilePreview | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [showPasswordModal, setShowPasswordModal] = useState(false);
  const [password, setPassword] = useState('');
  const [passwordError, setPasswordError] = useState('');
  const [pendingFile, setPendingFile] = useState<File | null>(null);
  const navigate = useNavigate();

  // Initialize PDF.js worker when component mounts
  useEffect(() => {
    initPdfWorker();
    return () => {
      cleanupPdfWorker();
    };
  }, []);

  const handleUpload = async (file: File, password?: string) => {
    setIsUploading(true);
    setUploadProgress(0);

    const formData = new FormData();
    formData.append('file', file);
    if (password) {
      formData.append('password', password);
    }

    try {
      const response = await axios.post(`${API_URL}/upload`, formData, {
        timeout: 300000, // Increase timeout to 5 minutes
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        onUploadProgress: (progressEvent) => {
          const progress = Math.round(
            (progressEvent.loaded * 100) / (progressEvent.total || file.size)
          );
          setUploadProgress(progress);
        },
      });

      if (response.data.requires_password) {
        setPendingFile(file);
        setShowPasswordModal(true);
        return;
      }

      if (response.data.file_id) {
        toast.success('File uploaded successfully!');
        navigate('/preview', { 
          state: { 
            fileId: response.data.file_id,
            fileName: file.name,
            preview: filePreview?.preview,
            currentStep: 1
          }
        });
      } else {
        throw new Error(response.data.message || 'Upload failed');
      }
    } catch (error: any) {
      console.error('Upload error:', error);
      let errorMessage = 'Error uploading file. Please try again.';
      
      if (error.code === 'ECONNABORTED') {
        errorMessage = 'Upload timed out. The file might be too large or your connection might be slow. Please try again with a smaller file or check your connection.';
      } else if (error.response?.status === 401 && error.response?.data?.requires_password) {
        setPendingFile(file);
        setShowPasswordModal(true);
        return;
      } else if (error.response?.data?.message) {
        errorMessage = error.response.data.message;
      } else if (error.message) {
        errorMessage = error.message;
      }
      
      toast.error(errorMessage);
      setFilePreview(null);
    } finally {
      setIsUploading(false);
      setUploadProgress(0);
    }
  };

  const handlePasswordSubmit = async () => {
    if (!pendingFile || !password.trim()) {
      setPasswordError('Please enter a password');
      return;
    }

    try {
      setIsUploading(true);
      const formData = new FormData();
      formData.append('file', pendingFile);
      formData.append('password', password);

      const response = await axios.post(`${API_URL}/upload`, formData, {
        timeout: 300000, // 5 minutes
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        onUploadProgress: (progressEvent) => {
          const progress = Math.round(
            (progressEvent.loaded * 100) / (progressEvent.total || pendingFile.size)
          );
          setUploadProgress(progress);
        },
      });

      if (response.data.file_id) {
        toast.success('File uploaded successfully!');
        setShowPasswordModal(false);
        setPassword('');
        setPendingFile(null);
        setPasswordError('');
        navigate('/preview', { 
          state: { 
            fileId: response.data.file_id,
            fileName: pendingFile.name,
            currentStep: 1
          }
        });
      } else {
        throw new Error(response.data.message || 'Upload failed');
      }
    } catch (error: any) {
      console.error('Password submission error:', error);
      if (error.response?.status === 401) {
        setPasswordError('Incorrect password. Please try again.');
      } else {
        setPasswordError('Error processing file. Please try again.');
      }
    } finally {
      setIsUploading(false);
    }
  };

  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    const file = acceptedFiles[0];
    if (!file) return;

    if (file.type === 'application/pdf') {
      try {
        const arrayBuffer = await file.arrayBuffer();
        
        // Create a typed array from the array buffer
        const typedArray = new Uint8Array(arrayBuffer);
        
        const pdf = await pdfjsLib.getDocument({ data: typedArray }).promise;
        const page = await pdf.getPage(1);
        const viewport = page.getViewport({ scale: 1.0 });
        
        const canvas = document.createElement('canvas');
        const context = canvas.getContext('2d');
        if (!context) {
          throw new Error('Could not get canvas context');
        }
        
        canvas.height = viewport.height;
        canvas.width = viewport.width;
        
        await page.render({
          canvasContext: context,
          viewport: viewport
        }).promise;
        
        setFilePreview({
          file,
          preview: canvas.toDataURL(),
          type: 'pdf',
          numPages: pdf.numPages,
          currentPage: 1
        });
      } catch (error: any) {
        console.error('Error loading PDF:', error);
        // Check if error is due to password protection
        if (error.message?.includes('password')) {
          setPendingFile(file);
          setShowPasswordModal(true);
          return;
        }
        toast.error(`Error loading PDF: ${error.message}`);
        return;
      }
    } else if (file.type.startsWith('image/')) {
      setFilePreview({
        file,
        preview: URL.createObjectURL(file),
        type: 'image',
        currentPage: 1
      });
    }

    // Start upload
    handleUpload(file);
  }, [navigate]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf'],
      'image/*': ['.png', '.jpg', '.jpeg', '.gif']
    },
    disabled: isUploading,
    maxFiles: 1
  });

  return (
    <div className="max-w-4xl mx-auto">
      <Toaster position="top-right" />
      
      {/* Password Modal */}
      {showPasswordModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white p-6 rounded-lg shadow-xl max-w-md w-full">
            <h3 className="text-lg font-semibold mb-4">PDF Password Required</h3>
            <form onSubmit={(e) => {
              e.preventDefault();
              handlePasswordSubmit();
            }}>
              <input
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="Enter PDF password"
                className="w-full p-2 border rounded mb-4"
                autoFocus
              />
              {passwordError && (
                <p className="text-red-500 text-sm mb-4">{passwordError}</p>
              )}
              <div className="flex justify-end space-x-4">
                <button
                  type="button"
                  onClick={() => {
                    setShowPasswordModal(false);
                    setPassword('');
                    setPendingFile(null);
                  }}
                  className="px-4 py-2 text-gray-600 hover:text-gray-800"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
                  disabled={isUploading}
                >
                  {isUploading ? 'Uploading...' : 'Submit'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
      
      <div
        {...getRootProps()}
        className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors
          ${isDragActive ? 'border-blue-500 bg-blue-50' : 'border-gray-300 hover:border-gray-400'}
          ${isUploading ? 'opacity-50 cursor-not-allowed' : ''}`}
      >
        <input {...getInputProps()} />
        <div className="text-gray-600">
          {isUploading ? (
            <div className="flex flex-col items-center">
              <div className="w-full max-w-xs bg-gray-200 rounded-full h-2.5 mb-4">
                <div 
                  className="bg-blue-600 h-2.5 rounded-full transition-all duration-300" 
                  style={{ width: `${uploadProgress}%` }}
                />
              </div>
              <p className="text-lg">Uploading... {uploadProgress}%</p>
            </div>
          ) : (
            <>
              <p className="text-lg mb-2">Drag and drop your files here</p>
              <p className="text-sm">or click to select files</p>
              <p className="text-xs mt-2">(PDF and image files only)</p>
            </>
          )}
        </div>
      </div>

      {filePreview && !isUploading && (
        <div className="mt-8">
          <div className="bg-white rounded-lg shadow-lg p-4">
            <div className="flex justify-center">
              <img
                src={filePreview.preview}
                alt="File preview"
                className="max-w-full max-h-[600px] object-contain"
              />
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default FileUpload; 