import { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import axios from 'axios';
import toast, { Toaster } from 'react-hot-toast';
import { API_URL } from '../config';
import { useNavigate } from 'react-router-dom';

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
  const navigate = useNavigate();

  const handleUpload = async (file: File) => {
    setIsUploading(true);
    setUploadProgress(0);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post('/api/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        onUploadProgress: (progressEvent) => {
          const progress = progressEvent.total
            ? Math.round((progressEvent.loaded * 100) / progressEvent.total)
            : 0;
          setUploadProgress(progress);
        },
        timeout: 120000, // Increased to 120 seconds
        validateStatus: function (status) {
          return status >= 200 && status < 300;
        },
      });

      if (response.data.status === 'success') {
        toast.success('File uploaded successfully!');
        // Wait for a moment to ensure the file is processed
        await new Promise(resolve => setTimeout(resolve, 500));
        // Navigate to preview page with file details
        navigate('/preview', { 
          state: { 
            fileId: response.data.file_id,
            fileName: file.name,
            preview: filePreview?.preview,
            currentStep: 1 // Set to Preview step
          }
        });
      } else {
        throw new Error(response.data.message || 'Upload failed');
      }
    } catch (error: any) {
      console.error('Upload error:', error);
      let errorMessage = 'Error uploading file. Please try again.';
      if (error.code === 'ECONNABORTED') {
        errorMessage = 'Upload timed out. Please try again with a smaller file or check your connection.';
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

  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    const file = acceptedFiles[0];
    if (!file) return;

    if (file.type === 'application/pdf') {
      try {
        const arrayBuffer = await file.arrayBuffer();
        const pdfjsLib = await import('pdfjs-dist');
        const pdf = await pdfjsLib.getDocument(arrayBuffer).promise;
        const page = await pdf.getPage(1);
        const viewport = page.getViewport({ scale: 1.0 });
        
        const canvas = document.createElement('canvas');
        const context = canvas.getContext('2d');
        canvas.height = viewport.height;
        canvas.width = viewport.width;
        
        await page.render({
          canvasContext: context!,
          viewport: viewport
        }).promise;
        
        setFilePreview({
          file,
          preview: canvas.toDataURL(),
          type: 'pdf',
          numPages: pdf.numPages,
          currentPage: 1
        });
      } catch (error) {
        console.error('Error loading PDF:', error);
        toast.error('Error loading PDF preview');
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