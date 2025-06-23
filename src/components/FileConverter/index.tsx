import React, { useState, useCallback } from 'react';
import { toast } from 'react-hot-toast';
import FileUploader from '../FileUploader';
import FilePreview from '../FilePreview';
import ValidationTable from '../ValidationTable';
import { PDFValidationResult, PDFProcessingOptions } from '../../types/ValidationTypes';
import { FileMetadata, UploadFile } from '../../types/file';

interface ConversionResult {
  success: boolean;
  message: string;
  data?: any;
}

const FileConverter: React.FC = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isConverting, setIsConverting] = useState(false);
  const [conversionResult, setConversionResult] = useState<ConversionResult | null>(null);
  const [pdfValidation, setPdfValidation] = useState<PDFValidationResult | null>(null);
  const [password, setPassword] = useState('');
  const [showPasswordInput, setShowPasswordInput] = useState(false);
  const [uploadedFiles, setUploadedFiles] = useState<UploadFile[]>([]);

  const validatePDF = async (file: File, options?: PDFProcessingOptions) => {
    const formData = new FormData();
    formData.append('file', file);
    if (options?.password) {
      formData.append('password', options.password);
    }

    try {
      const response = await fetch('/api/validate-pdf', {
        method: 'POST',
        body: formData,
      });

      const result = await response.json();

      if (!response.ok) {
        throw new Error(result.message || 'PDF validation failed');
      }

      setPdfValidation(result);
      
      if (result.isProtected && result.needsPassword && !options?.password) {
        setShowPasswordInput(true);
        return false;
      }
      
      return true;
    } catch (error) {
      console.error('PDF validation error:', error);
      toast.error('Failed to validate PDF');
      return false;
    }
  };

  const handleUploadComplete = useCallback((metadata: FileMetadata[]) => {
    const newUploadedFiles = metadata.map(meta => ({
      id: meta.id,
      file: selectedFile as File,
      status: 'uploaded' as const,
      progress: 100,
      metadata: meta,
      uploadedAt: new Date()
    }));
    setUploadedFiles(newUploadedFiles);
    
    if (selectedFile) {
      validatePDF(selectedFile);
    }
  }, [selectedFile]);

  const handlePasswordSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!selectedFile) return;

    const isValid = await validatePDF(selectedFile, { password });
    if (isValid) {
      setShowPasswordInput(false);
      toast.success('Password accepted');
    } else {
      toast.error('Invalid password');
    }
  };

  const handleConversion = async () => {
    if (!selectedFile) {
      toast.error('Please select a file first');
      return;
    }

    if (pdfValidation?.isProtected && pdfValidation?.needsPassword && !password) {
      toast.error('Please enter the PDF password');
      setShowPasswordInput(true);
      return;
    }

    setIsConverting(true);
    try {
      const formData = new FormData();
      formData.append('file', selectedFile);
      if (password) {
        formData.append('password', password);
      }

      const response = await fetch('/api/convert', {
        method: 'POST',
        body: formData,
      });

      const result = await response.json();

      if (!response.ok) {
        throw new Error(result.message || 'Conversion failed');
      }

      setConversionResult({
        success: true,
        message: 'Conversion successful',
        data: result.data,
      });
      toast.success('File converted successfully');
    } catch (error) {
      console.error('Conversion error:', error);
      setConversionResult({
        success: false,
        message: error instanceof Error ? error.message : 'An error occurred during conversion',
      });
      toast.error('Conversion failed');
    } finally {
      setIsConverting(false);
    }
  };

  const handleFileSelect = useCallback((file: File) => {
    setSelectedFile(file);
    setConversionResult(null);
    setPdfValidation(null);
    setShowPasswordInput(false);
    setPassword('');
  }, []);

  return (
    <div className="space-y-6">
      <div className="bg-white rounded-lg shadow p-6">
        <FileUploader
          onUploadComplete={handleUploadComplete}
          maxFiles={1}
        />
      </div>

      {showPasswordInput && (
        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-lg font-medium mb-4">PDF is password protected</h3>
          <form onSubmit={handlePasswordSubmit} className="space-y-4">
            <div>
              <label htmlFor="password" className="block text-sm font-medium text-gray-700">
                Enter PDF Password
              </label>
              <input
                type="password"
                id="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
                required
              />
            </div>
            <button
              type="submit"
              className="w-full bg-blue-500 text-white py-2 px-4 rounded hover:bg-blue-600"
            >
              Submit Password
            </button>
          </form>
        </div>
      )}

      {selectedFile && !showPasswordInput && uploadedFiles[0]?.metadata && (
        <div className="bg-white rounded-lg shadow p-6">
          <FilePreview
            file={selectedFile}
            metadata={uploadedFiles[0].metadata}
            previewUrl={URL.createObjectURL(selectedFile)}
          />
          <div className="mt-4">
            <button
              onClick={handleConversion}
              disabled={isConverting}
              className={`w-full px-4 py-2 text-white rounded-md ${
                isConverting
                  ? 'bg-gray-400 cursor-not-allowed'
                  : 'bg-blue-600 hover:bg-blue-700'
              }`}
            >
              {isConverting ? 'Converting...' : 'Convert File'}
            </button>
          </div>
        </div>
      )}

      {conversionResult && (
        <div className="bg-white rounded-lg shadow p-6">
          <ValidationTable data={conversionResult.data} />
        </div>
      )}
    </div>
  );
};

export default FileConverter;
