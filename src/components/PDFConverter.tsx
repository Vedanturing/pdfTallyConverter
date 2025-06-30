import React, { useState, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Document, Page, pdfjs } from 'react-pdf';
import { useDropzone } from 'react-dropzone';
import {
  ArrowPathIcon,
  DocumentArrowUpIcon,
  DocumentMagnifyingGlassIcon,
  TableCellsIcon,
  CheckCircleIcon,
  ArrowLeftIcon,
} from '@heroicons/react/24/outline';
import FinancialTable from './FinancialTable';
import WorkflowStepper from './WorkflowStepper';
import { API_URL } from '../config';
import { FinancialEntry } from '../types/financial';
import toast from 'react-hot-toast';
import type { PDFDocumentProxy } from 'pdfjs-dist';
import { Upload, FileText, ChevronLeft, ChevronRight, Loader2 } from 'lucide-react';
import { Button } from './ui/button';
import { convertPDF } from '../utils/pdf-utils';
import { useLocation } from 'react-router-dom';

// Initialize PDF worker
pdfjs.GlobalWorkerOptions.workerSrc = new URL(
  'pdfjs-dist/build/pdf.worker.min.js',
  import.meta.url
).href;

interface Step {
  title: string;
  icon: React.ComponentType<any>;
}

interface PDFConverterProps {
  onConvert: (data: FinancialEntry[]) => void;
  onPdfUrl: (url: string) => void;
  onStartValidation: () => void;
}

const PDFConverter: React.FC<PDFConverterProps> = ({
  onConvert,
  onPdfUrl,
  onStartValidation,
}) => {
  const location = useLocation();
  const [file, setFile] = useState<File | null>(null);
  const [pdfUrl, setPdfUrl] = useState<string | null>(null);
  const [numPages, setNumPages] = useState<number | null>(null);
  const [currentPage, setCurrentPage] = useState(1);
  const [convertedData, setConvertedData] = useState<FinancialEntry[] | null>(null);
  const [isConverting, setIsConverting] = useState(false);
  const [currentStep, setCurrentStep] = useState(0);
  const [fileId, setFileId] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [sortColumn, setSortColumn] = useState<keyof FinancialEntry | null>(null);
  const [sortDirection, setSortDirection] = useState<'asc' | 'desc'>('asc');

  useEffect(() => {
    const state = location.state as { data?: FinancialEntry[], currentStep?: number };
    if (state?.data) {
      setConvertedData(state.data);
    }
    if (state?.currentStep) {
      setCurrentStep(state.currentStep);
    }
  }, [location.state]);

  // Cleanup function for PDF resources
  useEffect(() => {
    return () => {
      if (pdfUrl) {
        URL.revokeObjectURL(pdfUrl);
      }
    };
  }, [pdfUrl]);

  const steps: Step[] = [
    { title: 'Upload', icon: DocumentArrowUpIcon },
    { title: 'Preview', icon: DocumentMagnifyingGlassIcon },
    { title: 'Convert', icon: TableCellsIcon },
    { title: 'Validate', icon: CheckCircleIcon },
  ];

  const goBack = () => {
    if (currentStep === 1) {
      if (pdfUrl) {
        URL.revokeObjectURL(pdfUrl);
        setPdfUrl(null);
      }
      setFile(null);
      setFileId(null);
    } else if (currentStep === 2) {
      setConvertedData(null);
    }
    setCurrentStep(Math.max(0, currentStep - 1));
  };

  const onDrop = useCallback((acceptedFiles: File[]) => {
    const file = acceptedFiles[0];
    if (file && file.type === 'application/pdf') {
      const url = URL.createObjectURL(file);
      setPdfUrl(url);
      onPdfUrl(url);
      setCurrentPage(1);
      setConvertedData(null);
    } else {
      toast.error('Please upload a valid PDF file');
    }
  }, [onPdfUrl]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf'],
    },
    maxFiles: 1,
  });

  const handleConvert = async () => {
    if (!pdfUrl) return;
    
    setIsLoading(true);
    try {
      const result = await convertPDF(pdfUrl);
      setConvertedData(result);
      onConvert(result);
      onStartValidation();
      toast.success('PDF converted successfully!');
    } catch (error) {
      console.error('Error converting PDF:', error);
      toast.error('Failed to convert PDF');
    } finally {
      setIsLoading(false);
    }
  };

  const onDocumentLoadSuccess = ({ numPages }: { numPages: number }) => {
    setNumPages(numPages);
  };

  const onDocumentLoadError = (error: Error) => {
    console.error('Error loading PDF:', error);
    toast.error('Failed to load PDF. Please try uploading again.');
  };

  const handleExport = async (format: 'xlsx' | 'csv' | 'xml') => {
    try {
      const response = await fetch(`${API_URL}/export/${format}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ data: convertedData }),
      });

      if (!response.ok) {
        throw new Error(`Export failed: ${response.statusText}`);
      }

      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      
      // Get filename from headers or generate default
      let filename = `financial_data.${format}`;
      const contentDisposition = response.headers.get('content-disposition');
      if (contentDisposition) {
        const filenameMatch = contentDisposition.match(/filename[^;=\n]*=((['"]).*?\2|[^;\n]*)/);
        if (filenameMatch) {
          filename = filenameMatch[1].replace(/['"]/g, '');
        }
      }
      
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);

      toast.success(`Successfully exported to ${format.toUpperCase()}`);
    } catch (error: any) {
      console.error('Export error:', error);
      toast.error(`Failed to export: ${error.message}`);
    }
  };

  const handleSort = (column: keyof FinancialEntry) => {
    if (!convertedData) return;

    const direction = column === sortColumn && sortDirection === 'asc' ? 'desc' : 'asc';
    setSortDirection(direction);
    setSortColumn(column);

    const sortedData = [...convertedData].sort((a, b) => {
      const aValue = a[column];
      const bValue = b[column];

      if (typeof aValue === 'number' && typeof bValue === 'number') {
        return direction === 'asc' ? aValue - bValue : bValue - aValue;
      }

      const aString = String(aValue).toLowerCase();
      const bString = String(bValue).toLowerCase();
      return direction === 'asc'
        ? aString.localeCompare(bString)
        : bString.localeCompare(aString);
    });

    setConvertedData(sortedData);
  };

  const renderStep = () => {
    switch (currentStep) {
      case 0: // Upload
        return (
          <div
            {...getRootProps()}
            className={`
              border-2 border-dashed rounded-lg p-8 text-center cursor-pointer
              transition-colors duration-200 ease-in-out
              ${isDragActive 
                ? 'border-primary bg-primary/5' 
                : 'border-gray-300 hover:border-primary'
              }
            `}
          >
            <input {...getInputProps()} />
            <motion.div
              initial={{ scale: 1 }}
              whileHover={{ scale: 1.02 }}
              className="space-y-4"
            >
              <Upload className="mx-auto h-12 w-12 text-gray-400" />
              <div className="space-y-2">
                <p className="text-lg font-medium">
                  {isDragActive ? 'Drop your PDF here' : 'Drag & drop your PDF here'}
                </p>
                <p className="text-sm text-gray-500">
                  or click to browse from your computer
                </p>
              </div>
            </motion.div>
          </div>
        );

      case 1: // Preview
        return (
          <div className="space-y-4">
            <div className="flex justify-between items-center mb-4">
              <button
                onClick={goBack}
                className="flex items-center px-3 py-2 text-gray-600 hover:text-gray-900"
              >
                <ArrowLeftIcon className="h-5 w-5 mr-2" />
                Back
              </button>
              <button
                onClick={handleConvert}
                disabled={!fileId}
                className={`px-4 py-2 rounded ${
                  fileId
                    ? 'bg-blue-500 text-white hover:bg-blue-600'
                    : 'bg-gray-300 text-gray-500 cursor-not-allowed'
                }`}
              >
                Convert to Table
              </button>
            </div>
            <div className="bg-white rounded-lg shadow-lg p-4">
              <Document
                file={pdfUrl}
                onLoadSuccess={onDocumentLoadSuccess}
                onLoadError={onDocumentLoadError}
              >
                <Page pageNumber={currentPage} />
              </Document>
            </div>
          </div>
        );

      case 2: // Convert
        return (
          <div className="space-y-4">
            {convertedData && (
              <FinancialTable
                data={convertedData}
                onSort={handleSort}
                sortColumn={sortColumn}
                sortDirection={sortDirection}
              />
            )}
          </div>
        );

      default:
        return null;
    }
  };

  return (
    <div className="max-w-4xl mx-auto">
      {renderStep()}
    </div>
  );
};

export default PDFConverter; 