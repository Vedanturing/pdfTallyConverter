import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
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
  const [file, setFile] = useState<File | null>(null);
  const [pdfUrl, setPdfUrl] = useState<string | null>(null);
  const [numPages, setNumPages] = useState<number>(0);
  const [currentPage, setCurrentPage] = useState<number>(1);
  const [convertedData, setConvertedData] = useState<FinancialEntry[] | null>(null);
  const [isConverting, setIsConverting] = useState(false);
  const [currentStep, setCurrentStep] = useState(0);
  const [fileId, setFileId] = useState<string | null>(null);

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

  const { getRootProps, getInputProps } = useDropzone({
    accept: {
      'application/pdf': ['.pdf'],
    },
    maxFiles: 1,
    onDrop: async (acceptedFiles) => {
      const file = acceptedFiles[0];
      if (pdfUrl) {
        URL.revokeObjectURL(pdfUrl);
      }
      
      setFile(file);
      const url = URL.createObjectURL(file);
      setPdfUrl(url);
      onPdfUrl(url);
      setCurrentStep(1);

      const formData = new FormData();
      formData.append('file', file);

      try {
        const response = await fetch(`${API_URL}/upload`, {
          method: 'POST',
          body: formData,
        });

        if (!response.ok) {
          throw new Error('Upload failed');
        }

        const data = await response.json();
        if (data.file_id) {
          setFileId(data.file_id);
          toast.success('File uploaded successfully');
        }
      } catch (error) {
        console.error('Upload error:', error);
        toast.error('Failed to upload file');
      }
    },
  });

  const handleConvert = async () => {
    if (!fileId) {
      toast.error('No file uploaded');
      return;
    }

    setIsConverting(true);
    setCurrentStep(2);

    try {
      console.log('Converting file with ID:', fileId);
      const convertResponse = await fetch(`${API_URL}/convert/${fileId}`, {
        method: 'POST',
      });

      if (!convertResponse.ok) {
        throw new Error('Conversion failed');
      }

      const convertData = await convertResponse.json();
      console.log('Conversion response:', convertData);

      const dataResponse = await fetch(`${API_URL}/get-data/${fileId}`);
      if (!dataResponse.ok) {
        throw new Error('Failed to get converted data');
      }

      const data = await dataResponse.json();
      console.log('Data response:', data);

      if (data && data.rows) {
        // Transform the data to match the FinancialEntry format
        const transformedData = data.rows.map((row: any, index: number) => ({
          id: `row-${index}`,
          date: row.Date || row.DATE || '',
          voucherNo: row['Voucher No'] || row['VOUCHER NO'] || '',
          ledgerName: row['Ledger Name'] || row['LEDGER NAME'] || '',
          amount: typeof row.Amount === 'string' ? parseFloat(row.Amount.replace(/[^0-9.-]+/g, '')) || 0 : row.Amount || 0,
          narration: row.Narration || row.NARRATION || '',
          balance: typeof row.Balance === 'string' ? parseFloat(row.Balance.replace(/[^0-9.-]+/g, '')) || 0 : row.Balance || 0
        }));

        console.log('Transformed data:', transformedData);
        setConvertedData(transformedData);
        onConvert(transformedData);
        toast.success('Conversion successful');
      } else {
        throw new Error('Invalid data format');
      }
    } catch (error) {
      console.error('Error converting PDF:', error);
      toast.error('Failed to convert file');
      setCurrentStep(1);
    } finally {
      setIsConverting(false);
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
      a.download = `financial_data.${format}`;
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

  const renderStep = () => {
    switch (currentStep) {
      case 0: // Upload
        return (
          <div
            {...getRootProps()}
            className="border-2 border-dashed border-gray-300 rounded-lg p-12 text-center cursor-pointer hover:border-blue-500 transition-colors"
          >
            <input {...getInputProps()} />
            <DocumentArrowUpIcon className="h-12 w-12 mx-auto text-gray-400" />
            <p className="mt-4 text-gray-600">
              Drag & drop a PDF file here, or click to select one
            </p>
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
                loading={
                  <div className="text-center py-12">
                    <ArrowPathIcon className="h-8 w-8 mx-auto animate-spin text-blue-500" />
                    <p className="mt-2 text-gray-600">Loading PDF...</p>
                  </div>
                }
                options={{
                  cMapUrl: 'cmaps/',
                  standardFontDataUrl: 'standard_fonts/',
                }}
              >
                <Page
                  pageNumber={currentPage}
                  width={600}
                  className="mx-auto"
                  renderTextLayer={true}
                  renderAnnotationLayer={true}
                />
              </Document>
              {numPages > 1 && (
                <div className="flex justify-center mt-4 space-x-2">
                  <button
                    onClick={() => setCurrentPage(Math.max(1, currentPage - 1))}
                    disabled={currentPage === 1}
                    className="px-3 py-1 bg-blue-500 text-white rounded disabled:bg-gray-300"
                  >
                    Previous
                  </button>
                  <span className="px-3 py-1">
                    Page {currentPage} of {numPages}
                  </span>
                  <button
                    onClick={() => setCurrentPage(Math.min(numPages, currentPage + 1))}
                    disabled={currentPage === numPages}
                    className="px-3 py-1 bg-blue-500 text-white rounded disabled:bg-gray-300"
                  >
                    Next
                  </button>
                </div>
              )}
            </div>
          </div>
        );

      case 2: // Convert & Download
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
              <div className="flex space-x-2">
                <button
                  onClick={() => handleExport('xlsx')}
                  className="px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700"
                >
                  Export XLSX
                </button>
                <button
                  onClick={() => handleExport('csv')}
                  className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
                >
                  Export CSV
                </button>
                <button
                  onClick={() => handleExport('xml')}
                  className="px-4 py-2 bg-purple-600 text-white rounded hover:bg-purple-700"
                >
                  Export XML
                </button>
                <button
                  onClick={onStartValidation}
                  className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
                >
                  Start Validation
                </button>
              </div>
            </div>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              <div className="bg-white rounded-lg shadow-lg p-4">
                <Document
                  file={pdfUrl}
                  onLoadSuccess={onDocumentLoadSuccess}
                  onLoadError={onDocumentLoadError}
                  loading={
                    <div className="text-center py-12">
                      <ArrowPathIcon className="h-8 w-8 mx-auto animate-spin text-blue-500" />
                      <p className="mt-2 text-gray-600">Loading PDF...</p>
                    </div>
                  }
                  options={{
                    cMapUrl: 'cmaps/',
                    standardFontDataUrl: 'standard_fonts/',
                  }}
                >
                  <Page
                    pageNumber={currentPage}
                    width={400}
                    renderTextLayer={true}
                    renderAnnotationLayer={true}
                  />
                </Document>
              </div>
              <div className="bg-white rounded-lg shadow-lg p-4 overflow-x-auto">
                {convertedData && convertedData.length > 0 ? (
                  <table className="min-w-full divide-y divide-gray-200 border border-gray-300">
                    <thead className="bg-gray-50">
                      <tr>
                        <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider border-b border-gray-300">Date</th>
                        <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider border-b border-gray-300">Voucher No</th>
                        <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider border-b border-gray-300">Ledger Name</th>
                        <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider border-b border-gray-300">Amount</th>
                        <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider border-b border-gray-300">Narration</th>
                        <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider border-b border-gray-300">Balance</th>
                      </tr>
                    </thead>
                    <tbody className="bg-white divide-y divide-gray-200">
                      {convertedData.map((row) => (
                        <tr key={row.id} className="hover:bg-gray-50">
                          <td className="px-4 py-2 text-sm border-r border-gray-300 whitespace-nowrap">{row.date}</td>
                          <td className="px-4 py-2 text-sm border-r border-gray-300 whitespace-nowrap">{row.voucherNo}</td>
                          <td className="px-4 py-2 text-sm border-r border-gray-300 whitespace-nowrap">{row.ledgerName}</td>
                          <td className="px-4 py-2 text-sm border-r border-gray-300 whitespace-nowrap">{row.amount}</td>
                          <td className="px-4 py-2 text-sm border-r border-gray-300 whitespace-nowrap">{row.narration}</td>
                          <td className="px-4 py-2 text-sm border-r border-gray-300 whitespace-nowrap">{row.balance}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                ) : (
                  <div className="text-center py-12 text-gray-500">
                    No data available
                  </div>
                )}
              </div>
            </div>
          </div>
        );

      case 3: // Validate
        return convertedData ? (
          <div className="space-y-4">
            <div className="flex justify-between items-center mb-4">
              <button
                onClick={goBack}
                className="flex items-center px-3 py-2 text-gray-600 hover:text-gray-900"
              >
                <ArrowLeftIcon className="h-5 w-5 mr-2" />
                Back
              </button>
            </div>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              <div className="bg-white rounded-lg shadow-lg p-4">
                <Document
                  file={pdfUrl}
                  onLoadSuccess={onDocumentLoadSuccess}
                  onLoadError={onDocumentLoadError}
                  loading={
                    <div className="text-center py-12">
                      <ArrowPathIcon className="h-8 w-8 mx-auto animate-spin text-blue-500" />
                      <p className="mt-2 text-gray-600">Loading PDF...</p>
                    </div>
                  }
                  options={{
                    cMapUrl: 'cmaps/',
                    standardFontDataUrl: 'standard_fonts/',
                  }}
                >
                  <Page
                    pageNumber={currentPage}
                    width={400}
                    renderTextLayer={true}
                    renderAnnotationLayer={true}
                  />
                </Document>
              </div>
              <div className="bg-white rounded-lg shadow-lg p-4">
                <FinancialTable
                  data={convertedData}
                  onDataChange={(newData) => setConvertedData(newData)}
                />
              </div>
            </div>
          </div>
        ) : null;

      default:
        return null;
    }
  };

  return (
    <div className="max-w-7xl mx-auto py-6 px-4 sm:px-6 lg:px-8">
      <WorkflowStepper steps={steps} currentStep={currentStep} />
      <div className="mt-8">{renderStep()}</div>
    </div>
  );
};

export default PDFConverter; 