import React, { useState, useEffect, useCallback, useRef } from 'react';
import axios from 'axios';
import { Document, Page } from 'react-pdf';
import { API_URL } from '../config';
import {
  ChevronLeftIcon,
  ChevronRightIcon,
  DocumentArrowDownIcon,
  ArrowRightIcon,
  TableCellsIcon,
  DocumentTextIcon,
  DocumentIcon,
  ArrowDownTrayIcon,
  ArrowLeftIcon,
  EyeIcon,
  RocketLaunchIcon,
} from '@heroicons/react/24/outline';
import toast from 'react-hot-toast';
import { useNavigate, useLocation } from 'react-router-dom';
import FinancialTable from './FinancialTable';
import { initPdfWorker, cleanupPdfWorker } from '../utils/pdfjs-config';
import { FinancialEntry } from '../types/financial';


interface ConversionResponse {
  rows: Record<string, any>[];
  headers: string[];
}

interface ViewComponentProps {
  onNext: (data: FinancialEntry[]) => void;
  onBack: () => void;
}

const ViewComponent: React.FC<ViewComponentProps> = ({ onNext, onBack }) => {
  const [files, setFiles] = useState<any[]>([]);
  const [selectedFile, setSelectedFile] = useState<string | null>(null);
  const [numPages, setNumPages] = useState<number | null>(null);
  const [pageNumber, setPageNumber] = useState(1);
  const [loading, setLoading] = useState(true);
  const [fileType, setFileType] = useState<'pdf' | 'image' | null>(null);
  const [fileUrl, setFileUrl] = useState<string | null>(null);
  const [convertedData, setConvertedData] = useState<FinancialEntry[]>([]);
  const [activeTab, setActiveTab] = useState<'document' | 'table'>('document');
  const [conversionLoading, setConversionLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const navigate = useNavigate();
  const location = useLocation();
  const currentBlobUrl = useRef<string | null>(null);
  const isMounted = useRef(true);
  const [scale, setScale] = useState<number>(1.5);
  const [isFirstConvert, setIsFirstConvert] = useState(true);


  // Initialize PDF worker when component mounts
  useEffect(() => {
    initPdfWorker();
    return () => {
      cleanupPdfWorker();
      if (currentBlobUrl.current) {
        URL.revokeObjectURL(currentBlobUrl.current);
        currentBlobUrl.current = null;
      }
    };
  }, []);



  const loadFile = useCallback(async (fileId: string) => {
    setLoading(true);
    setError(null);
    let retryCount = 0;
    const maxRetries = 3;
    const retryDelay = 1000; // 1 second

    const tryLoadFile = async (): Promise<void> => {
      try {
        const response = await axios.get(`${API_URL}/file/${fileId}`, {
          responseType: 'blob',
          headers: {
            'Accept': 'application/pdf,image/*'
          }
        });

        if (currentBlobUrl.current) {
          window.URL.revokeObjectURL(currentBlobUrl.current);
        }

        const type = response.headers['content-type'];
        const isPdf = type === 'application/pdf';
        setFileType(isPdf ? 'pdf' : 'image');

        const url = window.URL.createObjectURL(new Blob([response.data], { type }));
        currentBlobUrl.current = url;

        // Add a small delay to ensure the blob URL is ready
        await new Promise(resolve => setTimeout(resolve, 100));

        if (isMounted.current) {
          setFileUrl(url);
          setPageNumber(1);
          setLoading(false);
          setError(null);
        }

        // Start data conversion
        convertFile(fileId).catch(console.error);

      } catch (error: any) {
        console.error('Error loading file:', error);
        
        if (error.response?.status === 404) {
          // If file is not found and we haven't exceeded retries
          if (retryCount < maxRetries) {
            retryCount++;
            console.log(`Retrying (${retryCount}/${maxRetries})...`);
            await new Promise(resolve => setTimeout(resolve, retryDelay));
            return tryLoadFile();
          }
        }

        if (isMounted.current) {
          const errorMessage = error.response?.status === 404
            ? 'File not found. The file may still be processing.'
            : error.response?.data?.detail || error.message || 'Failed to load file';
          
          setError(errorMessage);
          toast.error(errorMessage);
          
          if (error.response?.status === 404) {
            // Only navigate away on 404 after retries
            navigate('/', { replace: true });
          }
        }
      } finally {
        if (isMounted.current) {
          setLoading(false);
        }
      }
    };

    await tryLoadFile();
  }, [navigate]);

  useEffect(() => {
    isMounted.current = true;
    
    const state = location.state as { fileId?: string, fileName?: string, preview?: string };
    
    if (!state?.fileId) {
      const error = 'No file selected';
      console.error(error);
      setError(error);
      toast.error(error);
      navigate('/', { replace: true });
      return;
    }

    setSelectedFile(state.fileId);
    
    // Add a small delay to ensure the file is saved on the server
    const timer = setTimeout(() => {
      if (isMounted.current) {
        loadFile(state.fileId!);
      }
    }, 500); // Reduced to 500ms since we have retry logic

    return () => {
      isMounted.current = false;
      clearTimeout(timer);
      
      if (currentBlobUrl.current) {
        window.URL.revokeObjectURL(currentBlobUrl.current);
        currentBlobUrl.current = null;
      }
    };
  }, [location.state, navigate, loadFile]);

  const handleFileSelect = async (fileId: string) => {
    setSelectedFile(fileId);
    setLoading(true);
    try {
      const response = await axios.get(`${API_URL}/file/${fileId}`, {
        responseType: 'blob'
      });
      
      const type = response.headers['content-type'];
      const isPdf = type === 'application/pdf';
      setFileType(isPdf ? 'pdf' : 'image');
      
      const url = window.URL.createObjectURL(new Blob([response.data]));
      setFileUrl(url);
      setPageNumber(1);

      // Load converted data if it exists
      convertFile(fileId);
    } catch (error) {
      console.error('Error loading file:', error);
      toast.error('Failed to load file');
    } finally {
      setLoading(false);
    }
  };

  const convertFile = async (fileId: string, conversionToast?: string) => {
    setConversionLoading(true);
    try {
      // First, try to get the data with conversion
      const response = await axios.get(`${API_URL}/file/${fileId}?convert=true`);
      
      if (response.data?.success && response.data?.data?.rows) {
        const rows = response.data.data.rows;
        
        // Convert the data to our format
        const convertedRows = rows.map((row: any, index: number) => ({
          id: `row-${index}`,
          date: row.date || row.DATE || '',
          voucherNo: row.voucherNo || row['VOUCHER NO'] || row.voucher_no || '',
          ledgerName: row.ledgerName || row['LEDGER NAME'] || row.ledger_name || '',
          amount: typeof row.amount === 'number' ? row.amount : 
                 typeof row.AMOUNT === 'number' ? row.AMOUNT :
                 parseFloat(String(row.amount || row.AMOUNT || '0').replace(/[^\d.-]/g, '')) || 0,
          narration: row.narration || row.NARRATION || row.description || '',
          balance: typeof row.balance === 'number' ? row.balance :
                  typeof row.BALANCE === 'number' ? row.BALANCE :
                  parseFloat(String(row.balance || row.BALANCE || '0').replace(/[^\d.-]/g, '')) || 0
        }));

        setConvertedData(convertedRows);
        setActiveTab('table');
        

        
        if (conversionToast) {
          toast.success('Document converted successfully', {
            id: conversionToast
          });
        }
      } else {
        throw new Error('No data received from conversion');
      }
    } catch (error: any) {
      console.error('Error converting file:', error);
      const errorMessage = error.response?.data?.detail || error.message || 'Failed to convert file';
      
      // Log the error
      await logAction('error', `Failed to convert file ${fileId}`, {
        error: errorMessage,
        fileId
      });
      
      if (conversionToast) {
        toast.error(errorMessage, {
          id: conversionToast
        });
      } else {
        toast.error(errorMessage);
      }
      throw error;
    } finally {
      setConversionLoading(false);
    }
  };

  const onDocumentLoadSuccess = ({ numPages }: { numPages: number }) => {
    setNumPages(numPages);
    setLoading(false);
  };

  const onDocumentLoadError = (error: Error) => {
    console.error('Error loading PDF:', error);
    setError(error.message);
    setLoading(false);
    toast.error(`Error loading PDF: ${error.message}`);
  };

  const changePage = (offset: number) => {
    if (!numPages) return;
    const newPage = pageNumber + offset;
    if (newPage >= 1 && newPage <= numPages) {
      setPageNumber(newPage);
    }
  };

  const handleProceed = () => {
    if (convertedData.length > 0) {
      onNext(convertedData);
    } else {
      toast.error('Please convert the document first');
    }
  };

  const handleBack = () => {
    onBack();
  };

  const handleConvert = async () => {
    if (!selectedFile) {
      toast.error('No file selected');
      return;
    }

    if (isFirstConvert) {
      const conversionToast = toast.loading('Converting document...');
      await convertFile(selectedFile, conversionToast);
      setIsFirstConvert(false);
      setActiveTab('table');
    } else {
      navigate('/convert', { 
        state: { 
          fileId: selectedFile,
          data: convertedData 
        }
      });
    }
  };

  const handleZoomIn = () => {
    setScale(prev => Math.min(prev + 0.2, 3));
  };

  const handleZoomOut = () => {
    setScale(prev => Math.max(prev - 0.2, 0.5));
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[400px] text-red-500">
        <p className="text-lg font-medium mb-4">Error loading document</p>
        <p className="text-sm">{error}</p>
      </div>
    );
  }

  console.log('Rendering preview section:', {
    fileUrl,
    fileType,
    selectedFile,
    convertedData: convertedData.length
  });

  return (
    <div className="container mx-auto p-4 space-y-6">
      <div className="flex justify-between items-center mb-4">
        <button
          onClick={handleBack}
          className="flex items-center text-gray-600 hover:text-gray-900"
        >
          <ArrowLeftIcon className="h-5 w-5 mr-2" />
          Back
        </button>
        <div className="flex items-center space-x-4">
          <div className="flex space-x-4">
            <button
              onClick={() => setActiveTab('document')}
              className={`flex items-center px-4 py-2 rounded-lg ${
                activeTab === 'document' ? 'bg-blue-100 text-blue-700' : 'text-gray-600'
              }`}
            >
              <DocumentIcon className="h-5 w-5 mr-2" />
              Document
            </button>
            <button
              onClick={() => setActiveTab('table')}
              className={`flex items-center px-4 py-2 rounded-lg ${
                activeTab === 'table' ? 'bg-blue-100 text-blue-700' : 'text-gray-600'
              }`}
            >
              <TableCellsIcon className="h-5 w-5 mr-2" />
              Table
            </button>
          </div>
          <button
            onClick={handleConvert}
            className="flex items-center px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
          >
            {isFirstConvert ? 'Convert to Table' : 'Proceed to Convert'}
            <ArrowRightIcon className="h-5 w-5 ml-2" />
          </button>
        </div>
      </div>



      {activeTab === 'document' && (
        <div className="flex flex-col items-center space-y-4">
          {/* Document viewer controls */}
          <div className="flex items-center space-x-4 mb-4">
            <button
              onClick={handleZoomOut}
              className="p-2 text-gray-600 hover:text-gray-900 disabled:text-gray-300"
            >
              -
            </button>
            <span>{Math.round(scale * 100)}%</span>
            <button
              onClick={handleZoomIn}
              className="p-2 text-gray-600 hover:text-gray-900 disabled:text-gray-300"
            >
              +
            </button>
          </div>

          {/* PDF/Image viewer */}
          <div className="w-full max-w-4xl mx-auto bg-white rounded-lg shadow-lg overflow-hidden">
            {loading ? (
              <div className="flex items-center justify-center h-96">
                <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
              </div>
            ) : error ? (
              <div className="flex items-center justify-center h-96 text-red-600">
                {error}
              </div>
            ) : fileType === 'pdf' && fileUrl ? (
              <div className="flex flex-col items-center">
                <Document
                  file={fileUrl}
                  onLoadSuccess={onDocumentLoadSuccess}
                  onLoadError={onDocumentLoadError}
                  loading={<div>Loading PDF...</div>}
                  error={<div>Error loading PDF. Please try again.</div>}
                >
                  <Page
                    pageNumber={pageNumber}
                    scale={scale}
                    loading={<div>Loading page {pageNumber}...</div>}
                    error={<div>Error loading page {pageNumber}. Please try again.</div>}
                  />
                </Document>
                {numPages && numPages > 1 && (
                  <div className="flex items-center space-x-4 mt-4">
                    <button
                      onClick={() => changePage(-1)}
                      disabled={pageNumber <= 1}
                      className="p-2 text-gray-600 hover:text-gray-900 disabled:text-gray-300"
                    >
                      <ChevronLeftIcon className="h-5 w-5" />
                    </button>
                    <span>
                      Page {pageNumber} of {numPages}
                    </span>
                    <button
                      onClick={() => changePage(1)}
                      disabled={pageNumber >= (numPages || 1)}
                      className="p-2 text-gray-600 hover:text-gray-900 disabled:text-gray-300"
                    >
                      <ChevronRightIcon className="h-5 w-5" />
                    </button>
                  </div>
                )}
              </div>
            ) : fileType === 'image' && fileUrl ? (
              <div className="flex-1 overflow-auto flex justify-center items-start p-4 bg-gray-100">
                <img
                  src={fileUrl}
                  alt="Document preview"
                  className="max-w-full h-auto"
                  onLoad={() => setLoading(false)}
                  onError={() => {
                    setError('Failed to load image');
                    setLoading(false);
                  }}
                />
              </div>
            ) : null}
          </div>
        </div>
      )}

      {activeTab === 'table' && (
        <div className="w-full">
          <FinancialTable 
            data={convertedData}
            readOnly={true}
          />
        </div>
      )}

      {/* Fast Navigation Section */}
      <div className="mt-8 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-200 dark:border-blue-800">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-lg font-medium text-blue-900 dark:text-blue-100 flex items-center">
              <RocketLaunchIcon className="h-5 w-5 mr-2" />
              Quick Convert
            </h3>
            <p className="text-sm text-blue-700 dark:text-blue-300 mt-1">
              Skip heavy processing and go directly to convert with cached data
            </p>
          </div>
          <button
            onClick={() => {
              // Use cached data for fast navigation
              const cachedData = convertedData.length > 0 ? convertedData : [];
              navigate('/convert', {
                state: {
                  fileId: selectedFile,
                  data: cachedData,
                  fromPreview: true,
                  skipProcessing: true
                }
              });
            }}
            disabled={!selectedFile}
            className="flex items-center px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            <ArrowRightIcon className="h-4 w-4 mr-2" />
            Quick Convert
          </button>
        </div>
      </div>
    </div>
  );
};

export default ViewComponent; 