import React, { useState, useEffect, useCallback, useRef } from 'react';
import axios from 'axios';
import { Document, Page, pdfjs } from 'react-pdf';
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
    };
  }, []);

  const loadFile = useCallback(async (fileId: string) => {
    if (!isMounted.current) return;
    
    setLoading(true);
    setError(null);
    setFileUrl(null); // Clear previous file URL

    // Cleanup previous blob URL if it exists
    if (currentBlobUrl.current) {
      window.URL.revokeObjectURL(currentBlobUrl.current);
      currentBlobUrl.current = null;
    }

    try {
      console.log('Fetching file with ID:', fileId);
      const response = await axios.get(`${API_URL}/file/${fileId}`, {
        responseType: 'blob'
      });
      
      if (!isMounted.current) return;

      const type = response.headers['content-type'];
      const isPdf = type === 'application/pdf';
      setFileType(isPdf ? 'pdf' : 'image');
      
      const blob = new Blob([response.data], { type });
      const url = window.URL.createObjectURL(blob);
      currentBlobUrl.current = url;
      
      // Small delay to ensure cleanup is complete
      setTimeout(() => {
        if (isMounted.current) {
          setFileUrl(url);
          setPageNumber(1);
        }
      }, 100);
    } catch (error) {
      if (!isMounted.current) return;
      
      console.error('Error loading file:', error);
      const errorMessage = error instanceof Error ? error.message : 'Failed to load file';
      setError(errorMessage);
      toast.error(errorMessage);
      navigate('/', { replace: true });
    } finally {
      if (isMounted.current) {
        setLoading(false);
      }
    }
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
    }, 1000); // 1 second delay

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
      // First, try to get the data
      const response = await axios.get(`${API_URL}/get-data/${fileId}`);
      
      if (response.data && (response.data.rows || response.data.data?.rows)) {
        const rows = response.data.rows || response.data.data.rows;
        
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
      
      if (conversionToast) {
        toast.error(errorMessage, {
          id: conversionToast
        });
      } else {
        toast.error(errorMessage);
      }
      throw error; // Re-throw to handle in the calling function
    } finally {
      setConversionLoading(false);
    }
  };

  const onDocumentLoadSuccess = useCallback(({ numPages }: { numPages: number }) => {
    console.log('PDF loaded successfully with', numPages, 'pages');
    setNumPages(numPages);
    setLoading(false);
    setError(null);
  }, []);

  const onDocumentLoadError = useCallback((error: Error) => {
    console.error('Error loading PDF:', error);
    setError(error.message);
    setLoading(false);
    toast.error('Failed to load PDF document');
  }, []);

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
                  onLoadSuccess={({ numPages }) => setNumPages(numPages)}
                  error={<div>Failed to load PDF</div>}
                >
                  <Page
                    pageNumber={pageNumber}
                    scale={scale}
                    renderAnnotationLayer={false}
                    renderTextLayer={false}
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
              <img src={fileUrl} alt="Preview" className="max-w-full h-auto" />
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
    </div>
  );
};

export default ViewComponent; 