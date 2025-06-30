import React, { useState, useEffect, useCallback } from 'react';
import axios from 'axios';
import { API_URL } from '../config';
import {
  DocumentArrowDownIcon,
  ArrowPathIcon,
  CheckCircleIcon,
  XCircleIcon,
  ArrowRightIcon,
  TableCellsIcon,
  DocumentIcon,
  ArrowLeftIcon,
  ChevronLeftIcon,
  ChevronRightIcon,
} from '@heroicons/react/24/outline';
import toast from 'react-hot-toast';
import { useNavigate, useLocation, useParams } from 'react-router-dom';
import { FinancialEntry } from '../types/financial';
import { useDynamicFilename } from '../hooks/useDynamicFilename';

interface ConversionStatus {
  status: 'pending' | 'processing' | 'completed' | 'failed';
  progress?: number;
  error?: string;
}

interface ExportFormat {
  id: string;
  name: string;
  extension: string;
  icon: React.ComponentType<any>;
  description: string;
}

const EXPORT_FORMATS: ExportFormat[] = [
  {
    id: 'xlsx',
    name: 'Excel Spreadsheet',
    extension: 'xlsx',
    icon: TableCellsIcon,
    description: 'Export as Microsoft Excel file'
  },
  {
    id: 'csv',
    name: 'CSV File',
    extension: 'csv',
    icon: DocumentArrowDownIcon,
    description: 'Export as CSV for universal compatibility'
  },
  {
    id: 'xml',
    name: 'XML Document',
    extension: 'xml',
    icon: DocumentIcon,
    description: 'Export as XML for structured data'
  }
];

const ConvertComponent: React.FC = () => {
  const { fileId } = useParams<{ fileId: string }>();
  const [loading, setLoading] = useState(false);
  const [convertedFormats, setConvertedFormats] = useState<Set<string>>(new Set());
  const [tableData, setTableData] = useState<any[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [conversionStatus, setConversionStatus] = useState<ConversionStatus>({ status: 'pending' });
  const navigate = useNavigate();
  const location = useLocation();
  const [showPasswordModal, setShowPasswordModal] = useState(false);
  const [password, setPassword] = useState('');
  const [passwordError, setPasswordError] = useState('');
  
  // Pagination state
  const [currentPage, setCurrentPage] = useState(1);
  const [itemsPerPage] = useState(10);

  const { generateFilename } = useDynamicFilename();

  const [useHybridParser, setUseHybridParser] = useState<boolean>(true);

  // Calculate pagination
  const indexOfLastItem = currentPage * itemsPerPage;
  const indexOfFirstItem = indexOfLastItem - itemsPerPage;
  const currentItems = tableData.slice(indexOfFirstItem, indexOfLastItem);
  const totalPages = Math.ceil(tableData.length / itemsPerPage);

  useEffect(() => {
    if (!fileId) {
      toast.error('No file ID provided');
      navigate('/', { replace: true });
      return;
    }
    
    // Try to load existing converted data
    loadConvertedData();
  }, [fileId, navigate]);

  const loadConvertedData = async () => {
    try {
      setLoading(true);
      const response = await axios.get(`${API_URL}/file/${fileId}?convert=true`);
      
      if (response.data?.data?.rows) {
        setTableData(response.data.data.rows);
        setConversionStatus({ status: 'completed' });
        toast.success('Data loaded successfully');
      } else if (response.data?.rows) {
        setTableData(response.data.rows);
        setConversionStatus({ status: 'completed' });
        toast.success('Data loaded successfully');
      }
    } catch (error: any) {
      console.error('Error loading data:', error);
      if (error.response?.status === 404) {
        toast.error('File not found. Please upload the file again.');
        navigate('/');
      } else {
        setError('Failed to load data. You may need to convert the file first.');
      }
    } finally {
      setLoading(false);
      }
  };

  const parseAmount = (value: any): number => {
    if (!value) return 0;
    
    // Convert to string and handle different formats
    let str_value = String(value).trim();
    
    if (!str_value || str_value === '-' || str_value.toLowerCase() === 'nil') {
      return 0;
    }
    
    // Handle negative values in parentheses
    let isNegative = false;
    if (str_value.startsWith('(') && str_value.endsWith(')')) {
      isNegative = true;
      str_value = str_value.slice(1, -1);
    } else if (str_value.startsWith('-')) {
      isNegative = true;
    }
    
    // Remove currency symbols and formatting
    const cleaned = str_value
      .replace(/[₹$€£¥Rs.,\s]/g, '')
      .replace(/[^\d.-]/g, '')
      .trim();
    
    if (!cleaned) return 0;
    
    try {
      const parsed = parseFloat(cleaned);
      return isNaN(parsed) ? 0 : (isNegative ? -parsed : parsed);
    } catch {
      return 0;
    }
  };

  const formatValue = (value: any, header: string) => {
    // Detect amount columns
    const amountKeywords = [
      'amount', 'amt', 'balance', 'total', 'sum', 'value', 'money',
      'debit', 'credit', 'withdrawal', 'deposit', 'rupees', 'rs', 'inr',
      'price', 'cost', 'fee', 'charge'
    ];
    
    const isAmountColumn = amountKeywords.some(keyword => 
      header.toLowerCase().includes(keyword)
    );
    
    if (isAmountColumn) {
      const numValue = parseAmount(value);
      if (numValue === 0 && (!value || value === '' || value === '0')) {
        return '₹0.00';
      }
      return '₹' + numValue.toLocaleString('en-IN', {
        maximumFractionDigits: 2,
        minimumFractionDigits: 2
      });
    }
    return value || '';
  };

  const getCellClassName = (value: any, header: string) => {
    const baseClasses = 'px-6 py-4 whitespace-nowrap text-sm';
    
    // Detect amount columns
    const amountKeywords = [
      'amount', 'amt', 'balance', 'total', 'sum', 'value', 'money',
      'debit', 'credit', 'withdrawal', 'deposit', 'rupees', 'rs', 'inr',
      'price', 'cost', 'fee', 'charge'
    ];
    
    const isAmountColumn = amountKeywords.some(keyword => 
      header.toLowerCase().includes(keyword)
    );
    
    if (isAmountColumn) {
      const numValue = parseAmount(value);
      if (numValue === 0 && (!value || value === '' || value === '0')) {
        return `${baseClasses} text-red-600 bg-red-50 font-mono text-right`;
      }
      return `${baseClasses} text-green-600 bg-green-50 font-mono text-right`;
    }
    
    if (!value || value === '') {
      return `${baseClasses} text-gray-400`;
    }
    
    return `${baseClasses} text-gray-900`;
  };

  const handleConvert = async () => {
    if (!fileId) {
      toast.error('No file selected');
      return;
    }

    setLoading(true);
    setError(null);
    setConversionStatus({ status: 'processing', progress: 0 });
    const toastId = toast.loading('Converting file...');
    
    try {
      // Real-time progress tracking
      const progressInterval = setInterval(async () => {
        try {
          const progressResponse = await axios.get(`${API_URL}/convert-progress/${fileId}`);
          const progressData = progressResponse.data;
          
          setConversionStatus({
            status: progressData.stage === 'error' ? 'failed' : 
                   progressData.stage === 'completed' ? 'completed' : 'processing',
            progress: progressData.progress || 0,
            error: progressData.stage === 'error' ? progressData.message : undefined
          });
          
          // Update toast message with current stage
          if (progressData.message) {
            toast.loading(progressData.message, { id: toastId });
          }
          
          // Clear interval when done
          if (progressData.stage === 'completed' || progressData.stage === 'error') {
            clearInterval(progressInterval);
          }
        } catch (e) {
          // Ignore progress polling errors
        }
      }, 1000);

      // Start conversion
      await new Promise(resolve => setTimeout(resolve, 100));

      const response = await axios.post(
        `${API_URL}/convert/${fileId}`,
        { use_hybrid_parser: useHybridParser },
        {
          headers: { 'Content-Type': 'application/json' },
          onUploadProgress: (progressEvent) => {
            // Handle progress event
          },
          timeout: 180000 // 3 minute timeout
        }
      );
        
      clearInterval(progressInterval);

      if (response.data?.rows) {
        setTableData(response.data.rows);
        setConvertedFormats(new Set(['xlsx', 'csv', 'xml']));
        setConversionStatus({ status: 'completed', progress: 100 });
        toast.success('File converted successfully!', { id: toastId });
      } else {
        throw new Error('No data received from conversion');
      }

    } catch (error: any) {
      console.error('Conversion error:', error);
      setConversionStatus({ status: 'failed', error: error.message });
      
      let errorMessage = 'Failed to convert file';
      
      if (error.code === 'ECONNABORTED') {
        errorMessage = 'Request timed out. Please try again.';
      } else if (error.response?.status === 404) {
        errorMessage = 'File not found. Please upload again.';
      } else if (error.response?.status === 401 && error.response?.data?.requires_password) {
        setShowPasswordModal(true);
        errorMessage = 'PDF is password protected.';
      } else if (error.response?.data?.detail) {
        errorMessage = error.response.data.detail;
      }
      
      setError(errorMessage);
      toast.error(errorMessage, { id: toastId });
    } finally {
      setLoading(false);
    }
  };

  const handleExport = async (format: string) => {
    if (!fileId) {
      toast.error('No file selected');
      return;
    }

    if (tableData.length === 0) {
      toast.error('No data to export. Please convert the file first.');
      return;
    }

    setLoading(true);
    const toastId = toast.loading(`Downloading ${format.toUpperCase()}...`);
    
    try {
      // Generate dynamic filename
      const dynamicFilename = await generateFilename(fileId, format, 'en');
      
      const response = await axios.get(`${API_URL}/api/download/${fileId}/${format}`, {
            responseType: 'blob',
        timeout: 60000, // 1 minute timeout
            headers: {
              'Accept': format === 'xlsx' 
                ? 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                : format === 'csv'
                ? 'text/csv'
                : 'application/xml'
            }
          });

      if (!response.data || response.data.size === 0) {
        throw new Error('Received empty response from server');
      }

      // Create blob and download
      const blob = new Blob([response.data], { 
        type: format === 'xlsx' 
          ? 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
          : format === 'csv'
          ? 'text/csv'
          : 'application/xml'
      });

      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.style.display = 'none';
      a.href = url;
      a.download = dynamicFilename || `converted_file.${format}`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);

      setConvertedFormats(prev => new Set([...prev, format]));
      toast.success(`${format.toUpperCase()} downloaded successfully!`, { id: toastId });
    } catch (error: any) {
      console.error('Export error:', error);
      let errorMessage = error.response?.data?.detail || error.message || 'Failed to export file';
      
      // Handle validation error
      if (error.response?.status === 400 && errorMessage.includes('validate')) {
        toast.error(errorMessage, { id: toastId });
        // Navigate to validation page
        navigate('/validate', { 
          state: { 
            fileId: fileId,
            data: tableData,
            convertedFormats: Array.from(convertedFormats)
          }
        });
        return;
      }
      
      toast.error(`Export failed: ${errorMessage}`, { id: toastId });
    } finally {
      setLoading(false);
    }
  };

  const handleBack = () => {
    navigate(-1);
  };

  const handlePasswordSubmit = async () => {
    setPasswordError('');
    if (!password.trim()) {
      setPasswordError('Please enter a password');
      return;
    }

    try {
      await handleConvert();
      setShowPasswordModal(false);
      setPassword('');
    } catch (error: any) {
      console.error('Password validation error:', error);
      if (error.response?.status === 401) {
        setPasswordError('Incorrect password. Please try again.');
      } else {
        setPasswordError('Error validating password. Please try again.');
      }
    }
  };

  const handleValidate = () => {
    if (tableData.length === 0) {
      toast.error('No data to validate. Please convert the file first.');
      return;
    }

        navigate('/validate', { 
          state: { 
        fileId: fileId,
            data: tableData,
        convertedFormats: Array.from(convertedFormats)
      }
    });
  };

  const renderPagination = () => {
    if (totalPages <= 1) return null;

    return (
      <div className="flex items-center justify-between px-6 py-3 bg-gray-50 border-t border-gray-200">
        <div className="flex-1 flex justify-between sm:hidden">
          <button
            onClick={() => setCurrentPage(prev => Math.max(prev - 1, 1))}
            disabled={currentPage === 1}
            className="relative inline-flex items-center px-4 py-2 border border-gray-300 text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Previous
          </button>
          <button
            onClick={() => setCurrentPage(prev => Math.min(prev + 1, totalPages))}
            disabled={currentPage === totalPages}
            className="ml-3 relative inline-flex items-center px-4 py-2 border border-gray-300 text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Next
          </button>
        </div>
        <div className="hidden sm:flex-1 sm:flex sm:items-center sm:justify-between">
          <div>
            <p className="text-sm text-gray-700">
              Showing <span className="font-medium">{indexOfFirstItem + 1}</span> to{' '}
              <span className="font-medium">{Math.min(indexOfLastItem, tableData.length)}</span> of{' '}
              <span className="font-medium">{tableData.length}</span> results
            </p>
          </div>
          <div>
            <nav className="relative z-0 inline-flex rounded-md shadow-sm -space-x-px" aria-label="Pagination">
              <button
                onClick={() => setCurrentPage(prev => Math.max(prev - 1, 1))}
                disabled={currentPage === 1}
                className="relative inline-flex items-center px-2 py-2 rounded-l-md border border-gray-300 bg-white text-sm font-medium text-gray-500 hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <ChevronLeftIcon className="h-5 w-5" />
              </button>
              
              {/* Page numbers */}
              {[...Array(totalPages)].map((_, index) => {
                const pageNumber = index + 1;
                if (pageNumber === 1 || pageNumber === totalPages || 
                    (pageNumber >= currentPage - 1 && pageNumber <= currentPage + 1)) {
                  return (
                    <button
                      key={pageNumber}
                      onClick={() => setCurrentPage(pageNumber)}
                      className={`relative inline-flex items-center px-4 py-2 border text-sm font-medium ${
                        currentPage === pageNumber
                          ? 'z-10 bg-blue-50 border-blue-500 text-blue-600'
                          : 'bg-white border-gray-300 text-gray-500 hover:bg-gray-50'
                      }`}
                    >
                      {pageNumber}
                    </button>
                  );
                } else if (pageNumber === currentPage - 2 || pageNumber === currentPage + 2) {
                  return <span key={pageNumber} className="relative inline-flex items-center px-4 py-2 border border-gray-300 bg-white text-sm font-medium text-gray-700">...</span>;
                }
                return null;
              })}
              
              <button
                onClick={() => setCurrentPage(prev => Math.min(prev + 1, totalPages))}
                disabled={currentPage === totalPages}
                className="relative inline-flex items-center px-2 py-2 rounded-r-md border border-gray-300 bg-white text-sm font-medium text-gray-500 hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <ChevronRightIcon className="h-5 w-5" />
              </button>
            </nav>
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="container mx-auto p-6 space-y-8">
      <div className="flex justify-between items-center">
        <button
          onClick={handleBack}
          className="flex items-center text-gray-600 hover:text-gray-900"
        >
          <ArrowLeftIcon className="h-5 w-5 mr-2" />
          Back
        </button>
        <h2 className="text-2xl font-semibold text-gray-900">Convert Document</h2>
        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-2">
            <span className="text-sm font-medium text-gray-700">Smart Extraction</span>
            <label htmlFor="hybrid-parser-toggle" className="relative inline-flex items-center cursor-pointer">
              <input 
                type="checkbox" 
                id="hybrid-parser-toggle" 
                className="sr-only peer"
                checked={useHybridParser}
                onChange={() => setUseHybridParser(!useHybridParser)}
              />
              <div className="w-11 h-6 bg-gray-200 rounded-full peer peer-focus:ring-4 peer-focus:ring-blue-300 dark:peer-focus:ring-blue-800 dark:bg-gray-700 peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-0.5 after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all dark:border-gray-600 peer-checked:bg-blue-600"></div>
            </label>
          </div>
          <div className="flex space-x-4">
            {tableData.length === 0 ? (
        <button
                onClick={handleConvert}
                disabled={loading}
                className="flex items-center px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-400"
        >
          {loading ? (
            <ArrowPathIcon className="h-5 w-5 animate-spin" />
          ) : (
                  'Convert File'
          )}
        </button>
            ) : (
              <button
                onClick={handleValidate}
                className="flex items-center px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700"
                disabled={loading}
              >
                Proceed to Validate
                <ArrowRightIcon className="h-5 w-5 ml-2" />
              </button>
            )}
          </div>
        </div>
      </div>

      {/* Password Modal */}
      {showPasswordModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white p-6 rounded-lg shadow-xl max-w-md w-full">
            <h3 className="text-lg font-semibold mb-4">PDF Password Required</h3>
            <form onSubmit={async (e) => {
              e.preventDefault();
              await handlePasswordSubmit();
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
                  }}
                  className="px-4 py-2 text-gray-600 hover:text-gray-800"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
                >
                  Submit
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* Progress Display */}
      {conversionStatus.status === 'processing' && (
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-blue-900">Processing File</h3>
            <span className="text-blue-600 font-medium">{conversionStatus.progress || 0}%</span>
          </div>
          
          {/* Progress Bar */}
          <div className="w-full bg-blue-200 rounded-full h-3 mb-4">
            <div 
              className="bg-blue-600 h-3 rounded-full transition-all duration-300 ease-out"
              style={{ width: `${conversionStatus.progress || 0}%` }}
            ></div>
          </div>
          
          {/* Progress Message */}
          {conversionStatus.error && (
            <p className="text-red-600 text-sm">
              {conversionStatus.error}
            </p>
          )}
          
          {/* Processing Animation */}
          <div className="flex items-center justify-center space-x-2">
            <ArrowPathIcon className="h-5 w-5 animate-spin text-blue-600" />
            <span className="text-blue-700">Processing your document...</span>
          </div>
        </div>
      )}

      {/* Error Display */}
      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <div className="flex items-center">
            <XCircleIcon className="h-5 w-5 text-red-500 mr-2" />
            <span className="text-red-700">{error}</span>
          </div>
        </div>
      )}

      {/* Export Formats */}
      {tableData.length > 0 && (
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {EXPORT_FORMATS.map((format) => (
          <div
            key={format.id}
            className="bg-white p-6 rounded-lg shadow-md border border-gray-200 hover:border-blue-500 transition-colors"
          >
            <div className="flex items-start justify-between">
              <div className="flex-1">
                <format.icon className="h-8 w-8 text-blue-600 mb-4" />
                <h3 className="text-lg font-semibold text-gray-900 mb-2">{format.name}</h3>
                <p className="text-gray-600 mb-4">{format.description}</p>
              </div>
              {convertedFormats.has(format.id) && (
                <CheckCircleIcon className="h-6 w-6 text-green-500" />
              )}
            </div>
            <button
              onClick={() => handleExport(format.id)}
              disabled={loading}
              className="w-full flex items-center justify-center px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-400"
            >
              {loading ? (
                <ArrowPathIcon className="h-5 w-5 animate-spin" />
              ) : (
                <>
                  <DocumentArrowDownIcon className="h-5 w-5 mr-2" />
                    Download {format.extension.toUpperCase()}
                </>
              )}
            </button>
          </div>
        ))}
      </div>
      )}

      {/* Data Preview with Pagination */}
      {tableData.length > 0 && (
        <div className="mt-8">
          <div className="flex justify-between items-center mb-4">
            <h3 className="text-lg font-semibold text-gray-900">Preview Data</h3>
            <span className="text-sm text-gray-600">
              {tableData.length} rows total
            </span>
          </div>
          <div className="bg-white rounded-lg shadow overflow-hidden">
            <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  {Object.keys(tableData[0]).map((header) => (
                    <th
                      key={header}
                      className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider"
                    >
                      {header}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                  {currentItems.map((row, index) => (
                    <tr key={indexOfFirstItem + index} className="hover:bg-gray-50">
                      {Object.entries(row).map(([header, value]) => (
                        <td key={header} className={getCellClassName(value, header)}>
                          {formatValue(value, header)}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
              </div>
            {renderPagination()}
          </div>
        </div>
      )}
    </div>
  );
};

export default ConvertComponent; 