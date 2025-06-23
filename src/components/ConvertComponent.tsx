import React, { useState, useEffect } from 'react';
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
} from '@heroicons/react/24/outline';
import toast from 'react-hot-toast';
import { useNavigate, useLocation } from 'react-router-dom';
import { FinancialEntry } from '../types/financial';

interface ConversionStatus {
  status: 'pending' | 'processing' | 'completed' | 'failed';
  progress?: number;
  error?: string;
}

interface ExportFormat {
  id: string;
  name: string;
  extension: string;
  icon: React.ForwardRefExoticComponent<any>;
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
  const [currentFile, setCurrentFile] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [convertedFormats, setConvertedFormats] = useState<Set<string>>(new Set());
  const [tableData, setTableData] = useState<FinancialEntry[]>([]);
  const [error, setError] = useState<string | null>(null);
  const navigate = useNavigate();
  const location = useLocation();
  const [showPasswordModal, setShowPasswordModal] = useState(false);
  const [password, setPassword] = useState('');
  const [passwordError, setPasswordError] = useState('');

  useEffect(() => {
    const state = location.state as { 
      fileId?: string, 
      data?: FinancialEntry[], 
      fromPreview?: boolean,
      skipProcessing?: boolean 
    };
    
    if (!state?.fileId) {
      toast.error('No file selected');
      navigate('/', { replace: true });
      return;
    }
    
    setCurrentFile({
      id: state.fileId,
      name: 'Uploaded File',
    });

    if (state.data) {
      setTableData(state.data);
      
      // If coming from preview with skip processing flag, mark as already processed
      if (state.fromPreview && state.skipProcessing) {
        setConvertedFormats(new Set(['xlsx', 'csv', 'xml'])); // Mark as converted
        toast.success('Using cached data for faster processing!');
      }
    }
  }, [location.state, navigate]);

  const handleExport = async (format: string) => {
    if (!currentFile?.id) {
      toast.error('No file selected');
      return;
    }

    const toastId = toast.loading(`Converting to ${format.toUpperCase()}...`);
    setLoading(true);
    setError(null);
    
    try {
      // Check if we already have converted data and can skip API conversion
      const hasData = tableData.length > 0;
      const isAlreadyConverted = convertedFormats.has(format);
      
      if (!hasData || !isAlreadyConverted) {
        // Only call convert API if we don't have data or format not converted
        const convertResponse = await axios.post(`${API_URL}/convert/${currentFile.id}`);
        
        // Update table data if available
        if (convertResponse.data?.data?.rows) {
          setTableData(convertResponse.data.data.rows);
        }
      }

      // Always try to download the converted file
      const downloadResponse = await axios.get(`${API_URL}/api/convert/${currentFile.id}/${format}`, {
        responseType: 'blob',
        headers: {
          'Accept': format === 'xlsx' 
            ? 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            : format === 'csv'
            ? 'text/csv'
            : 'application/xml'
        }
      });

      // Check if we got a valid blob response
      if (!downloadResponse.data || downloadResponse.data.size === 0) {
        throw new Error('Received empty response from server');
      }

      // Update the list of converted formats
      setConvertedFormats(prev => new Set([...prev, format]));
      
      // Create and trigger download
      const blob = new Blob([downloadResponse.data], { 
        type: format === 'xlsx' 
          ? 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
          : format === 'csv'
          ? 'text/csv'
          : 'application/xml'
      });

      // Create download link
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.style.display = 'none';
      a.href = url;
      a.download = `converted-file.${format}`;
      
      // Trigger download
      document.body.appendChild(a);
      a.click();
      
      // Cleanup after a short delay to ensure download starts
      setTimeout(() => {
        document.body.removeChild(a);
        window.URL.revokeObjectURL(url);
      }, 200);

      toast.success(`Successfully exported as ${format.toUpperCase()}`, { id: toastId });
    } catch (error: any) {
      console.error('Export error:', error);
      // Only treat actual errors as errors
      if (error.message === 'File converted successfully') {
        toast.success(`Successfully exported as ${format.toUpperCase()}`, { id: toastId });
      } else {
        const errorMessage = error.response?.data?.detail || error.message || 'Export failed';
        setError(errorMessage);
        toast.error(`Failed to export: ${errorMessage}`, { id: toastId });
      }
    } finally {
      setLoading(false);
    }
  };

  const handleBack = () => {
    navigate('/preview', { 
      state: { 
        fileId: currentFile?.id,
        data: tableData
      }
    });
  };

  const handlePasswordSubmit = async () => {
    setPasswordError('');
    if (!password.trim()) {
      setPasswordError('Please enter a password');
      return;
    }

    try {
      await handleValidate(password);
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

  const handleValidate = async (pdfPassword?: string) => {
    if (!currentFile?.id) {
      toast.error('No file selected');
      return;
    }

    if (convertedFormats.size === 0) {
      toast.error('Please convert to at least one format before proceeding');
      return;
    }

    const toastId = toast.loading('Validating data...');
    setLoading(true);
    setError(null);

    try {
      // Format the data properly for validation
      const sanitizedData = tableData.map(row => {
        const sanitizedRow: Record<string, any> = {};
        Object.entries(row).forEach(([key, value]) => {
          // Handle null/undefined
          if (value === null || value === undefined) {
            sanitizedRow[key] = '';
          }
          // Handle date objects and timestamps
          else if (value && typeof value === 'object' && 
                   ('toISOString' in value || 
                    (typeof (value as any).toDate === 'function'))) {
            try {
              // Handle both native Date objects and Timestamp objects
              const dateValue = ('toDate' in value) ? (value as any).toDate() : value;
              sanitizedRow[key] = dateValue.toISOString().split('T')[0];
            } catch {
              sanitizedRow[key] = '';
            }
          }
          // Handle string dates
          else if (typeof value === 'string' && /^\d{4}-\d{2}-\d{2}/.test(value)) {
            sanitizedRow[key] = value.split('T')[0];
          }
          // Handle numbers
          else if (typeof value === 'number') {
            sanitizedRow[key] = value.toString();
          }
          // Everything else as string
          else {
            sanitizedRow[key] = String(value);
          }
        });
        return sanitizedRow;
      });

      const validationPayload = {
        data: sanitizedData,
        formats: Array.from(convertedFormats),
        fileId: currentFile.id,
        password: pdfPassword || undefined
      };

      const response = await axios.post(`${API_URL}/validate/${currentFile.id}`, validationPayload);

      if (response.data?.validationResults) {
        toast.success('Validation complete', { id: toastId });
        setShowPasswordModal(false);
        navigate('/validate', { 
          state: { 
            fileId: currentFile?.id,
            data: tableData,
            convertedFormats: Array.from(convertedFormats),
            validationResults: response.data.validationResults
          }
        });
      } else if (response.data?.status === "success" && response.data?.message) {
        toast.success(response.data.message, { id: toastId });
        setShowPasswordModal(false);
        navigate('/validate', { 
          state: { 
            fileId: currentFile?.id,
            data: tableData,
            convertedFormats: Array.from(convertedFormats),
            validationResults: response.data.validationResults || []
          }
        });
      } else {
        throw new Error(response.data?.message || 'Validation failed');
      }
    } catch (error: any) {
      console.error('Validation error:', error);
      const errorMessage = error.response?.data?.detail || error.response?.data?.message || error.message || 'Validation failed';
      
      // Handle specific error cases
      if (error.response?.status === 401 && error.response?.data?.requires_password) {
        setShowPasswordModal(true);
        toast.error('PDF is password protected. Please enter the password.', { id: toastId });
      } else if (error.response?.status === 422) {
        setError(errorMessage);
        toast.error(`Data validation error: ${errorMessage}`, { id: toastId });
      } else {
        setError(errorMessage);
        toast.error(`Validation failed: ${errorMessage}`, { id: toastId });
      }
    } finally {
      setLoading(false);
    }
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
        <button
          onClick={() => handleValidate()}
          className="flex items-center px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700"
          disabled={convertedFormats.size === 0 || loading}
        >
          {loading ? (
            <ArrowPathIcon className="h-5 w-5 animate-spin" />
          ) : (
            <>
              Validate
              <ArrowRightIcon className="h-5 w-5 ml-2" />
            </>
          )}
        </button>
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
                  {convertedFormats.has(format.id) ? 'Download Again' : 'Convert & Download'}
                </>
              )}
            </button>
          </div>
        ))}
      </div>

      {tableData.length > 0 && (
        <div className="mt-8">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Preview Data</h3>
          <div className="bg-white rounded-lg shadow overflow-x-auto">
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
                {tableData.slice(0, 5).map((row, index) => (
                  <tr key={index}>
                    {Object.values(row).map((value, i) => (
                      <td key={i} className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                        {value}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
            {tableData.length > 5 && (
              <div className="px-6 py-3 bg-gray-50 text-sm text-gray-500">
                Showing 5 of {tableData.length} rows
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default ConvertComponent; 