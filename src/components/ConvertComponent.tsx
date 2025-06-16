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
} from '@heroicons/react/24/outline';
import toast from 'react-hot-toast';
import { useNavigate, useLocation } from 'react-router-dom';
import WorkflowStepper from './WorkflowStepper';
import { useWorkflowStore } from '../store/useWorkflowStore';
import { Button, Box, Typography, CircularProgress } from '@mui/material';

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
  const [loading, setLoading] = useState(true);
  const [converting, setConverting] = useState(false);
  const [selectedFormat, setSelectedFormat] = useState<string | null>(null);
  const [convertedFormats, setConvertedFormats] = useState<Set<string>>(new Set());
  const [tableData, setTableData] = useState<any[]>([]);
  const [error, setError] = useState<string | null>(null);
  const navigate = useNavigate();
  const location = useLocation();

  useEffect(() => {
    const state = location.state as { fileId?: string, fileName?: string };
    if (!state?.fileId) {
      toast.error('No file selected');
      navigate('/', { replace: true });
      return;
    }
    
    setCurrentFile({
      id: state.fileId,
      name: state.fileName || 'Uploaded File',
      uploadedAt: new Date().toISOString()
    });

    // Fetch table data
    const fetchData = async () => {
      setLoading(true);
      setError(null);
      try {
        const response = await axios.get(`${API_URL}/get-data/${state.fileId}`);
        if (response.data) {
          if (response.data.rows && Array.isArray(response.data.rows)) {
            setTableData(response.data.rows);
          } else {
            throw new Error('Invalid data format received from server');
          }
        } else {
          throw new Error('No data received from server');
        }
      } catch (error: any) {
        console.error('Error fetching table data:', error);
        const errorMessage = error.response?.data?.detail || error.message || 'Failed to load table data';
        setError(errorMessage);
        toast.error(errorMessage);
        if (error.response?.status === 404) {
          navigate('/', { replace: true });
        }
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, [location.state, navigate]);

  const handleExport = async (format: string) => {
    if (!currentFile?.id) {
      toast.error('No file selected');
      return;
    }

    setSelectedFormat(format);
    setConverting(true);
    
    try {
      console.log(`Starting export for file ${currentFile.id} to ${format}`);
      const response = await axios.get(`${API_URL}/api/download/${currentFile.id}/${format}`, {
        responseType: 'blob'
      });

      // Check if the response is valid
      if (!response.data || response.data.size === 0) {
        throw new Error('Received empty response from server');
      }

      setConvertedFormats(prev => new Set([...prev, format]));
      
      const blob = new Blob([response.data], { 
        type: format === 'xlsx' 
          ? 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
          : format === 'csv'
          ? 'text/csv'
          : 'application/xml'
      });

      // Create and trigger download
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `converted-file.${format}`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      window.URL.revokeObjectURL(url);

      toast.success(`Successfully exported as ${format.toUpperCase()}`);
    } catch (error: any) {
      console.error('Export error:', error);
      let errorMessage = `Failed to export as ${format.toUpperCase()}`;
      
      // Extract error message from response if available
      if (error.response?.data) {
        try {
          const reader = new FileReader();
          reader.onload = () => {
            const text = reader.result as string;
            try {
              const errorData = JSON.parse(text);
              errorMessage = errorData.detail || errorMessage;
            } catch (e) {
              // If JSON parsing fails, use the text as is
              errorMessage = text || errorMessage;
            }
            toast.error(errorMessage);
          };
          reader.readAsText(error.response.data);
        } catch (e) {
          toast.error(errorMessage);
        }
      } else {
        toast.error(errorMessage);
      }
    } finally {
      setSelectedFormat(null);
      setConverting(false);
    }
  };

  const handleMoveToValidate = () => {
    if (convertedFormats.size === 0) {
      toast.error('Please convert to at least one format before proceeding');
      return;
    }
    navigate('/validate', { 
      state: { 
        data: tableData,
        fileId: currentFile?.id,
        convertedFormats: Array.from(convertedFormats)
      },
      replace: true
    });
  };

  const handleValidate = () => {
    if (tableData.length > 0) {
      navigate('/validate', { 
        state: { 
          data: tableData,
          fileId: currentFile?.id 
        } 
      });
    }
  };

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center min-h-screen p-4">
        <CircularProgress />
        <Typography variant="h6" className="mt-4">
          Converting your document...
        </Typography>
        <Typography variant="body2" color="textSecondary" className="mt-2">
          This may take a few moments
        </Typography>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex flex-col items-center justify-center min-h-screen p-4">
        <XCircleIcon className="w-16 h-16 text-red-500" />
        <Typography variant="h6" className="mt-4">
          Error Loading Document
        </Typography>
        <Typography variant="body2" color="textSecondary" className="mt-2">
          {error}
        </Typography>
        <Button
          variant="contained"
          color="primary"
          onClick={() => navigate('/')}
          className="mt-4"
        >
          Return to Upload
        </Button>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Workflow Stepper */}
      <WorkflowStepper currentStep="convert" />

      {/* User Guidance */}
      <div className="bg-blue-50 border-l-4 border-blue-400 p-4 rounded-r-md shadow-sm">
        <div className="flex">
          <div className="flex-shrink-0">
            <svg className="h-5 w-5 text-blue-400" viewBox="0 0 20 20" fill="currentColor">
              <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
            </svg>
          </div>
          <div className="ml-3">
            <p className="text-sm text-blue-700">
              Choose your desired export format(s). You can export to multiple formats before proceeding to validation.
              Once you're done converting, click "Move to Validate" to check and edit your data.
            </p>
          </div>
        </div>
      </div>

      {/* Current File */}
      {currentFile && (
        <div className="bg-white rounded-lg shadow-sm p-6">
          <h3 className="text-lg font-medium text-gray-900 mb-4">Current File</h3>
          <div className="p-4 rounded-lg border border-gray-200">
            <div className="flex items-center space-x-3">
              <DocumentArrowDownIcon className="h-6 w-6 text-gray-400" />
              <div className="flex-1">
                <p className="text-sm font-medium text-gray-900">{currentFile.name}</p>
                <p className="text-xs text-gray-500">
                  {new Date(currentFile.uploadedAt).toLocaleDateString()}
                </p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Export Format Grid */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {EXPORT_FORMATS.map((format) => (
          <div
            key={format.id}
            className="relative rounded-lg border border-gray-200 bg-white p-6 shadow-sm hover:shadow-md transition-all duration-200"
          >
            <div className="flex items-center">
              <div className="flex-shrink-0">
                <format.icon className={`h-8 w-8 ${
                  convertedFormats.has(format.id) ? 'text-indigo-600' : 'text-gray-400'
                }`} />
              </div>
              <div className="ml-4">
                <h3 className="text-lg font-medium text-gray-900">
                  {format.name}
                </h3>
                <p className="mt-1 text-sm text-gray-500">
                  {format.description}
                </p>
              </div>
            </div>
            <div className="mt-4">
              <button
                onClick={() => handleExport(format.id)}
                disabled={converting && selectedFormat === format.id}
                className={`inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white ${
                  convertedFormats.has(format.id)
                    ? 'bg-green-600 hover:bg-green-700'
                    : 'bg-indigo-600 hover:bg-indigo-700'
                } focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200 w-full justify-center`}
              >
                {converting && selectedFormat === format.id ? (
                  <>
                    <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                    </svg>
                    Converting...
                  </>
                ) : (
                  <>
                    {convertedFormats.has(format.id) ? (
                      <>
                        <CheckCircleIcon className="mr-2 h-5 w-5 text-white" />
                        Downloaded
                      </>
                    ) : (
                      <>Export as {format.extension.toUpperCase()}</>
                    )}
                  </>
                )}
              </button>
            </div>
          </div>
        ))}
      </div>

      {/* Action Buttons */}
      <div className="flex justify-end space-x-4">
        <button
          onClick={handleMoveToValidate}
          disabled={convertedFormats.size === 0}
          className="inline-flex items-center px-6 py-3 border border-transparent text-base font-medium rounded-md shadow-sm text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200"
        >
          Move to Validate
          <ArrowRightIcon className="ml-2 h-5 w-5" />
        </button>
      </div>
    </div>
  );
};

export default ConvertComponent; 