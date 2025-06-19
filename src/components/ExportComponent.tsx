import React, { useState, useEffect } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import axios from 'axios';
import { API_URL } from '../config';
import { FinancialEntry, ValidationSummary } from '../types/financial';
import toast from 'react-hot-toast';
import {
  ArrowLeftIcon,
  DocumentArrowDownIcon,
  TableCellsIcon,
  DocumentIcon,
  DocumentTextIcon,
  ArrowPathIcon,
} from '@heroicons/react/24/outline';
import ExportSummaryModal from './ExportSummaryModal';

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
    name: 'Standard Excel',
    extension: 'xlsx',
    icon: TableCellsIcon,
    description: 'Export as standard Microsoft Excel file'
  },
  {
    id: 'tally',
    name: 'Tally Excel',
    extension: 'xlsx',
    icon: TableCellsIcon,
    description: 'Export in Tally-compatible Excel format'
  },
  {
    id: 'json',
    name: 'JSON Data',
    extension: 'json',
    icon: DocumentIcon,
    description: 'Export as JSON for system integrations'
  },
  {
    id: 'pdf',
    name: 'PDF Summary',
    extension: 'pdf',
    icon: DocumentTextIcon,
    description: 'Export as printable PDF summary'
  }
];

const ExportComponent: React.FC = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const [data, setData] = useState<FinancialEntry[]>([]);
  const [fileId, setFileId] = useState<string>('');
  const [loading, setLoading] = useState(false);
  const [clientName, setClientName] = useState('');
  const [exportedFormats, setExportedFormats] = useState<Set<string>>(new Set());
  const [showSummary, setShowSummary] = useState(false);
  const [summaryData, setSummaryData] = useState<{
    fileName: string;
    fileType: string;
    data: FinancialEntry[];
    validationSummary?: ValidationSummary;
    exportFormat: string;
    clientName: string;
    exportSuccess: boolean;
    downloadPath?: string;
  } | null>(null);

  useEffect(() => {
    const state = location.state as { fileId?: string; data?: FinancialEntry[]; validationSummary?: ValidationSummary };
    if (!state?.fileId || !state?.data) {
      toast.error('No data to export');
      navigate('/', { replace: true });
      return;
    }
    setFileId(state.fileId);
    setData(state.data);
  }, [location.state, navigate]);

  const handleExport = async (format: ExportFormat) => {
    if (!clientName.trim()) {
      toast.error('Please enter a client name');
      return;
    }

    setLoading(true);
    const toastId = toast.loading(`Exporting as ${format.name}...`);

    try {
      const response = await axios.post(
        `${API_URL}/export/${fileId}/${format.id}`,
        { clientName },
        { responseType: 'blob' }
      );

      if (!response.data || response.data.size === 0) {
        throw new Error('Received empty response from server');
      }

      setExportedFormats(prev => new Set([...prev, format.id]));

      const date = new Date().toISOString().split('T')[0];
      const filename = `${clientName.replace(/[^a-z0-9]/gi, '_').toLowerCase()}_${date}_${format.id}.${format.extension}`;
      
      const blob = new Blob([response.data], {
        type: format.extension === 'xlsx'
          ? 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
          : format.extension === 'json'
          ? 'application/json'
          : format.extension === 'pdf'
          ? 'application/pdf'
          : 'application/octet-stream'
      });

      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      window.URL.revokeObjectURL(url);

      // Log export action
      await axios.post(`${API_URL}/audit/log`, {
        action: 'export',
        format: format.id,
        filename,
        timestamp: new Date().toISOString()
      });

      toast.success(`Successfully exported as ${format.name}`, { id: toastId });

      // Show export summary
      setSummaryData({
        fileName: filename,
        fileType: location.state?.fileType || 'Unknown',
        data,
        validationSummary: location.state?.validationSummary,
        exportFormat: format.name,
        clientName,
        exportSuccess: true,
        downloadPath: filename
      });
      setShowSummary(true);
    } catch (error) {
      console.error('Export error:', error);
      toast.error(`Failed to export as ${format.name}`, { id: toastId });

      // Show error summary
      setSummaryData({
        fileName: `${clientName}_${format.id}.${format.extension}`,
        fileType: location.state?.fileType || 'Unknown',
        data,
        validationSummary: location.state?.validationSummary,
        exportFormat: format.name,
        clientName,
        exportSuccess: false
      });
      setShowSummary(true);
    } finally {
      setLoading(false);
    }
  };

  const handleBack = () => {
    navigate('/validate', {
      state: {
        fileId,
        data
      }
    });
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
        <h2 className="text-2xl font-semibold text-gray-900">Export Data</h2>
      </div>

      <div className="bg-white rounded-lg shadow-md p-6">
        <div className="mb-6">
          <label htmlFor="clientName" className="block text-sm font-medium text-gray-700 mb-2">
            Client Name
          </label>
          <input
            type="text"
            id="clientName"
            value={clientName}
            onChange={(e) => setClientName(e.target.value)}
            placeholder="Enter client name"
            className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-blue-500 focus:border-blue-500"
          />
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {EXPORT_FORMATS.map((format) => (
            <div
              key={format.id}
              className="bg-white p-6 rounded-lg shadow-sm border border-gray-200 hover:border-blue-500 transition-colors"
            >
              <div className="flex items-start justify-between mb-4">
                <div>
                  <format.icon className="h-8 w-8 text-blue-600 mb-2" />
                  <h3 className="text-lg font-semibold text-gray-900">{format.name}</h3>
                  <p className="text-gray-600 text-sm mt-1">{format.description}</p>
                </div>
                {exportedFormats.has(format.id) && (
                  <span className="text-green-600 text-sm">Exported</span>
                )}
              </div>
              <button
                onClick={() => handleExport(format)}
                disabled={loading}
                className="w-full flex items-center justify-center px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-400"
              >
                {loading ? (
                  <ArrowPathIcon className="h-5 w-5 animate-spin" />
                ) : (
                  <>
                    <DocumentArrowDownIcon className="h-5 w-5 mr-2" />
                    {exportedFormats.has(format.id) ? 'Export Again' : 'Export'}
                  </>
                )}
              </button>
            </div>
          ))}
        </div>
      </div>

      {summaryData && (
        <ExportSummaryModal
          isOpen={showSummary}
          onClose={() => setShowSummary(false)}
          exportData={summaryData}
        />
      )}
    </div>
  );
};

export default ExportComponent; 