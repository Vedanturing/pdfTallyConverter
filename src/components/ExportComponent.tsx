import React, { useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { 
  ArrowLeftIcon, 
  DocumentArrowDownIcon,
  TableCellsIcon,
  DocumentTextIcon,
  CodeBracketIcon,
  CurrencyDollarIcon
} from '@heroicons/react/24/outline';
import axios from 'axios';
import toast from 'react-hot-toast';
import { API_URL } from '../config';

interface ExportFormat {
  id: string;
  name: string;
  description: string;
  icon: React.ComponentType<{ className?: string }>;
  disabled?: boolean;
}

export default function ExportComponent() {
  const { t, i18n } = useTranslation();
  const location = useLocation();
  const navigate = useNavigate();
  const [isExporting, setIsExporting] = useState(false);
  const [selectedLanguage, setSelectedLanguage] = useState(i18n.language || 'en');

  const fileId = location.state?.fileId;
  const data = location.state?.data;
  const convertedFormats = location.state?.convertedFormats;

  const exportFormats: ExportFormat[] = [
    {
      id: 'xlsx',
      name: 'Excel Spreadsheet',
      description: 'Export as Microsoft Excel file (.xlsx)',
      icon: TableCellsIcon,
    },
    {
      id: 'csv',
      name: 'CSV File',
      description: 'Export as Comma Separated Values (.csv)',
      icon: DocumentTextIcon,
    },
    {
      id: 'json',
      name: 'JSON Document',
      description: 'Export as JSON for structured data',
      icon: CodeBracketIcon,
    },
    {
      id: 'xml',
      name: 'XML Document',
      description: 'Export as generic XML format',
      icon: CodeBracketIcon,
    },
    {
      id: 'tally',
      name: 'Tally XML',
      description: 'Export in Tally-compatible XML format',
      icon: CurrencyDollarIcon,
    },
  ];

  const languageOptions = [
    { code: 'en', name: 'English', flag: '🇺🇸' },
    { code: 'hi', name: 'हिन्दी', flag: '🇮🇳' },
    { code: 'mr', name: 'मराठी', flag: '🇮🇳' },
  ];

  const handleExport = async (format: string) => {
    if (!fileId || !data) {
      toast.error('No data available for export');
      return;
    }

    setIsExporting(true);
    const toastId = toast.loading(`Exporting as ${format.toUpperCase()} in ${languageOptions.find(l => l.code === selectedLanguage)?.name || 'English'}...`);

    try {
      const response = await axios.post(
        `${API_URL}/export/${fileId}/${format}`,
        {
          data,
          clientName: 'export',
          language: selectedLanguage, // Include language in request
          localization: {
            dateFormat: selectedLanguage === 'en' ? 'MM/DD/YYYY' : 'DD/MM/YYYY',
            currency: '₹',
            numberFormat: selectedLanguage === 'en' ? 'US' : 'IN'
          }
        },
        {
          responseType: 'blob',
          timeout: 30000, // 30 second timeout
          headers: {
            'Content-Type': 'application/json',
          },
        }
      );

      // Get proper MIME type based on format
      const mimeTypes: Record<string, string> = {
        'xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        'csv': 'text/csv',
        'json': 'application/json',
        'xml': 'application/xml',
        'tally': 'application/xml'
      };

      const mimeType = mimeTypes[format] || 'application/octet-stream';

      // Create download link with proper MIME type
      const blob = new Blob([response.data], { type: mimeType });
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      
      // Get filename from headers or generate default with language suffix
      let filename = '';
      const contentDisposition = response.headers['content-disposition'];
      if (contentDisposition) {
        const filenameMatch = contentDisposition.match(/filename[^;=\n]*=((['"]).*?\2|[^;\n]*)/);
        if (filenameMatch) {
          filename = filenameMatch[1].replace(/['"]/g, '');
        }
      }
      
      if (!filename) {
        const timestamp = new Date().toISOString().slice(0, 10);
        const extensions: Record<string, string> = {
          'xlsx': 'xlsx',
          'csv': 'csv', 
          'json': 'json',
          'xml': 'xml',
          'tally': 'xml'
        };
        filename = `export_${selectedLanguage}_${timestamp}.${extensions[format] || format}`;
      }
      
      link.setAttribute('download', filename);
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.URL.revokeObjectURL(url);

      toast.success(`Successfully exported as ${format.toUpperCase()} in ${languageOptions.find(l => l.code === selectedLanguage)?.name}`, { id: toastId });
    } catch (error: any) {
      console.error('Export error:', error);
      let errorMessage = 'Export failed';
      
      if (error.response?.status === 404) {
        errorMessage = 'Data not found. Please validate your data first.';
      } else if (error.response?.status === 500) {
        errorMessage = 'Server error during export. Please try again.';
      } else if (error.code === 'ECONNABORTED') {
        errorMessage = 'Export timeout. Please try again.';
      } else {
        errorMessage = error.response?.data?.detail || error.message || errorMessage;
      }
      
      toast.error(errorMessage, { id: toastId });
    } finally {
      setIsExporting(false);
    }
  };

  const handleBack = () => {
    navigate('/validate', {
      state: {
        fileId,
        data,
        convertedFormats,
      },
    });
  };

  if (!fileId || !data) {
    return (
      <div className="min-h-screen bg-gray-50 dark:bg-gray-900 flex items-center justify-center">
        <div className="text-center">
          <DocumentArrowDownIcon className="h-12 w-12 text-gray-400 dark:text-gray-500 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-gray-900 dark:text-gray-100 mb-2">
            No Data Available
          </h3>
          <p className="text-sm text-gray-500 dark:text-gray-400 mb-4">
            No data found for export. Please go back to process your document.
          </p>
          <button
            onClick={() => navigate('/convert')}
            className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700"
          >
            Go to Convert
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      {/* Header */}
      <div className="bg-white dark:bg-gray-800 shadow-sm border-b border-gray-200 dark:border-gray-700">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-4">
            <div className="flex items-center space-x-4">
              <button
                onClick={handleBack}
                className="flex items-center text-gray-600 hover:text-gray-900 dark:text-gray-400 dark:hover:text-gray-100 transition-colors"
              >
                <ArrowLeftIcon className="h-5 w-5 mr-2" />
                {t('common.back')}
              </button>
              <div>
                <h1 className="text-2xl font-bold text-gray-900 dark:text-gray-100">
                  {t('export.title')}
                </h1>
                <p className="text-sm text-gray-500 dark:text-gray-400">
                  Choose your preferred export format
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Content */}
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="mb-6">
          <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-2">
            {t('export.formats')}
          </h2>
          <p className="text-sm text-gray-600 dark:text-gray-400">
            Select a format to download your processed data
          </p>
        </div>

        {/* Language Selection */}
        <div className="mb-8">
          <h3 className="text-lg font-medium text-gray-900 dark:text-gray-100 mb-4">
            Select Export Language
          </h3>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            {languageOptions.map((lang) => (
              <button
                key={lang.code}
                onClick={() => setSelectedLanguage(lang.code)}
                className={`p-4 rounded-lg border-2 transition-all ${
                  selectedLanguage === lang.code
                    ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
                    : 'border-gray-200 dark:border-gray-700 hover:border-gray-300 dark:hover:border-gray-600'
                }`}
              >
                <div className="flex items-center justify-center space-x-3">
                  <span className="text-2xl">{lang.flag}</span>
                  <span className="font-medium text-gray-900 dark:text-gray-100">
                    {lang.name}
                  </span>
                </div>
                {selectedLanguage === lang.code && (
                  <div className="mt-2 text-sm text-blue-600 dark:text-blue-400">
                    ✓ Selected
                  </div>
                )}
              </button>
            ))}
          </div>
          <p className="mt-3 text-sm text-gray-600 dark:text-gray-400">
            The exported file will use localized date formats, number formats, and column headers based on your selection.
          </p>
        </div>

        {/* Export Formats */}
        <div>
          <h3 className="text-lg font-medium text-gray-900 dark:text-gray-100 mb-6">
            Choose Export Format
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {exportFormats.map((format) => {
              const IconComponent = format.icon;
              const isDisabled = format.disabled || isExporting;

              return (
                <button
                  key={format.id}
                  onClick={() => handleExport(format.id)}
                  disabled={isDisabled}
                  className={`
                    relative p-6 rounded-lg border-2 transition-all duration-200
                    ${
                      isDisabled
                        ? 'border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-800 cursor-not-allowed opacity-50'
                        : 'border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 hover:border-blue-300 dark:hover:border-blue-600 hover:shadow-md cursor-pointer'
                    }
                  `}
                >
                  <div className="flex flex-col items-center text-center">
                    <IconComponent 
                      className={`h-8 w-8 mb-3 ${
                        isDisabled 
                          ? 'text-gray-400 dark:text-gray-600' 
                          : 'text-blue-600 dark:text-blue-400'
                      }`} 
                    />
                    <h3 className="font-medium text-gray-900 dark:text-gray-100 mb-1">
                      {format.name}
                    </h3>
                    <p className="text-xs text-gray-500 dark:text-gray-400">
                      {format.description}
                    </p>
                  </div>
                  
                  {isExporting && (
                    <div className="absolute inset-0 flex items-center justify-center bg-white dark:bg-gray-800 bg-opacity-90 dark:bg-opacity-90 rounded-lg">
                      <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-600"></div>
                    </div>
                  )}
                </button>
              );
            })}
          </div>
        </div>

        {/* Data Summary */}
        <div className="mt-8 bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
          <h3 className="text-lg font-medium text-gray-900 dark:text-gray-100 mb-4">
            Export Summary
          </h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="text-center">
              <div className="text-2xl font-bold text-blue-600 dark:text-blue-400">
                {data?.length || 0}
              </div>
              <div className="text-sm text-gray-500 dark:text-gray-400">
                Total Rows
              </div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-green-600 dark:text-green-400">
                {Object.keys(data?.[0] || {}).length || 0}
              </div>
              <div className="text-sm text-gray-500 dark:text-gray-400">
                Columns
              </div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-purple-600 dark:text-purple-400">
                {convertedFormats?.length || 0}
              </div>
              <div className="text-sm text-gray-500 dark:text-gray-400">
                Formats Available
              </div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-orange-600 dark:text-orange-400">
                PDF
              </div>
              <div className="text-sm text-gray-500 dark:text-gray-400">
                Source Format
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
} 