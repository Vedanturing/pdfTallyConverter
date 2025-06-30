import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { toast } from 'react-hot-toast';
import { API_URL } from '../config';
import { 
  DocumentArrowDownIcon, 
  TableCellsIcon,
  DocumentTextIcon 
} from '@heroicons/react/24/outline';

interface TallyDataPreviewProps {
  fileId: string;
  fileName: string;
  data: {
    headers: string[];
    rows: any[][];
  };
}

const TallyDataPreview: React.FC<TallyDataPreviewProps> = ({ fileId, fileName, data }) => {
  const [isExporting, setIsExporting] = useState<{[key: string]: boolean}>({});

  const handleExport = async (format: 'xlsx' | 'csv') => {
    try {
      setIsExporting(prev => ({ ...prev, [format]: true }));
      
      const response = await axios.get(`${API_URL}/api/convert/${fileId}/${format}`, {
        responseType: 'blob',
        timeout: 60000, // 1 minute timeout
      });

      // Create blob and download
      const blob = new Blob([response.data], {
        type: format === 'xlsx' 
          ? 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
          : 'text/csv'
      });

      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      
      // Get filename from headers or generate default
      let filename = `${fileName.replace(/\.[^/.]+$/, '')}.${format}`;
      const contentDisposition = response.headers['content-disposition'];
      if (contentDisposition) {
        const filenameMatch = contentDisposition.match(/filename[^;=\n]*=((['"]).*?\2|[^;\n]*)/);
        if (filenameMatch) {
          filename = filenameMatch[1].replace(/['"]/g, '');
        }
      }
      
      link.download = filename;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      window.URL.revokeObjectURL(url);

      toast.success(`Successfully exported as ${format.toUpperCase()}`);
    } catch (error: any) {
      console.error('Export error:', error);
      toast.error(`Failed to export as ${format.toUpperCase()}`);
    } finally {
      setIsExporting(prev => ({ ...prev, [format]: false }));
    }
  };

  return (
    <div className="bg-white rounded-lg shadow-lg p-6">
      <div className="flex justify-between items-center mb-6">
        <div>
          <h2 className="text-xl font-semibold text-gray-900">Tally Data Preview</h2>
          <p className="text-sm text-gray-600 mt-1">
            File: {fileName} • {data.rows.length} records found
          </p>
        </div>
        
        <div className="flex space-x-3">
          <button
            onClick={() => handleExport('xlsx')}
            disabled={isExporting.xlsx}
            className="inline-flex items-center px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            {isExporting.xlsx ? (
              <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
            ) : (
              <TableCellsIcon className="h-4 w-4 mr-2" />
            )}
            Export Excel
          </button>
          
          <button
            onClick={() => handleExport('csv')}
            disabled={isExporting.csv}
            className="inline-flex items-center px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            {isExporting.csv ? (
              <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
            ) : (
              <DocumentTextIcon className="h-4 w-4 mr-2" />
            )}
            Export CSV
          </button>
        </div>
      </div>

      {/* Data Table */}
      <div className="overflow-x-auto border rounded-lg">
        <table className="min-w-full divide-y divide-gray-200">
          <thead className="bg-gray-50">
            <tr>
              {data.headers.map((header, index) => (
                <th
                  key={index}
                  className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider"
                >
                  {header}
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-gray-200">
            {data.rows.slice(0, 10).map((row, rowIndex) => (
              <tr key={rowIndex} className="hover:bg-gray-50">
                {row.map((cell, cellIndex) => (
                  <td
                    key={cellIndex}
                    className="px-6 py-4 whitespace-nowrap text-sm text-gray-900"
                  >
                    {cell || '-'}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {data.rows.length > 10 && (
        <div className="mt-4 text-center">
          <p className="text-sm text-gray-500">
            Showing first 10 rows of {data.rows.length} total records.
            Export to view all data.
          </p>
        </div>
      )}

      {/* Data Summary */}
      <div className="mt-6 grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="bg-blue-50 p-4 rounded-lg">
          <div className="text-sm font-medium text-blue-800">Total Records</div>
          <div className="text-2xl font-bold text-blue-900">{data.rows.length}</div>
        </div>
        
        <div className="bg-green-50 p-4 rounded-lg">
          <div className="text-sm font-medium text-green-800">Columns</div>
          <div className="text-2xl font-bold text-green-900">{data.headers.length}</div>
        </div>
        
        <div className="bg-purple-50 p-4 rounded-lg">
          <div className="text-sm font-medium text-purple-800">File Type</div>
          <div className="text-2xl font-bold text-purple-900">
            {fileName.toLowerCase().endsWith('.xml') ? 'XML' : 'TXT'}
          </div>
        </div>
      </div>
    </div>
  );
};

export default TallyDataPreview; 