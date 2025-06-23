import React, { useState } from 'react';
import axios from 'axios';
import toast, { Toaster } from 'react-hot-toast';
import { ArrowDownTrayIcon, DocumentIcon, InformationCircleIcon } from '@heroicons/react/24/outline';
import { v4 as uuidv4 } from 'uuid';
import { TableCell, TableRow, TableData, ConversionResult } from '../types/DataValidation';
import { API_URL } from '../config';

interface TablePreview {
  headers: string[];
  rows: string[][];
}

const FileConverter: React.FC = () => {
  const [file, setFile] = useState<File | null>(null);
  const [isConverting, setIsConverting] = useState(false);
  const [converted, setConverted] = useState(false);
  const [tableData, setTableData] = useState<TableData | null>(null);
  const [isDragging, setIsDragging] = useState(false);

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (files && files.length > 0) {
      setFile(files[0]);
      setConverted(false);
      toast.success('File selected successfully!');
    }
  };

  const convertToTable = async () => {
    if (!file) {
      toast.error('Please select a file first');
      return;
    }

    setIsConverting(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post<ConversionResult>(`${API_URL}/convert`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      if (!response.data.headers || !response.data.rows) {
        throw new Error('Invalid response format from server');
      }

      const { headers, rows } = response.data;
      const tableRows: TableRow[] = rows.map((row: any) => ({
        id: { value: uuidv4(), isEdited: false },
        ...Object.fromEntries(
          Object.keys(row).map(key => [
            key,
            { value: String(row[key]), isEdited: false }
          ])
        )
      }));

      const newTableData: TableData = {
        headers,
        rows: tableRows
      };

      setTableData(newTableData);
      setConverted(true);
      toast.success('File converted successfully!');
    } catch (error) {
      console.error('Error converting file:', error);
      toast.error('Error converting file. Please make sure the backend server is running.');
    } finally {
      setIsConverting(false);
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const files = e.dataTransfer.files;
    if (files && files.length > 0) {
      setFile(files[0]);
      setConverted(false);
      toast.success('File selected successfully!');
    }
  };

  return (
    <div className="space-y-8">
      {/* User Guide Section */}
      <div className="bg-gradient-to-r from-blue-50 to-indigo-50 p-6 rounded-xl shadow-sm border border-blue-100 transform transition-all duration-300 hover:scale-[1.02]">
        <div className="flex items-start space-x-4">
          <InformationCircleIcon className="h-8 w-8 text-indigo-500 mt-1 animate-pulse" />
          <div>
            <h3 className="text-xl font-semibold bg-clip-text text-transparent bg-gradient-to-r from-blue-600 to-indigo-600">
              How to use:
            </h3>
            <ul className="list-disc list-inside text-gray-600 text-sm mt-3 space-y-2">
              <li className="transform transition-all duration-200 hover:translate-x-2">
                Upload your PDF file using the drop zone below or browse button
              </li>
              <li className="transform transition-all duration-200 hover:translate-x-2">
                Click the "Convert to Table" button to process your file
              </li>
              <li className="transform transition-all duration-200 hover:translate-x-2">
                View and verify the converted data in the table below
              </li>
            </ul>
          </div>
        </div>
      </div>

      {/* File Upload Section */}
      <div
        className={`relative overflow-hidden rounded-xl p-8 text-center transition-all duration-300 transform hover:scale-[1.02] ${
          isDragging 
            ? 'bg-gradient-to-r from-indigo-50 to-blue-50 border-2 border-dashed border-indigo-400' 
            : 'bg-white border-2 border-dashed border-gray-300 hover:border-indigo-400 hover:bg-gradient-to-r hover:from-indigo-50 hover:to-blue-50'
        }`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        <div className="relative z-10">
          <div className="flex flex-col items-center space-y-4">
            <DocumentIcon className={`h-16 w-16 transition-colors duration-300 ${
              file ? 'text-indigo-500' : 'text-gray-400'
            }`} />
            <div className="space-y-2">
              <h4 className="text-xl font-medium text-gray-700">
                {file ? file.name : 'Drop your PDF file here'}
              </h4>
              <p className="text-sm text-gray-500">
                or
              </p>
              <label className="inline-flex items-center px-6 py-3 bg-gradient-to-r from-indigo-500 to-blue-500 text-white rounded-lg shadow-lg hover:from-indigo-600 hover:to-blue-600 transition-all duration-300 cursor-pointer transform hover:scale-105 hover:shadow-xl">
                <input
                  type="file"
                  accept=".pdf"
                  onChange={handleFileChange}
                  className="sr-only"
                />
                Browse Files
              </label>
            </div>
            <p className="text-xs text-gray-500">Supported format: PDF</p>
          </div>
        </div>
        <div className="absolute inset-0 bg-gradient-to-r from-indigo-50/30 to-blue-50/30 opacity-0 transition-opacity duration-300 pointer-events-none"></div>
      </div>

      {/* Convert Button */}
      <div className="flex justify-center">
        <button
          onClick={convertToTable}
          disabled={!file || isConverting}
          className={`
            relative overflow-hidden inline-flex items-center px-8 py-4 rounded-xl text-lg font-medium shadow-lg
            transform transition-all duration-300 
            ${!file || isConverting
              ? 'bg-gray-300 cursor-not-allowed text-gray-500'
              : 'bg-gradient-to-r from-indigo-500 to-blue-500 hover:from-indigo-600 hover:to-blue-600 text-white hover:scale-105 hover:shadow-xl'
            }
          `}
        >
          {isConverting ? (
            <>
              <ArrowDownTrayIcon className="animate-bounce h-6 w-6 mr-3" />
              Converting...
            </>
          ) : (
            <>
              <ArrowDownTrayIcon className="h-6 w-6 mr-3" />
              Convert to Table
            </>
          )}
          <div className="absolute inset-0 bg-gradient-to-r from-white/0 via-white/20 to-white/0 transform translate-x-[-100%] animate-shimmer"></div>
        </button>
      </div>

      {/* Results Table */}
      {tableData && (
        <div className="mt-8 transform transition-all duration-300 hover:scale-[1.01]">
          <h2 className="text-3xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-indigo-600 to-blue-600 mb-6">
            Converted Table
          </h2>
          <div className="overflow-hidden bg-white rounded-xl shadow-lg border border-gray-100">
            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-gray-200">
                <thead>
                  <tr className="bg-gradient-to-r from-indigo-50 to-blue-50">
                    {tableData.headers.map((header: string, index: number) => (
                      <th
                        key={index}
                        className="px-6 py-4 text-left text-sm font-semibold text-gray-900 uppercase tracking-wider"
                      >
                        {header}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                  {tableData.rows.map((row: TableRow, rowIndex: number) => (
                    <tr 
                      key={row.id.value} 
                      className={`
                        transform transition-all duration-200 hover:bg-gradient-to-r hover:from-indigo-50/30 hover:to-blue-50/30
                        ${rowIndex % 2 === 0 ? 'bg-gray-50' : 'bg-white'}
                      `}
                    >
                      {tableData.headers.map((header: string) => (
                        <td
                          key={`${row.id.value}-${header}`}
                          className="px-6 py-4 whitespace-nowrap text-sm text-gray-600"
                        >
                          {row[header]?.value || ''}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default FileConverter; 