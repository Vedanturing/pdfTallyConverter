import React, { useState, useEffect } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { ValidationModule } from './ValidationModule';
import { ArrowLeftIcon, ArrowDownTrayIcon, ArrowRightIcon } from '@heroicons/react/24/outline';
import axios from 'axios';
import toast from 'react-hot-toast';
import { API_URL } from '../config';
import ValidationTable from './ValidationTable';

const ValidationComponent: React.FC = () => {
  const { t } = useTranslation();
  const location = useLocation();
  const navigate = useNavigate();
  
  // Get state from navigation
  const { fileId, data: initialData, convertedFormats } = location.state || {};
  
  const [tableData, setTableData] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!fileId) {
      toast.error('No file selected for validation');
      navigate('/convert');
      return;
    }
    
    if (initialData && initialData.length > 0) {
      // Filter out irrelevant columns for validation
      const filteredData = filterIrrelevantColumns(initialData);
      setTableData(filteredData);
      toast.success(`Loaded ${filteredData.length} rows for validation`);
    } else {
      toast.error('No data available for validation');
      navigate('/convert');
    }
  }, [fileId, initialData, navigate]);

  // Filter out irrelevant columns for validation
  const filterIrrelevantColumns = (data: any[]): any[] => {
    if (!data || data.length === 0) return data;
    
    const irrelevantColumns = [
      'gstin', 'gst', 'tax', 'voucher_no', 'voucherno', 'voucher', 
      'invoice_no', 'invoiceno', 'invoice', 'bill_no', 'billno',
      'reference_no', 'referenceno', 'ref_no', 'refno', 'utr',
      'cheque_no', 'chequeno', 'cheque'
    ];
    
    return data.map(row => {
      const filteredRow: any = {};
      Object.keys(row).forEach(key => {
        const keyLower = key.toLowerCase().replace(/[_\s]/g, '');
        const isIrrelevant = irrelevantColumns.some(irrelevant => 
          keyLower.includes(irrelevant) || irrelevant.includes(keyLower)
        );
        
        if (!isIrrelevant) {
          filteredRow[key] = row[key];
        }
      });
      return filteredRow;
    });
  };

  const handleDataChange = (newData: any[]) => {
    setTableData(newData);
  };

  const handleBack = () => {
    navigate(-1);
  };

  const handleExport = () => {
    navigate('/export', {
      state: {
        fileId,
        data: tableData,
        convertedFormats
      }
    });
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500" />
      </div>
    );
  }

  return (
    <div className="container mx-auto p-6 space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
              <button
                onClick={handleBack}
          className="flex items-center text-gray-600 hover:text-gray-900"
              >
                <ArrowLeftIcon className="h-5 w-5 mr-2" />
          Back
              </button>
        <h2 className="text-2xl font-semibold text-gray-900">{t('validation.title')}</h2>
            <button
              onClick={handleExport}
          className="flex items-center px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700"
            >
          Proceed to Export
          <ArrowRightIcon className="h-5 w-5 ml-2" />
            </button>
          </div>

      {/* Info Panel */}
      <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
        <h3 className="text-lg font-semibold text-blue-900 mb-2">Data Validation</h3>
        <p className="text-blue-700 text-sm">
          Review and edit your data below. Irrelevant columns like GSTIN, tax details, and voucher numbers 
          have been filtered out for focused validation. You can add/remove rows and columns as needed.
        </p>
        <div className="mt-3 flex items-center space-x-4 text-sm">
          <span className="px-2 py-1 bg-blue-100 text-blue-800 rounded">
            {tableData.length} rows
          </span>
          <span className="px-2 py-1 bg-blue-100 text-blue-800 rounded">
            {tableData.length > 0 ? Object.keys(tableData[0]).length : 0} columns
          </span>
        </div>
      </div>

      {/* Validation Table */}
      {tableData.length > 0 ? (
        <ValidationTable
          data={tableData}
          onDataChange={handleDataChange}
          readOnly={false}
          fileId={fileId} 
        />
      ) : (
        <div className="text-center py-12">
          <p className="text-gray-500">No data available for validation</p>
          <button
            onClick={() => navigate('/convert')}
            className="mt-4 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
          >
            Go Back to Convert
          </button>
      </div>
      )}
    </div>
  );
};

export default ValidationComponent; 