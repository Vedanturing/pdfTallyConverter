import React from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { ValidationModule } from './ValidationModule';
import { ArrowLeftIcon, ArrowDownTrayIcon, DocumentTextIcon } from '@heroicons/react/24/outline';
import axios from 'axios';
import toast from 'react-hot-toast';
import { API_URL } from '../config';

export default function ValidationComponent() {
  const { t } = useTranslation();
  const location = useLocation();
  const navigate = useNavigate();
  const fileId = location.state?.fileId;
  const data = location.state?.data;
  const validationResults = location.state?.validationResults;
  const convertedFormats = location.state?.convertedFormats;

  const handleExport = () => {
    navigate('/export', {
      state: {
        fileId,
        data,
        convertedFormats
      }
    });
  };

  const handleBack = () => {
    navigate('/convert', {
      state: {
        fileId,
        data
      }
    });
  };

  const handleGenerateGSTReports = async () => {
    if (!fileId) {
      toast.error('No file selected');
      return;
    }

    const toastId = toast.loading('Generating GST reports...');
    
    try {
      const response = await axios.post(`${API_URL}/gst/generate-reports/${fileId}`, {
        period: new Date().toISOString().slice(0, 7) // YYYY-MM format
      });

      if (response.data?.status === 'success') {
        toast.success('GST reports generated successfully!', { id: toastId });
        
        // Optionally navigate to a GST reports page or show results
        console.log('GST Report Data:', response.data);
      } else {
        throw new Error(response.data?.message || 'Failed to generate GST reports');
      }
    } catch (error: any) {
      console.error('GST report generation error:', error);
      const errorMessage = error.response?.data?.detail || error.message || 'Failed to generate GST reports';
      toast.error(errorMessage, { id: toastId });
    }
  };

  if (!fileId) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-center">
          <p className="text-gray-500 dark:text-gray-400 mb-4">No file selected for validation</p>
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
                <h1 className="text-2xl font-bold text-gray-900 dark:text-gray-100">{t('validation.title')}</h1>
                <p className="text-sm text-gray-500 dark:text-gray-400">
                  Review and fix validation issues before export
                </p>
              </div>
            </div>
            
            <div className="flex items-center space-x-3">
              <button
                onClick={handleGenerateGSTReports}
                className="flex items-center px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2"
              >
                <DocumentTextIcon className="h-5 w-5 mr-2" />
                Generate GST Reports
              </button>
              
              <button
                onClick={handleExport}
                className="flex items-center px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-green-500 focus:ring-offset-2"
              >
                <ArrowDownTrayIcon className="h-5 w-5 mr-2" />
                {t('common.export')}
              </button>
            </div>
          </div>
        </div>
      </div>

      <div className="container mx-auto py-6">
        <ValidationModule 
          fileId={fileId} 
          initialData={data}
          validationResults={validationResults}
        />
      </div>
    </div>
  );
} 