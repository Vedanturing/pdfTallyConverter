import React from 'react';
import {
  ExclamationCircleIcon,
  ExclamationTriangleIcon,
  CheckCircleIcon,
} from '@heroicons/react/24/outline';
import { motion } from 'framer-motion';
import { ValidationSummary as ValidationSummaryType } from '../hooks/useValidation';

interface ValidationSummaryProps {
  summary: ValidationSummaryType;
  onClose?: () => void;
}

const ValidationSummary: React.FC<ValidationSummaryProps> = ({ summary, onClose }) => {
  const { totalErrors, totalWarnings, byField } = summary;

  const getIcon = (type: 'error' | 'warning' | 'success') => {
    switch (type) {
      case 'error':
        return <ExclamationCircleIcon className="h-6 w-6 text-red-500" />;
      case 'warning':
        return <ExclamationTriangleIcon className="h-6 w-6 text-yellow-500" />;
      case 'success':
        return <CheckCircleIcon className="h-6 w-6 text-green-500" />;
    }
  };

  const getStatusType = () => {
    if (totalErrors > 0) return 'error';
    if (totalWarnings > 0) return 'warning';
    return 'success';
  };

  const getStatusMessage = () => {
    if (totalErrors > 0) {
      return `Found ${totalErrors} error${totalErrors > 1 ? 's' : ''}`;
    }
    if (totalWarnings > 0) {
      return `Found ${totalWarnings} warning${totalWarnings > 1 ? 's' : ''}`;
    }
    return 'All data is valid';
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: 20 }}
      className="bg-white rounded-lg shadow-lg p-6 max-w-2xl mx-auto"
    >
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center space-x-3">
          {getIcon(getStatusType())}
          <h2 className="text-xl font-semibold text-gray-900">
            Validation Summary
          </h2>
        </div>
        {onClose && (
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-gray-500"
          >
            <span className="sr-only">Close</span>
            <svg className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        )}
      </div>

      <div className="mb-6">
        <div className={`rounded-md p-4 ${
          getStatusType() === 'error' ? 'bg-red-50' :
          getStatusType() === 'warning' ? 'bg-yellow-50' :
          'bg-green-50'
        }`}>
          <div className="flex">
            <div className="flex-shrink-0">
              {getIcon(getStatusType())}
            </div>
            <div className="ml-3">
              <h3 className={`text-sm font-medium ${
                getStatusType() === 'error' ? 'text-red-800' :
                getStatusType() === 'warning' ? 'text-yellow-800' :
                'text-green-800'
              }`}>
                {getStatusMessage()}
              </h3>
            </div>
          </div>
        </div>
      </div>

      {(totalErrors > 0 || totalWarnings > 0) && (
        <div className="space-y-4">
          {Object.entries(byField).map(([field, { errors, warnings }]) => {
            if (errors === 0 && warnings === 0) return null;
            return (
              <div key={field} className="border-t border-gray-200 pt-4">
                <h4 className="text-sm font-medium text-gray-900 capitalize mb-2">
                  {field.replace(/([A-Z])/g, ' $1').trim()}
                </h4>
                <div className="space-y-2">
                  {errors > 0 && (
                    <div className="flex items-center text-sm text-red-600">
                      <ExclamationCircleIcon className="h-5 w-5 mr-2" />
                      {errors} error{errors > 1 ? 's' : ''}
                    </div>
                  )}
                  {warnings > 0 && (
                    <div className="flex items-center text-sm text-yellow-600">
                      <ExclamationTriangleIcon className="h-5 w-5 mr-2" />
                      {warnings} warning{warnings > 1 ? 's' : ''}
                    </div>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      )}

      <div className="mt-6 flex justify-end space-x-3">
        {onClose && (
          <button
            onClick={onClose}
            className="px-4 py-2 bg-white border border-gray-300 rounded-md text-sm font-medium text-gray-700 hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
          >
            Close
          </button>
        )}
      </div>
    </motion.div>
  );
};

export default ValidationSummary; 