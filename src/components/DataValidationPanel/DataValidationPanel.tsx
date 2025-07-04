import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  ExclamationTriangleIcon,
  CheckCircleIcon,
  XCircleIcon,
  EyeIcon,
  EyeSlashIcon,
} from '@heroicons/react/24/outline';

interface ValidationIssue {
  id: string;
  rowIndex: number;
  columnKey: string;
  message: string;
  type: 'error' | 'warning';
  severity: 'critical' | 'moderate' | 'low';
  suggestedFix?: string;
  suggestedValue?: any;
}

interface DataValidationPanelProps {
  data: any[];
  issues: ValidationIssue[];
  onFixIssue: (issueId: string, value?: any) => void;
  onIgnoreIssue: (issueId: string) => void;
  showOnlyIssues: boolean;
  onToggleShowOnlyIssues: (show: boolean) => void;
}

const DataValidationPanel: React.FC<DataValidationPanelProps> = ({
  data,
  issues,
  onFixIssue,
  onIgnoreIssue,
  showOnlyIssues,
  onToggleShowOnlyIssues,
}) => {
  const [filter, setFilter] = useState<'all' | 'error' | 'warning'>('all');

  const filteredIssues = issues.filter(issue => {
    if (filter === 'all') return true;
    return issue.type === filter;
  });

  const criticalIssues = issues.filter(issue => issue.severity === 'critical').length;
  const moderateIssues = issues.filter(issue => issue.severity === 'moderate').length;
  const lowIssues = issues.filter(issue => issue.severity === 'low').length;

  return (
    <div className="bg-white rounded-lg shadow-lg border border-gray-200">
      <div className="flex items-center justify-between p-4 border-b border-gray-200">
        <div className="flex items-center space-x-4">
          <h3 className="text-lg font-semibold text-gray-900">Data Validation</h3>
          <div className="flex items-center space-x-2">
            {criticalIssues > 0 && (
              <span className="px-2 py-1 bg-red-100 text-red-800 text-xs rounded-full">
                {criticalIssues} Critical
              </span>
            )}
            {moderateIssues > 0 && (
              <span className="px-2 py-1 bg-yellow-100 text-yellow-800 text-xs rounded-full">
                {moderateIssues} Moderate
              </span>
            )}
            {lowIssues > 0 && (
              <span className="px-2 py-1 bg-blue-100 text-blue-800 text-xs rounded-full">
                {lowIssues} Low
              </span>
            )}
          </div>
        </div>
        <button
          onClick={() => onToggleShowOnlyIssues(!showOnlyIssues)}
          className={`flex items-center px-3 py-2 rounded-md text-sm ${
            showOnlyIssues ? 'bg-blue-100 text-blue-700' : 'bg-gray-100 text-gray-700'
          }`}
        >
          {showOnlyIssues ? <EyeSlashIcon className="h-4 w-4 mr-2" /> : <EyeIcon className="h-4 w-4 mr-2" />}
          {showOnlyIssues ? 'Show All' : 'Issues Only'}
        </button>
      </div>

      <div className="max-h-96 overflow-y-auto">
        <AnimatePresence>
          {filteredIssues.length > 0 ? (
            filteredIssues.map((issue) => (
              <motion.div
                key={issue.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                className="p-4 border-b border-gray-100 hover:bg-gray-50"
              >
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <div className="flex items-center space-x-2 mb-2">
                      {issue.type === 'error' ? (
                        <XCircleIcon className="h-5 w-5 text-red-500" />
                      ) : (
                        <ExclamationTriangleIcon className="h-5 w-5 text-yellow-500" />
                      )}
                      <span className="text-sm font-medium text-gray-900">
                        Row {issue.rowIndex + 1}, Column: {issue.columnKey}
                      </span>
                    </div>
                    <p className="text-sm text-gray-700">{issue.message}</p>
                  </div>
                  <div className="flex items-center space-x-2 ml-4">
                    {issue.suggestedValue && (
                      <button
                        onClick={() => onFixIssue(issue.id, issue.suggestedValue)}
                        className="px-3 py-1 bg-green-600 text-white text-xs rounded hover:bg-green-700"
                      >
                        Apply Fix
                      </button>
                    )}
                    <button
                      onClick={() => onIgnoreIssue(issue.id)}
                      className="px-3 py-1 bg-gray-300 text-gray-700 text-xs rounded hover:bg-gray-400"
                    >
                      Ignore
                    </button>
                  </div>
                </div>
              </motion.div>
            ))
          ) : (
            <div className="p-8 text-center">
              <CheckCircleIcon className="h-12 w-12 text-green-500 mx-auto mb-4" />
              <h4 className="text-lg font-medium text-gray-900 mb-2">All Clear!</h4>
              <p className="text-gray-600">No validation issues found.</p>
            </div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
};

export default DataValidationPanel; 