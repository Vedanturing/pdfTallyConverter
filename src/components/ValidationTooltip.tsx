import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  ExclamationTriangleIcon,
  ExclamationCircleIcon,
  CheckIcon,
  EyeSlashIcon,
  WrenchIcon,
} from '@heroicons/react/24/outline';

interface ValidationIssue {
  id: string;
  message: string;
  type: 'error' | 'warning';
  severity: 'critical' | 'moderate' | 'low';
  suggestedFix?: string;
  suggestedValue?: any;
  ignored?: boolean;
  fixed?: boolean;
}

interface ValidationTooltipProps {
  issues: ValidationIssue[];
  onApplyFix: (issueId: string, suggestedValue: any) => void;
  onIgnore: (issueId: string) => void;
}

const ValidationTooltip: React.FC<ValidationTooltipProps> = ({
  issues,
  onApplyFix,
  onIgnore
}) => {
  const [isHovered, setIsHovered] = useState(false);

  if (issues.length === 0) return null;

  return (
    <div
      className="absolute inset-0 z-10"
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      <AnimatePresence>
        {isHovered && (
          <motion.div
            initial={{ opacity: 0, scale: 0.95, y: 10 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.95, y: 10 }}
            transition={{ duration: 0.15 }}
            className="absolute bottom-full left-1/2 transform -translate-x-1/2 mb-2 z-50"
          >
            <div className="bg-white rounded-lg shadow-xl border border-gray-200 p-4 min-w-[300px] max-w-[400px]">
              {/* Arrow */}
              <div className="absolute top-full left-1/2 transform -translate-x-1/2 -mt-1">
                <div className="w-3 h-3 bg-white border-r border-b border-gray-200 transform rotate-45"></div>
              </div>

              {/* Header */}
              <div className="flex items-center space-x-2 mb-3">
                {issues.some(i => i.type === 'error') ? (
                  <ExclamationCircleIcon className="h-5 w-5 text-red-500" />
                ) : (
                  <ExclamationTriangleIcon className="h-5 w-5 text-yellow-500" />
                )}
                <h4 className="font-medium text-gray-900">
                  {issues.length === 1 ? 'Validation Issue' : `${issues.length} Validation Issues`}
                </h4>
              </div>

              {/* Issues list */}
              <div className="space-y-3">
                {issues.map((issue, index) => (
                  <div key={issue.id} className="border-l-2 border-gray-200 pl-3">
                    <div className="flex items-start justify-between">
                      <div className="flex-1">
                        <div className="flex items-center space-x-2 mb-1">
                          <span className={`inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium ${
                            issue.type === 'error' 
                              ? 'bg-red-100 text-red-800' 
                              : 'bg-yellow-100 text-yellow-800'
                          }`}>
                            {issue.type === 'error' ? 'Error' : 'Warning'}
                          </span>
                          {issue.severity && (
                            <span className="text-xs text-gray-500 capitalize">
                              {issue.severity}
                            </span>
                          )}
                        </div>
                        
                        <p className="text-sm text-gray-700 mb-2">
                          {issue.message}
                        </p>

                        {issue.suggestedFix && (
                          <p className="text-xs text-gray-600 italic mb-2">
                            💡 {issue.suggestedFix}
                          </p>
                        )}

                        {issue.suggestedValue !== undefined && (
                          <div className="bg-green-50 border border-green-200 rounded-md p-2 mb-2">
                            <p className="text-xs text-green-700 mb-1">Suggested value:</p>
                            <p className="text-sm font-mono text-green-800">
                              {String(issue.suggestedValue)}
                            </p>
                          </div>
                        )}
                      </div>
                    </div>

                    {/* Action buttons */}
                    <div className="flex items-center space-x-2 mt-2">
                      {issue.suggestedValue !== undefined && (
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            onApplyFix(issue.id, issue.suggestedValue);
                          }}
                          className="inline-flex items-center px-2 py-1 border border-green-300 text-xs font-medium rounded text-green-700 bg-green-50 hover:bg-green-100 focus:outline-none focus:ring-2 focus:ring-green-500 focus:ring-offset-2"
                        >
                          <WrenchIcon className="h-3 w-3 mr-1" />
                          Apply Fix
                        </button>
                      )}
                      
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          onIgnore(issue.id);
                        }}
                        className="inline-flex items-center px-2 py-1 border border-gray-300 text-xs font-medium rounded text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2"
                      >
                        <EyeSlashIcon className="h-3 w-3 mr-1" />
                        Ignore
                      </button>
                    </div>
                  </div>
                ))}
              </div>

              {issues.length > 1 && (
                <div className="mt-4 pt-3 border-t border-gray-200">
                  <div className="flex items-center justify-between">
                    <span className="text-xs text-gray-500">
                      {issues.filter(i => i.type === 'error').length} errors, {issues.filter(i => i.type === 'warning').length} warnings
                    </span>
                    <div className="flex space-x-2">
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          issues.forEach(issue => {
                            if (issue.suggestedValue !== undefined) {
                              onApplyFix(issue.id, issue.suggestedValue);
                            }
                          });
                        }}
                        disabled={!issues.some(i => i.suggestedValue !== undefined)}
                        className="text-xs text-green-600 hover:text-green-800 disabled:text-gray-400 font-medium"
                      >
                        Fix All
                      </button>
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          issues.forEach(issue => onIgnore(issue.id));
                        }}
                        className="text-xs text-gray-600 hover:text-gray-800 font-medium"
                      >
                        Ignore All
                      </button>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default ValidationTooltip; 