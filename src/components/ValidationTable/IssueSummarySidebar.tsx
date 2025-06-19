import React from 'react';
import { ValidationIssue } from './ValidationCell';
import {
  ExclamationCircleIcon,
  ExclamationTriangleIcon,
  CheckCircleIcon,
  XMarkIcon,
} from '@heroicons/react/24/outline';

interface IssueSummaryProps {
  issues: {
    rowIndex: number;
    field: string;
    issue: ValidationIssue;
    isIgnored: boolean;
  }[];
  onScrollToIssue: (rowIndex: number, field: string) => void;
  onIgnoreIssue: (rowIndex: number, field: string) => void;
  onApplyFix: (rowIndex: number, field: string, fix: string) => void;
}

export const IssueSummarySidebar: React.FC<IssueSummaryProps> = ({
  issues,
  onScrollToIssue,
  onIgnoreIssue,
  onApplyFix,
}) => {
  const groupedIssues = issues.reduce((acc, curr) => {
    const key = curr.issue.type;
    if (!acc[key]) acc[key] = [];
    acc[key].push(curr);
    return acc;
  }, {} as Record<string, typeof issues>);

  const issueTypeTitles: Record<string, string> = {
    amount: 'Amount Issues',
    date: 'Date Format Issues',
    gstin: 'GSTIN Issues',
    tax: 'Tax Rate Issues',
    voucher: 'Voucher Number Issues',
    other: 'Other Issues',
  };

  const getIssueTypeIcon = (type: string) => {
    switch (type) {
      case 'amount':
        return <ExclamationCircleIcon className="h-5 w-5 text-red-500" />;
      case 'date':
        return <ExclamationCircleIcon className="h-5 w-5 text-yellow-500" />;
      case 'gstin':
        return <ExclamationTriangleIcon className="h-5 w-5 text-orange-500" />;
      case 'tax':
        return <ExclamationTriangleIcon className="h-5 w-5 text-yellow-500" />;
      case 'voucher':
        return <ExclamationCircleIcon className="h-5 w-5 text-blue-500" />;
      default:
        return <ExclamationCircleIcon className="h-5 w-5 text-gray-500" />;
    }
  };

  const stats = {
    total: issues.length,
    errors: issues.filter((i) => i.issue.severity === 'error').length,
    warnings: issues.filter((i) => i.issue.severity === 'warning').length,
    ignored: issues.filter((i) => i.isIgnored).length,
    fixable: issues.filter((i) => i.issue.suggestedFix).length,
  };

  return (
    <div className="w-80 bg-white border-l border-gray-200 overflow-y-auto h-full">
      <div className="p-4 border-b border-gray-200">
        <h3 className="text-lg font-medium text-gray-900">Validation Summary</h3>
        <div className="mt-2 grid grid-cols-2 gap-2 text-sm">
          <div className="flex items-center">
            <ExclamationCircleIcon className="h-4 w-4 text-red-500 mr-1" />
            <span>{stats.errors} Errors</span>
          </div>
          <div className="flex items-center">
            <ExclamationTriangleIcon className="h-4 w-4 text-yellow-500 mr-1" />
            <span>{stats.warnings} Warnings</span>
          </div>
          <div className="flex items-center">
            <CheckCircleIcon className="h-4 w-4 text-gray-400 mr-1" />
            <span>{stats.ignored} Ignored</span>
          </div>
          <div className="flex items-center">
            <XMarkIcon className="h-4 w-4 text-green-500 mr-1" />
            <span>{stats.fixable} Fixable</span>
          </div>
        </div>
      </div>

      <div className="p-4 space-y-6">
        {Object.entries(groupedIssues).map(([type, typeIssues]) => (
          <div key={type}>
            <div className="flex items-center mb-2">
              {getIssueTypeIcon(type)}
              <h4 className="ml-2 font-medium text-gray-900">
                {issueTypeTitles[type] || type}
              </h4>
            </div>
            <div className="space-y-2">
              {typeIssues.map((issue, idx) => (
                <div
                  key={`${issue.rowIndex}-${issue.field}-${idx}`}
                  className={`p-2 rounded-lg text-sm ${
                    issue.isIgnored
                      ? 'bg-gray-50'
                      : issue.issue.severity === 'error'
                      ? 'bg-red-50'
                      : 'bg-yellow-50'
                  }`}
                >
                  <div className="flex justify-between items-start">
                    <button
                      onClick={() => onScrollToIssue(issue.rowIndex, issue.field)}
                      className="text-left hover:underline"
                    >
                      <span className="font-medium">
                        Row {issue.rowIndex + 1}, {issue.field}
                      </span>
                      <p className="text-gray-600 mt-1">{issue.issue.message}</p>
                    </button>
                    {!issue.isIgnored && (
                      <button
                        onClick={() => onIgnoreIssue(issue.rowIndex, issue.field)}
                        className="text-gray-400 hover:text-gray-600"
                      >
                        <XMarkIcon className="h-4 w-4" />
                      </button>
                    )}
                  </div>
                  {issue.issue.suggestedFix && !issue.isIgnored && (
                    <button
                      onClick={() =>
                        onApplyFix(
                          issue.rowIndex,
                          issue.field,
                          issue.issue.suggestedFix!
                        )
                      }
                      className="mt-2 px-2 py-1 text-xs bg-green-100 text-green-800 rounded hover:bg-green-200"
                    >
                      Apply Fix
                    </button>
                  )}
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}; 