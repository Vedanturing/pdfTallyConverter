import React, { useState } from 'react';
import { motion } from 'framer-motion';
import {
  ExclamationTriangleIcon,
  ExclamationCircleIcon,
  CheckCircleIcon,
  EyeSlashIcon,
  WrenchIcon,
  ChevronDownIcon,
  ChevronRightIcon,
  ClipboardDocumentListIcon,
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
  ignored?: boolean;
  fixed?: boolean;
}

interface ValidationSummary {
  totalIssues: number;
  errors: number;
  warnings: number;
  ignoredIssues: number;
  fixedIssues: number;
  byType: Record<string, number>;
  bySeverity: Record<string, number>;
}

interface IssueSummarySidebarProps {
  validationSummary: ValidationSummary;
  validationIssues: ValidationIssue[];
  onIgnoreIssue: (issueId: string) => void;
  onApplyFix: (issueId: string, suggestedValue: any) => void;
  onJumpToIssue: (rowIndex: number, columnKey: string) => void;
}

const IssueSummarySidebar: React.FC<IssueSummarySidebarProps> = ({
  validationSummary,
  validationIssues,
  onIgnoreIssue,
  onApplyFix,
  onJumpToIssue
}) => {
  const [expandedSections, setExpandedSections] = useState<Set<string>>(new Set(['errors', 'warnings']));
  const [selectedGrouping, setSelectedGrouping] = useState<'type' | 'severity' | 'column'>('type');

  // Filter active issues (not ignored or fixed)
  const activeIssues = validationIssues.filter(issue => !issue.ignored && !issue.fixed);

  const toggleSection = (section: string) => {
    const newExpanded = new Set(expandedSections);
    if (newExpanded.has(section)) {
      newExpanded.delete(section);
    } else {
      newExpanded.add(section);
    }
    setExpandedSections(newExpanded);
  };

  const groupIssues = () => {
    switch (selectedGrouping) {
      case 'type':
        return {
          'Errors': activeIssues.filter(issue => issue.type === 'error'),
          'Warnings': activeIssues.filter(issue => issue.type === 'warning')
        };
      case 'severity':
        return {
          'Critical': activeIssues.filter(issue => issue.severity === 'critical'),
          'Moderate': activeIssues.filter(issue => issue.severity === 'moderate'),
          'Low': activeIssues.filter(issue => issue.severity === 'low')
        };
      case 'column':
        const byColumn: Record<string, ValidationIssue[]> = {};
        activeIssues.forEach(issue => {
          const column = issue.columnKey.charAt(0).toUpperCase() + issue.columnKey.slice(1);
          if (!byColumn[column]) {
            byColumn[column] = [];
          }
          byColumn[column].push(issue);
        });
        return byColumn;
      default:
        return { 'All Issues': activeIssues };
    }
  };

  const groupedIssues = groupIssues();

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical': return 'text-red-600 bg-red-50';
      case 'moderate': return 'text-yellow-600 bg-yellow-50';
      case 'low': return 'text-blue-600 bg-blue-50';
      default: return 'text-gray-600 bg-gray-50';
    }
  };

  const getTypeIcon = (type: string) => {
    return type === 'error' ? (
      <ExclamationCircleIcon className="h-4 w-4 text-red-500" />
    ) : (
      <ExclamationTriangleIcon className="h-4 w-4 text-yellow-500" />
    );
  };

  return (
    <motion.div
      initial={{ x: 300, opacity: 0 }}
      animate={{ x: 0, opacity: 1 }}
      exit={{ x: 300, opacity: 0 }}
      transition={{ duration: 0.3 }}
      className="fixed right-0 top-0 h-full w-80 bg-white border-l border-gray-200 shadow-xl z-40 overflow-hidden flex flex-col"
    >
      {/* Header */}
      <div className="p-4 border-b border-gray-200 bg-gray-50">
        <div className="flex items-center space-x-2 mb-3">
          <ClipboardDocumentListIcon className="h-5 w-5 text-gray-600" />
          <h2 className="text-lg font-semibold text-gray-900">Issue Summary</h2>
        </div>

        {/* Summary stats */}
        <div className="grid grid-cols-2 gap-2 text-sm">
          <div className="bg-white rounded-lg p-2 border">
            <div className="flex items-center justify-between">
              <span className="text-gray-600">Total Issues</span>
              <span className="font-semibold text-gray-900">{validationSummary.totalIssues}</span>
            </div>
          </div>
          <div className="bg-white rounded-lg p-2 border">
            <div className="flex items-center justify-between">
              <span className="text-gray-600">Active</span>
              <span className="font-semibold text-gray-900">{activeIssues.length}</span>
            </div>
          </div>
          <div className="bg-red-50 rounded-lg p-2 border border-red-200">
            <div className="flex items-center justify-between">
              <span className="text-red-600">Errors</span>
              <span className="font-semibold text-red-700">{validationSummary.errors}</span>
            </div>
          </div>
          <div className="bg-yellow-50 rounded-lg p-2 border border-yellow-200">
            <div className="flex items-center justify-between">
              <span className="text-yellow-600">Warnings</span>
              <span className="font-semibold text-yellow-700">{validationSummary.warnings}</span>
            </div>
          </div>
        </div>

        {/* Status indicator */}
        <div className="mt-3">
          {validationSummary.errors === 0 && validationSummary.warnings === 0 ? (
            <div className="flex items-center space-x-2 text-green-700 bg-green-50 rounded-lg p-2">
              <CheckCircleIcon className="h-4 w-4" />
              <span className="text-sm font-medium">All issues resolved!</span>
            </div>
          ) : validationSummary.errors > 0 ? (
            <div className="flex items-center space-x-2 text-red-700 bg-red-50 rounded-lg p-2">
              <ExclamationCircleIcon className="h-4 w-4" />
              <span className="text-sm font-medium">Critical issues need attention</span>
            </div>
          ) : (
            <div className="flex items-center space-x-2 text-yellow-700 bg-yellow-50 rounded-lg p-2">
              <ExclamationTriangleIcon className="h-4 w-4" />
              <span className="text-sm font-medium">Warnings detected</span>
            </div>
          )}
        </div>
      </div>

      {/* Controls */}
      <div className="p-4 border-b border-gray-200">
        <label className="block text-sm font-medium text-gray-700 mb-2">Group by:</label>
        <select
          value={selectedGrouping}
          onChange={(e) => setSelectedGrouping(e.target.value as 'type' | 'severity' | 'column')}
          className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500 text-sm"
        >
          <option value="type">Issue Type</option>
          <option value="severity">Severity</option>
          <option value="column">Column</option>
        </select>
      </div>

      {/* Issues list */}
      <div className="flex-1 overflow-y-auto">
        {Object.entries(groupedIssues).map(([groupName, issues]) => (
          <div key={groupName} className="border-b border-gray-200">
            <button
              onClick={() => toggleSection(groupName.toLowerCase())}
              className="w-full px-4 py-3 text-left flex items-center justify-between hover:bg-gray-50 transition-colors"
            >
              <div className="flex items-center space-x-2">
                {selectedGrouping === 'type' && getTypeIcon(groupName.toLowerCase())}
                <span className="font-medium text-gray-900">{groupName}</span>
                <span className="px-2 py-0.5 bg-gray-100 text-gray-600 text-xs rounded-full">
                  {issues.length}
                </span>
              </div>
              {expandedSections.has(groupName.toLowerCase()) ? (
                <ChevronDownIcon className="h-4 w-4 text-gray-500" />
              ) : (
                <ChevronRightIcon className="h-4 w-4 text-gray-500" />
              )}
            </button>

            {expandedSections.has(groupName.toLowerCase()) && (
              <motion.div
                initial={{ height: 0, opacity: 0 }}
                animate={{ height: 'auto', opacity: 1 }}
                exit={{ height: 0, opacity: 0 }}
                transition={{ duration: 0.2 }}
                className="bg-gray-50"
              >
                {issues.length === 0 ? (
                  <div className="px-4 py-3 text-sm text-gray-500 text-center">
                    No issues in this category
                  </div>
                ) : (
                  <div className="space-y-1 p-2">
                    {issues.map((issue) => (
                      <div
                        key={issue.id}
                        className="bg-white rounded-md border border-gray-200 p-3 hover:shadow-sm transition-shadow"
                      >
                        <div className="flex items-start justify-between mb-2">
                          <div className="flex-1">
                            <div className="flex items-center space-x-2 mb-1">
                              <button
                                onClick={() => onJumpToIssue(issue.rowIndex, issue.columnKey)}
                                className="text-xs font-medium text-blue-600 hover:text-blue-800 hover:underline"
                              >
                                Row {issue.rowIndex + 1}, {issue.columnKey}
                              </button>
                              <span className={`px-1.5 py-0.5 rounded-full text-xs font-medium ${getSeverityColor(issue.severity)}`}>
                                {issue.severity}
                              </span>
                            </div>
                            <p className="text-xs text-gray-700 mb-2">{issue.message}</p>
                            
                            {issue.suggestedValue !== undefined && (
                              <div className="bg-green-50 border border-green-200 rounded p-2 mb-2">
                                <p className="text-xs text-green-700 mb-1">Suggested:</p>
                                <p className="text-xs font-mono text-green-800">
                                  {String(issue.suggestedValue)}
                                </p>
                              </div>
                            )}
                          </div>
                        </div>

                        <div className="flex items-center space-x-2">
                          {issue.suggestedValue !== undefined && (
                            <button
                              onClick={() => onApplyFix(issue.id, issue.suggestedValue)}
                              className="inline-flex items-center px-2 py-1 border border-green-300 text-xs font-medium rounded text-green-700 bg-green-50 hover:bg-green-100"
                            >
                              <WrenchIcon className="h-3 w-3 mr-1" />
                              Apply
                            </button>
                          )}
                          <button
                            onClick={() => onIgnoreIssue(issue.id)}
                            className="inline-flex items-center px-2 py-1 border border-gray-300 text-xs font-medium rounded text-gray-700 bg-white hover:bg-gray-50"
                          >
                            <EyeSlashIcon className="h-3 w-3 mr-1" />
                            Ignore
                          </button>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </motion.div>
            )}
          </div>
        ))}
      </div>

      {/* Footer actions */}
      <div className="p-4 border-t border-gray-200 bg-gray-50">
        <div className="space-y-2">
          <button
            onClick={() => {
              activeIssues.forEach(issue => {
                if (issue.suggestedValue !== undefined) {
                  onApplyFix(issue.id, issue.suggestedValue);
                }
              });
            }}
            disabled={!activeIssues.some(i => i.suggestedValue !== undefined)}
            className="w-full px-3 py-2 bg-green-600 text-white text-sm font-medium rounded-md hover:bg-green-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors"
          >
            Apply All Fixes
          </button>
          <button
            onClick={() => {
              activeIssues.forEach(issue => onIgnoreIssue(issue.id));
            }}
            disabled={activeIssues.length === 0}
            className="w-full px-3 py-2 bg-gray-600 text-white text-sm font-medium rounded-md hover:bg-gray-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors"
          >
            Ignore All Issues
          </button>
        </div>
      </div>
    </motion.div>
  );
};

export default IssueSummarySidebar; 