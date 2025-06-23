import React, { useState, useEffect, useCallback } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import axios from 'axios';
import { toast } from 'react-hot-toast';
import { API_URL } from '../config';
import { FinancialEntry } from '../types/financial';
import {
  ArrowLeftIcon,
  ArrowRightIcon,
  ExclamationTriangleIcon,
  ExclamationCircleIcon,
  CheckCircleIcon,
  EyeIcon,
  EyeSlashIcon,
  DocumentArrowDownIcon,
  MagnifyingGlassIcon,
  AdjustmentsHorizontalIcon,
  XMarkIcon,
  CheckIcon,
  PencilIcon,
  ShieldCheckIcon,
} from '@heroicons/react/24/outline';

// Import modular components
import ValidationCell from './ValidationCell';
import ValidationTooltip from './ValidationTooltip';
import IgnoreButton from './IgnoreButton';
import IssueSummarySidebar from './IssueSummarySidebar';
import LoadingIndicator from './LoadingIndicator';
import PasswordPrompt from './PasswordPrompt';

// Types for validation
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
  cellId: string;
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

interface CellData {
  value: any;
  originalValue: any;
  isEdited: boolean;
  isIgnored: boolean;
  issues: ValidationIssue[];
}

interface TableRow {
  [key: string]: CellData;
}

const ValidationPage: React.FC = () => {
  const location = useLocation();
  const navigate = useNavigate();
  
  // State from navigation
  const { fileId, data: initialData, validationResults, convertedFormats } = location.state || {};
  
  // Component state
  const [loading, setLoading] = useState(true);
  const [tableData, setTableData] = useState<TableRow[]>([]);
  const [originalData, setOriginalData] = useState<FinancialEntry[]>([]);
  const [validationIssues, setValidationIssues] = useState<ValidationIssue[]>([]);
  const [ignoredIssues, setIgnoredIssues] = useState<Set<string>>(new Set());
  const [fixedIssues, setFixedIssues] = useState<Set<string>>(new Set());
  const [validationSummary, setValidationSummary] = useState<ValidationSummary>({
    totalIssues: 0,
    errors: 0,
    warnings: 0,
    ignoredIssues: 0,
    fixedIssues: 0,
    byType: {},
    bySeverity: {}
  });
  
  // UI state
  const [showOnlyIssues, setShowOnlyIssues] = useState(false);
  const [showSidebar, setShowSidebar] = useState(true);
  const [editingCell, setEditingCell] = useState<{rowIndex: number, columnKey: string} | null>(null);
  const [editValue, setEditValue] = useState('');
  const [searchQuery, setSearchQuery] = useState('');
  const [showPasswordPrompt, setShowPasswordPrompt] = useState(false);
  const [isValidating, setIsValidating] = useState(false);
  
  // Column definitions
  const columns = [
    { key: 'date', label: 'Date', type: 'date' },
    { key: 'description', label: 'Description', type: 'text' },
    { key: 'amount', label: 'Amount', type: 'number' },
    { key: 'balance', label: 'Balance', type: 'number' },
    { key: 'debit', label: 'Debit', type: 'number' },
    { key: 'credit', label: 'Credit', type: 'number' },
    { key: 'gstin', label: 'GSTIN', type: 'text' },
    { key: 'voucherNo', label: 'Voucher No.', type: 'text' },
  ];

  // Initialize data and validation
  useEffect(() => {
    if (!fileId) {
      navigate('/convert');
      return;
    }
    initializeValidation();
  }, [fileId]);

  const initializeValidation = async () => {
    setLoading(true);
    try {
      // If we have data from navigation, use it
      if (initialData && validationResults) {
        processInitialData(initialData, validationResults);
      } else {
        // Fetch fresh data from backend
        await fetchValidationData();
      }
    } catch (error) {
      console.error('Error initializing validation:', error);
      toast.error('Failed to load validation data');
    } finally {
      setLoading(false);
    }
  };

  const processInitialData = (data: FinancialEntry[], validation: any) => {
    setOriginalData(data);
    
    // Convert data to table format
    const processedTableData = data.map((row, index) => {
      const tableRow: TableRow = {};
      columns.forEach(column => {
        tableRow[column.key] = {
          value: row[column.key as keyof FinancialEntry] || '',
          originalValue: row[column.key as keyof FinancialEntry] || '',
          isEdited: false,
          isIgnored: false,
          issues: []
        };
      });
      return tableRow;
    });

    setTableData(processedTableData);

    // Process validation results
    if (validation && validation.results) {
      const issues = processValidationResults(validation.results);
      setValidationIssues(issues);
      updateValidationSummary(issues);
      
      // Apply issues to table data
      const updatedTableData = applyIssuesToTableData(processedTableData, issues);
      setTableData(updatedTableData);
    }
  };

  const fetchValidationData = async (password?: string) => {
    try {
      setIsValidating(true);
      
      // First get the parsed data
      const dataResponse = await axios.get(`${API_URL}/file/${fileId}?convert=true`, {
        params: password ? { password } : {}
      });

      if (dataResponse.data && dataResponse.data.data) {
        const parsedData = dataResponse.data.data;
        setOriginalData(parsedData);
        
        // Format data for validation
        const sanitizedData = parsedData.map((row: any) => {
          const sanitizedRow: Record<string, any> = {};
          Object.entries(row).forEach(([key, value]) => {
            if (value === null || value === undefined) {
              sanitizedRow[key] = '';
            } else if (typeof value === 'object' && value.toISOString) {
              sanitizedRow[key] = value.toISOString().split('T')[0];
            } else {
              sanitizedRow[key] = String(value);
            }
          });
          return sanitizedRow;
        });

        // Run validation
        const validationResponse = await axios.post(`${API_URL}/validate/${fileId}`, {
          data: sanitizedData,
          formats: convertedFormats || ['xlsx'],
          password
        });

        if (validationResponse.data?.validationResults) {
          processInitialData(parsedData, validationResponse.data.validationResults);
        } else {
          // No validation results, just show the data
          processInitialData(parsedData, { results: [] });
        }
      }
    } catch (error: any) {
      console.error('Error fetching validation data:', error);
      
      if (error.response?.status === 401 && error.response?.data?.requires_password) {
        setShowPasswordPrompt(true);
      } else {
        toast.error('Failed to load data for validation');
      }
    } finally {
      setIsValidating(false);
    }
  };

  const processValidationResults = (results: any[]): ValidationIssue[] => {
    const issues: ValidationIssue[] = [];
    
    results.forEach((result, resultIndex) => {
      if (result.issues && Array.isArray(result.issues)) {
        result.issues.forEach((issue: any, issueIndex: number) => {
          issues.push({
            id: `${result.row || resultIndex}-${issue.field}-${issueIndex}`,
            rowIndex: result.row || resultIndex,
            columnKey: issue.field,
            message: issue.message || issue.description || 'Validation issue',
            type: issue.type || (issue.severity === 'error' ? 'error' : 'warning'),
            severity: issue.severity || 'moderate',
            suggestedFix: issue.suggestedFix,
            suggestedValue: issue.suggestedValue,
            cellId: `${result.row || resultIndex}-${issue.field}`,
            ignored: false,
            fixed: false
          });
        });
      }
    });
    
    return issues;
  };

  const applyIssuesToTableData = (tableData: TableRow[], issues: ValidationIssue[]): TableRow[] => {
    const updatedData = [...tableData];
    
    issues.forEach(issue => {
      if (updatedData[issue.rowIndex] && updatedData[issue.rowIndex][issue.columnKey]) {
        updatedData[issue.rowIndex][issue.columnKey].issues.push(issue);
      }
    });
    
    return updatedData;
  };

  const updateValidationSummary = (issues: ValidationIssue[]) => {
    const summary: ValidationSummary = {
      totalIssues: issues.length,
      errors: issues.filter(i => i.type === 'error' && !i.ignored).length,
      warnings: issues.filter(i => i.type === 'warning' && !i.ignored).length,
      ignoredIssues: ignoredIssues.size,
      fixedIssues: fixedIssues.size,
      byType: {},
      bySeverity: {}
    };

    // Group by type
    issues.forEach(issue => {
      if (!issue.ignored) {
        summary.byType[issue.type] = (summary.byType[issue.type] || 0) + 1;
        summary.bySeverity[issue.severity] = (summary.bySeverity[issue.severity] || 0) + 1;
      }
    });

    setValidationSummary(summary);
  };

  // Event handlers
  const handleIgnoreIssue = (issueId: string) => {
    const newIgnoredIssues = new Set(ignoredIssues);
    newIgnoredIssues.add(issueId);
    setIgnoredIssues(newIgnoredIssues);

    // Update validation issues
    const updatedIssues = validationIssues.map(issue => 
      issue.id === issueId ? { ...issue, ignored: true } : issue
    );
    setValidationIssues(updatedIssues);
    updateValidationSummary(updatedIssues);

    toast.success('Issue ignored');
  };

  const handleApplyFix = (issueId: string, suggestedValue: any) => {
    const issue = validationIssues.find(i => i.id === issueId);
    if (!issue) return;

    // Update the cell value
    const updatedTableData = [...tableData];
    if (updatedTableData[issue.rowIndex] && updatedTableData[issue.rowIndex][issue.columnKey]) {
      updatedTableData[issue.rowIndex][issue.columnKey] = {
        ...updatedTableData[issue.rowIndex][issue.columnKey],
        value: suggestedValue,
        isEdited: true
      };
    }
    setTableData(updatedTableData);

    // Mark issue as fixed
    const newFixedIssues = new Set(fixedIssues);
    newFixedIssues.add(issueId);
    setFixedIssues(newFixedIssues);

    const updatedIssues = validationIssues.map(i => 
      i.id === issueId ? { ...i, fixed: true } : i
    );
    setValidationIssues(updatedIssues);
    updateValidationSummary(updatedIssues);

    toast.success('Fix applied');
  };

  const handleCellEdit = (rowIndex: number, columnKey: string, newValue: any) => {
    const updatedTableData = [...tableData];
    if (updatedTableData[rowIndex] && updatedTableData[rowIndex][columnKey]) {
      updatedTableData[rowIndex][columnKey] = {
        ...updatedTableData[rowIndex][columnKey],
        value: newValue,
        isEdited: true
      };
    }
    setTableData(updatedTableData);
    setEditingCell(null);
    toast.success('Cell updated');
  };

  const startCellEdit = (rowIndex: number, columnKey: string) => {
    setEditingCell({ rowIndex, columnKey });
    setEditValue(tableData[rowIndex][columnKey].value);
  };

  const saveCellEdit = () => {
    if (editingCell) {
      handleCellEdit(editingCell.rowIndex, editingCell.columnKey, editValue);
    }
  };

  const cancelCellEdit = () => {
    setEditingCell(null);
    setEditValue('');
  };

  const handlePasswordSubmit = (password: string) => {
    setShowPasswordPrompt(false);
    fetchValidationData(password);
  };

  const handleExport = () => {
    // Convert table data back to export format
    const exportData = tableData.map(row => {
      const exportRow: any = {};
      columns.forEach(column => {
        exportRow[column.key] = row[column.key].value;
      });
      return exportRow;
    });

    navigate('/export', {
      state: {
        fileId,
        data: exportData,
        validationSummary,
        convertedFormats: convertedFormats || ['xlsx']
      }
    });
  };

  const handleBack = () => {
    navigate('/convert', {
      state: {
        fileId,
        data: originalData
      }
    });
  };

  // Filter rows based on issues toggle
  const filteredTableData = showOnlyIssues 
    ? tableData.filter((row, index) => 
        columns.some(col => 
          row[col.key].issues.some(issue => !issue.ignored && !issue.fixed)
        )
      )
    : tableData;

  const hasUnresolvedIssues = validationSummary.errors > 0 || 
    (validationSummary.warnings > 0 && validationSummary.warnings > validationSummary.ignoredIssues);

  if (loading || isValidating) {
    return <LoadingIndicator message="Loading validation data..." />;
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-4">
            <div className="flex items-center space-x-4">
              <button
                onClick={handleBack}
                className="flex items-center text-gray-600 hover:text-gray-900 transition-colors"
              >
                <ArrowLeftIcon className="h-5 w-5 mr-2" />
                Back to Convert
              </button>
              <div>
                <h1 className="text-2xl font-bold text-gray-900">Data Validation</h1>
                <p className="text-sm text-gray-500">
                  Review and fix validation issues before export
                </p>
              </div>
            </div>
            
            <div className="flex items-center space-x-3">
              {/* Toggle sidebar */}
              <button
                onClick={() => setShowSidebar(!showSidebar)}
                className="p-2 text-gray-500 hover:text-gray-700 rounded-md hover:bg-gray-100"
                title={showSidebar ? 'Hide sidebar' : 'Show sidebar'}
              >
                <AdjustmentsHorizontalIcon className="h-5 w-5" />
              </button>

              {/* Export button */}
              <button
                onClick={handleExport}
                disabled={hasUnresolvedIssues}
                className={`flex items-center px-4 py-2 rounded-md font-medium transition-colors ${
                  hasUnresolvedIssues
                    ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
                    : 'bg-green-600 text-white hover:bg-green-700'
                }`}
                title={hasUnresolvedIssues ? 'Resolve all critical issues before export' : 'Export data'}
              >
                <DocumentArrowDownIcon className="h-5 w-5 mr-2" />
                Export
              </button>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
        <div className="flex space-x-6">
          {/* Main content */}
          <div className={`flex-1 ${showSidebar ? 'lg:mr-80' : ''}`}>
            {/* Controls */}
            <div className="bg-white rounded-lg shadow-sm border p-4 mb-6">
              <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between space-y-3 sm:space-y-0">
                <div className="flex items-center space-x-4">
                  {/* Search */}
                  <div className="relative">
                    <MagnifyingGlassIcon className="h-5 w-5 absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" />
                    <input
                      type="text"
                      placeholder="Search data..."
                      value={searchQuery}
                      onChange={(e) => setSearchQuery(e.target.value)}
                      className="pl-10 pr-4 py-2 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500"
                    />
                  </div>

                  {/* Show only issues toggle */}
                  <button
                    onClick={() => setShowOnlyIssues(!showOnlyIssues)}
                    className={`flex items-center px-3 py-2 rounded-md text-sm font-medium transition-colors ${
                      showOnlyIssues
                        ? 'bg-blue-100 text-blue-700 border border-blue-300'
                        : 'bg-white text-gray-700 border border-gray-300 hover:bg-gray-50'
                    }`}
                  >
                    {showOnlyIssues ? <EyeSlashIcon className="h-4 w-4 mr-2" /> : <EyeIcon className="h-4 w-4 mr-2" />}
                    {showOnlyIssues ? 'Show All Rows' : 'Show Only Issues'}
                  </button>
                </div>

                {/* Summary badges */}
                <div className="flex items-center space-x-3">
                  <div className="flex items-center text-sm">
                    <div className="flex items-center space-x-2">
                      {validationSummary.errors > 0 && (
                        <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-red-100 text-red-800">
                          <ExclamationCircleIcon className="h-3 w-3 mr-1" />
                          {validationSummary.errors} Error{validationSummary.errors !== 1 ? 's' : ''}
                        </span>
                      )}
                      {validationSummary.warnings > 0 && (
                        <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-yellow-100 text-yellow-800">
                          <ExclamationTriangleIcon className="h-3 w-3 mr-1" />
                          {validationSummary.warnings} Warning{validationSummary.warnings !== 1 ? 's' : ''}
                        </span>
                      )}
                      {validationSummary.totalIssues === 0 && (
                        <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-green-100 text-green-800">
                          <CheckCircleIcon className="h-3 w-3 mr-1" />
                          No Issues
                        </span>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Data table */}
            <div className="bg-white rounded-lg shadow-sm border overflow-hidden">
              <div className="overflow-x-auto">
                <table className="min-w-full divide-y divide-gray-200">
                  <thead className="bg-gray-50">
                    <tr>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Row
                      </th>
                      {columns.map(column => (
                        <th
                          key={column.key}
                          className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider"
                        >
                          {column.label}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody className="bg-white divide-y divide-gray-200">
                    <AnimatePresence>
                      {filteredTableData.map((row, rowIndex) => {
                        const originalRowIndex = showOnlyIssues 
                          ? tableData.findIndex(r => r === row)
                          : rowIndex;
                        
                        return (
                          <motion.tr
                            key={originalRowIndex}
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            exit={{ opacity: 0, y: -20 }}
                            className="hover:bg-gray-50"
                          >
                            <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                              {originalRowIndex + 1}
                            </td>
                            {columns.map(column => {
                              const cellData = row[column.key];
                              const isEditing = editingCell?.rowIndex === originalRowIndex && 
                                               editingCell?.columnKey === column.key;
                              
                              return (
                                <ValidationCell
                                  key={`${originalRowIndex}-${column.key}`}
                                  cellData={cellData}
                                  isEditing={isEditing}
                                  editValue={editValue}
                                  onEditValueChange={setEditValue}
                                  onStartEdit={() => startCellEdit(originalRowIndex, column.key)}
                                  onSaveEdit={saveCellEdit}
                                  onCancelEdit={cancelCellEdit}
                                  onApplyFix={handleApplyFix}
                                  onIgnoreIssue={handleIgnoreIssue}
                                  columnType={column.type}
                                />
                              );
                            })}
                          </motion.tr>
                        );
                      })}
                    </AnimatePresence>
                  </tbody>
                </table>
              </div>

              {filteredTableData.length === 0 && (
                <div className="text-center py-12">
                  <ShieldCheckIcon className="mx-auto h-12 w-12 text-gray-400" />
                  <h3 className="mt-2 text-sm font-medium text-gray-900">
                    {showOnlyIssues ? 'No rows with issues' : 'No data available'}
                  </h3>
                  <p className="mt-1 text-sm text-gray-500">
                    {showOnlyIssues 
                      ? 'All validation issues have been resolved or ignored.'
                      : 'No data was found for validation.'
                    }
                  </p>
                </div>
              )}
            </div>
          </div>

          {/* Sidebar */}
          <AnimatePresence>
            {showSidebar && (
              <IssueSummarySidebar
                validationSummary={validationSummary}
                validationIssues={validationIssues}
                onIgnoreIssue={handleIgnoreIssue}
                onApplyFix={handleApplyFix}
                onJumpToIssue={(rowIndex, columnKey) => {
                  // Scroll to the issue in the table
                  const element = document.querySelector(`[data-cell="${rowIndex}-${columnKey}"]`);
                  element?.scrollIntoView({ behavior: 'smooth', block: 'center' });
                }}
              />
            )}
          </AnimatePresence>
        </div>
      </div>

      {/* Password prompt modal */}
      <AnimatePresence>
        {showPasswordPrompt && (
          <PasswordPrompt
            onSubmit={handlePasswordSubmit}
            onCancel={() => setShowPasswordPrompt(false)}
          />
        )}
      </AnimatePresence>
    </div>
  );
};

export default ValidationPage; 