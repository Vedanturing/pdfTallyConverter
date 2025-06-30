import React, { useState, useEffect, useCallback } from 'react';
import { PlusIcon, TrashIcon, PencilIcon, CheckIcon, XMarkIcon, ChevronLeftIcon, ChevronRightIcon } from '@heroicons/react/24/outline';
import toast from 'react-hot-toast';

interface ValidationTableProps {
  data: any[];
  onDataChange?: (newData: any[]) => void;
  readOnly?: boolean;
  fileId?: string;
}

interface EditingCell {
  rowIndex: number;
  columnKey: string;
}

const ValidationTable: React.FC<ValidationTableProps> = ({ 
  data, 
  onDataChange, 
  readOnly = false,
  fileId 
}) => {
  const [tableData, setTableData] = useState<any[]>(data || []);
  const [editingCell, setEditingCell] = useState<EditingCell | null>(null);
  const [editValue, setEditValue] = useState<string>('');
  const [isEditMode, setIsEditMode] = useState(false);
  const [columnsToRemove, setColumnsToRemove] = useState<Set<string>>(new Set());
  
  // Pagination state
  const [currentPage, setCurrentPage] = useState(1);
  const [itemsPerPage] = useState(20);

  // Calculate pagination
  const indexOfLastItem = currentPage * itemsPerPage;
  const indexOfFirstItem = indexOfLastItem - itemsPerPage;
  const currentItems = tableData.slice(indexOfFirstItem, indexOfLastItem);
  const totalPages = Math.ceil(tableData.length / itemsPerPage);

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

  useEffect(() => {
    // Filter irrelevant columns before setting data
    const filteredData = filterIrrelevantColumns(data || []);
    setTableData(filteredData);
    setCurrentPage(1); // Reset to first page when data changes
  }, [data]);

  const detectAmountColumns = useCallback((headers: string[]): string[] => {
    const amountKeywords = [
      'amount', 'amt', 'balance', 'total', 'sum', 'value', 'money',
      'debit', 'credit', 'withdrawal', 'deposit', 'rupees', 'rs', 'inr',
      'price', 'cost', 'fee', 'charge'
    ];
    
    return headers.filter(header => 
      amountKeywords.some(keyword => 
        header.toLowerCase().includes(keyword)
      )
    );
  }, []);

  const parseAmount = (value: any): number => {
    if (!value) return 0;
    
    // Convert to string and handle different formats
    let str_value = String(value).trim();
    
    if (!str_value || str_value === '-' || str_value.toLowerCase() === 'nil') {
      return 0;
    }
    
    // Handle negative values in parentheses
    let isNegative = false;
    if (str_value.startsWith('(') && str_value.endsWith(')')) {
      isNegative = true;
      str_value = str_value.slice(1, -1);
    } else if (str_value.startsWith('-')) {
      isNegative = true;
    }
    
    // Remove currency symbols and formatting
    const cleaned = str_value
      .replace(/[₹$€£¥Rs.,\s]/g, '')
      .replace(/[^\d.-]/g, '')
      .trim();
    
    if (!cleaned) return 0;
    
    try {
      const parsed = parseFloat(cleaned);
      return isNaN(parsed) ? 0 : (isNegative ? -parsed : parsed);
    } catch {
      return 0;
    }
  };

  const formatValue = (value: any, header: string) => {
    const amountColumns = detectAmountColumns([header]);
    
    if (amountColumns.includes(header)) {
      const numValue = parseAmount(value);
      if (numValue === 0 && (!value || value === '' || value === '0')) {
        return '0.00';
      }
      return numValue.toLocaleString('en-IN', {
        maximumFractionDigits: 2,
        minimumFractionDigits: 2
      });
    }
    return value || '';
  };

  const getCellClassName = (value: any, header: string, rowIndex: number, colKey: string) => {
    const baseClasses = 'px-4 py-2 text-sm border border-gray-200 relative';
    const isEditing = editingCell?.rowIndex === rowIndex && editingCell?.columnKey === colKey;
    
    if (isEditing) {
      return `${baseClasses} bg-blue-50 border-blue-300`;
    }
    
    const amountColumns = detectAmountColumns([header]);
    if (amountColumns.includes(header)) {
      const numValue = parseAmount(value);
      if (numValue === 0 && (!value || value === '' || value === '0')) {
        return `${baseClasses} bg-red-50 text-red-600`;
      }
      return `${baseClasses} text-right font-mono text-green-600 bg-green-50`;
    }
    
    if (!value || value === '') {
      return `${baseClasses} bg-gray-50 text-gray-400`;
    }
    
    return `${baseClasses} hover:bg-gray-50 cursor-pointer`;
  };

  const handleCellClick = (rowIndex: number, columnKey: string) => {
    if (readOnly || !isEditMode) return;
    
    setEditingCell({ rowIndex, columnKey });
    const currentValue = tableData[rowIndex][columnKey];
    setEditValue(String(currentValue || ''));
  };

  const handleCellSave = async () => {
    if (!editingCell) return;
    
    const newData = [...tableData];
    const { rowIndex, columnKey } = editingCell;
    
    // Parse amount if it's an amount column
    const amountColumns = detectAmountColumns([columnKey]);
    let finalValue = editValue;
    
    if (amountColumns.includes(columnKey)) {
      const numValue = parseAmount(editValue);
      finalValue = numValue;
    }
    
    newData[rowIndex][columnKey] = finalValue;
    
    setTableData(newData);
    setEditingCell(null);
    setEditValue('');
    
    if (onDataChange) {
      onDataChange(newData);
    }
    
    toast.success('Cell updated successfully');
  };

  const handleCellCancel = () => {
    setEditingCell(null);
    setEditValue('');
  };

  const addNewRow = () => {
    if (readOnly || tableData.length === 0) return;
    
    const headers = Object.keys(tableData[0]);
    const newRow: any = {};
    
    headers.forEach(header => {
      const amountColumns = detectAmountColumns([header]);
      if (amountColumns.includes(header)) {
        newRow[header] = 0;
      } else if (header.toLowerCase().includes('date')) {
        newRow[header] = new Date().toISOString().split('T')[0]; // Today's date
      } else {
        newRow[header] = '';
      }
    });
    
    const newData = [...tableData, newRow];
    setTableData(newData);
    
    // Move to the last page to show the new row
    const newTotalPages = Math.ceil(newData.length / itemsPerPage);
    setCurrentPage(newTotalPages);
    
    if (onDataChange) {
      onDataChange(newData);
    }
    
    toast.success('New row added successfully');
  };

  const removeRow = (rowIndex: number) => {
    if (readOnly) return;
    
    // Confirm deletion for safety
    if (!confirm('Are you sure you want to delete this row?')) {
      return;
    }
    
    const newData = tableData.filter((_, index) => index !== rowIndex);
    setTableData(newData);
    
    // Adjust current page if necessary
    const newTotalPages = Math.ceil(newData.length / itemsPerPage);
    if (currentPage > newTotalPages && newTotalPages > 0) {
      setCurrentPage(newTotalPages);
    }
    
    if (onDataChange) {
      onDataChange(newData);
    }
    
    toast.success('Row deleted successfully');
  };

  const addNewColumn = () => {
    if (readOnly || tableData.length === 0) return;
    
    const columnName = prompt('Enter new column name:');
    if (!columnName || columnName.trim() === '') {
      toast.error('Column name cannot be empty');
      return;
    }
    
    const cleanColumnName = columnName.trim();
    
    // Check if column already exists
    const existingHeaders = Object.keys(tableData[0]);
    if (existingHeaders.some(header => 
      header.toLowerCase() === cleanColumnName.toLowerCase()
    )) {
      toast.error('Column with this name already exists');
      return;
    }
    
    // Determine default value based on column name
    let defaultValue = '';
    const columnLower = cleanColumnName.toLowerCase();
    if (columnLower.includes('amount') || columnLower.includes('balance') || 
        columnLower.includes('debit') || columnLower.includes('credit')) {
      defaultValue = 0;
    } else if (columnLower.includes('date')) {
      defaultValue = new Date().toISOString().split('T')[0];
    }
    
    const newData = tableData.map(row => ({
      ...row,
      [cleanColumnName]: defaultValue
    }));
    
    setTableData(newData);
    
    if (onDataChange) {
      onDataChange(newData);
    }
    
    toast.success(`Column "${cleanColumnName}" added successfully`);
  };

  const removeColumn = (columnKey: string) => {
    if (readOnly) return;
    
    // Confirm deletion for safety
    if (!confirm(`Are you sure you want to delete the column "${columnKey}"?`)) {
      return;
    }
    
    const newData = tableData.map(row => {
      const newRow = { ...row };
      delete newRow[columnKey];
      return newRow;
    });
    
    setTableData(newData);
    setColumnsToRemove(prev => new Set([...prev, columnKey]));
    
    if (onDataChange) {
      onDataChange(newData);
    }
    
    toast.success(`Column "${columnKey}" deleted successfully`);
  };

  const toggleEditMode = () => {
    if (editingCell) {
      handleCellCancel();
    }
    setIsEditMode(!isEditMode);
    toast.success(isEditMode ? 'Edit mode disabled' : 'Edit mode enabled');
  };

  const renderPagination = () => {
    if (totalPages <= 1) return null;

    return (
      <div className="flex items-center justify-between px-4 py-3 bg-gray-50 border-t border-gray-200">
        <div className="flex-1 flex justify-between sm:hidden">
          <button
            onClick={() => setCurrentPage(prev => Math.max(prev - 1, 1))}
            disabled={currentPage === 1}
            className="relative inline-flex items-center px-4 py-2 border border-gray-300 text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Previous
          </button>
          <button
            onClick={() => setCurrentPage(prev => Math.min(prev + 1, totalPages))}
            disabled={currentPage === totalPages}
            className="ml-3 relative inline-flex items-center px-4 py-2 border border-gray-300 text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Next
          </button>
        </div>
        <div className="hidden sm:flex-1 sm:flex sm:items-center sm:justify-between">
          <div>
            <p className="text-sm text-gray-700">
              Showing <span className="font-medium">{indexOfFirstItem + 1}</span> to{' '}
              <span className="font-medium">{Math.min(indexOfLastItem, tableData.length)}</span> of{' '}
              <span className="font-medium">{tableData.length}</span> results
            </p>
          </div>
          <div>
            <nav className="relative z-0 inline-flex rounded-md shadow-sm -space-x-px" aria-label="Pagination">
              <button
                onClick={() => setCurrentPage(prev => Math.max(prev - 1, 1))}
                disabled={currentPage === 1}
                className="relative inline-flex items-center px-2 py-2 rounded-l-md border border-gray-300 bg-white text-sm font-medium text-gray-500 hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <ChevronLeftIcon className="h-5 w-5" />
              </button>
              
              {/* Page numbers */}
              {[...Array(totalPages)].map((_, index) => {
                const pageNumber = index + 1;
                if (pageNumber === 1 || pageNumber === totalPages || 
                    (pageNumber >= currentPage - 1 && pageNumber <= currentPage + 1)) {
                  return (
                    <button
                      key={pageNumber}
                      onClick={() => setCurrentPage(pageNumber)}
                      className={`relative inline-flex items-center px-4 py-2 border text-sm font-medium ${
                        currentPage === pageNumber
                          ? 'z-10 bg-blue-50 border-blue-500 text-blue-600'
                          : 'bg-white border-gray-300 text-gray-500 hover:bg-gray-50'
                      }`}
                    >
                      {pageNumber}
                    </button>
                  );
                } else if (pageNumber === currentPage - 2 || pageNumber === currentPage + 2) {
                  return <span key={pageNumber} className="relative inline-flex items-center px-4 py-2 border border-gray-300 bg-white text-sm font-medium text-gray-700">...</span>;
                }
                return null;
              })}
              
              <button
                onClick={() => setCurrentPage(prev => Math.min(prev + 1, totalPages))}
                disabled={currentPage === totalPages}
                className="relative inline-flex items-center px-2 py-2 rounded-r-md border border-gray-300 bg-white text-sm font-medium text-gray-500 hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <ChevronRightIcon className="h-5 w-5" />
              </button>
            </nav>
          </div>
        </div>
      </div>
    );
  };

  if (!tableData || tableData.length === 0) {
    return (
      <div className="text-center py-8">
        <div className="text-gray-500 mb-4">No data available</div>
        {!readOnly && (
          <button
            onClick={addNewRow}
            className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
          >
            Add First Row
          </button>
        )}
      </div>
    );
  }

  const headers = Object.keys(tableData[0]);
  const amountColumns = detectAmountColumns(headers);

  return (
    <div className="space-y-4">
      {/* Enhanced Action Bar */}
      <div className="flex items-center justify-between p-4 bg-gray-50 border-b border-gray-200">
        <div className="flex items-center space-x-4">
          <h3 className="text-lg font-semibold text-gray-900">
            Data Validation Table
          </h3>
          {!readOnly && (
            <div className="flex items-center space-x-2">
              <button
                onClick={toggleEditMode}
                className={`flex items-center px-3 py-2 rounded-md text-sm font-medium transition-colors ${
                  isEditMode 
                    ? 'bg-green-100 text-green-700 hover:bg-green-200' 
                    : 'bg-blue-100 text-blue-700 hover:bg-blue-200'
                }`}
              >
                <PencilIcon className="h-4 w-4 mr-2" />
                {isEditMode ? 'Exit Edit Mode' : 'Enable Edit Mode'}
              </button>
              
              {isEditMode && (
                <>
                  <button
                    onClick={addNewRow}
                    className="flex items-center px-3 py-2 bg-green-600 text-white rounded-md hover:bg-green-700 text-sm"
                  >
                    <PlusIcon className="h-4 w-4 mr-2" />
                    Add Row
                  </button>
                  
                  <button
                    onClick={addNewColumn}
                    className="flex items-center px-3 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 text-sm"
                  >
                    <PlusIcon className="h-4 w-4 mr-2" />
                    Add Column
                  </button>
                </>
              )}
            </div>
          )}
        </div>
        
        <div className="flex items-center space-x-4 text-sm text-gray-600">
          <span>
            {tableData.length} rows × {headers.length} columns
          </span>
          {amountColumns.length > 0 && (
            <span className="px-2 py-1 bg-green-100 text-green-700 rounded">
              {amountColumns.length} amount column{amountColumns.length !== 1 ? 's' : ''}
            </span>
          )}
        </div>
      </div>

      {/* Table */}
      <div className="overflow-x-auto border border-gray-200 rounded-lg">
        <table className="min-w-full divide-y divide-gray-200">
        <thead className="bg-gray-50">
          <tr>
              {isEditMode && !readOnly && (
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Actions
                </th>
              )}
            {headers.map((header) => (
              <th
                key={header}
                scope="col"
                  className={`px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider ${
                    amountColumns.includes(header) ? 'bg-green-50' : ''
                  }`}
              >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                {header}
                      {amountColumns.includes(header) && (
                        <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-green-100 text-green-800">
                          ₹
                        </span>
                      )}
                    </div>
                    {isEditMode && !readOnly && (
                      <button
                        onClick={() => removeColumn(header)}
                        className="ml-2 p-1 text-red-500 hover:text-red-700 hover:bg-red-50 rounded"
                        title={`Remove column "${header}"`}
                      >
                        <XMarkIcon className="h-3 w-3" />
                      </button>
                    )}
                  </div>
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="bg-white divide-y divide-gray-200">
            {currentItems.map((row, rowIndex) => {
              const actualRowIndex = indexOfFirstItem + rowIndex; // Calculate actual row index
              return (
                <tr key={actualRowIndex} className="hover:bg-gray-50">
                  {isEditMode && !readOnly && (
                    <td className="px-4 py-2 whitespace-nowrap">
                      <button
                        onClick={() => removeRow(actualRowIndex)}
                        className="p-1 text-red-500 hover:text-red-700 hover:bg-red-50 rounded"
                        title="Remove row"
                      >
                        <TrashIcon className="h-4 w-4" />
                      </button>
                    </td>
                  )}
                  {headers.map((header) => {
                    const isEditing = editingCell?.rowIndex === actualRowIndex && editingCell?.columnKey === header;
                    const cellValue = row[header];
                    
                    return (
                      <td
                        key={header}
                        className={getCellClassName(cellValue, header, actualRowIndex, header)}
                        onClick={() => handleCellClick(actualRowIndex, header)}
                >
                        {isEditing ? (
                          <div className="flex items-center gap-2">
                            <input
                              type="text"
                              value={editValue}
                              onChange={(e) => setEditValue(e.target.value)}
                              className="flex-1 px-2 py-1 text-sm border border-blue-300 rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
                              onKeyDown={(e) => {
                                if (e.key === 'Enter') {
                                  handleCellSave();
                                } else if (e.key === 'Escape') {
                                  handleCellCancel();
                                }
                              }}
                              autoFocus
                            />
                            <button
                              onClick={handleCellSave}
                              className="p-1 text-green-600 hover:text-green-800"
                            >
                              <CheckIcon className="h-4 w-4" />
                            </button>
                            <button
                              onClick={handleCellCancel}
                              className="p-1 text-red-600 hover:text-red-800"
                            >
                              <XMarkIcon className="h-4 w-4" />
                            </button>
                          </div>
                        ) : (
                          <div className="min-h-[1.5rem] flex items-center">
                            {formatValue(cellValue, header)}
                            {isEditMode && !readOnly && (
                              <PencilIcon className="h-3 w-3 ml-2 opacity-0 group-hover:opacity-100 text-gray-400" />
                            )}
                          </div>
                        )}
                </td>
                    );
                  })}
            </tr>
              );
            })}
        </tbody>
      </table>
      </div>
      {renderPagination()}

      {/* Data Summary */}
      <div className="flex items-center justify-between text-sm text-gray-600 bg-gray-50 px-4 py-2 rounded">
        <div>
          Total: {tableData.length} rows, {headers.length} columns
          {totalPages > 1 && (
            <span className="ml-4">
              Page {currentPage} of {totalPages}
            </span>
          )}
        </div>
        <div>
          {amountColumns.length > 0 && (
            <span className="text-green-600">
              {amountColumns.length} amount column{amountColumns.length !== 1 ? 's' : ''} detected
            </span>
          )}
        </div>
      </div>
    </div>
  );
};

export default ValidationTable;
