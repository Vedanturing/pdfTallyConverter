import React, { useState, useRef, useEffect } from 'react';
import {
  ArrowUturnLeftIcon,
  ArrowUturnRightIcon,
  ExclamationCircleIcon,
} from '@heroicons/react/24/outline';
import { toast } from 'react-hot-toast';
import useAuditStore from '../store/auditStore';
import useValidationStore from '../store/validationStore';
import AuditTrail from './AuditTrail';
import { API_URL } from '../config';
import { FinancialEntry } from '../types/financial';
import { validateTable } from '../utils/validation';
import { motion } from 'framer-motion';
import { ArrowUpDown, ChevronDown } from 'lucide-react';
import { Button } from './ui/button';

interface CorrectionNote {
  rowId: string;
  column: string;
  note: string;
  timestamp: string;
  userId: string;
}

interface FinancialTableProps {
  data: FinancialEntry[];
  readOnly?: boolean;
  onDataChange?: (data: FinancialEntry[]) => void;
  onSort?: (column: keyof FinancialEntry) => void;
  sortColumn?: keyof FinancialEntry | null;
  sortDirection?: 'asc' | 'desc';
}

const FinancialTable: React.FC<FinancialTableProps> = ({
  data,
  readOnly = false,
  onDataChange,
  onSort,
  sortColumn,
  sortDirection = 'asc',
}) => {
  const [tableData, setTableData] = useState<FinancialEntry[]>(data);
  const [editingCell, setEditingCell] = useState<{ rowId: string; field: string } | null>(null);
  const [history, setHistory] = useState<FinancialEntry[][]>([data]);
  const [currentHistoryIndex, setCurrentHistoryIndex] = useState(0);
  const [showAuditTrail, setShowAuditTrail] = useState(false);
  const [showOnlyErrors, setShowOnlyErrors] = useState(false);
  const [correctionNotes, setCorrectionNotes] = useState<CorrectionNote[]>([]);
  const [selectedCell, setSelectedCell] = useState<{ rowId: string; field: string } | null>(null);
  const isInitialMount = useRef(true);
  const previousData = useRef<FinancialEntry[]>(data);
  const auditStore = useAuditStore();
  const validationStore = useValidationStore();

  useEffect(() => {
    if (isInitialMount.current) {
      isInitialMount.current = false;
      return;
    }

    // Check if the data has actually changed to prevent unnecessary validation
    const hasDataChanged = JSON.stringify(tableData) !== JSON.stringify(previousData.current);
    if (!hasDataChanged) {
      return;
    }

    previousData.current = [...tableData];
    const newValidationMap = validateTable(tableData);
    validationStore.setValidationMap(newValidationMap);

    // Add validation entry to audit trail
    const totalErrors = Array.from(newValidationMap.values()).reduce(
      (sum, result) => sum + result.errors.filter(e => e.type === 'error').length,
      0
    );
    const totalWarnings = Array.from(newValidationMap.values()).reduce(
      (sum, result) => sum + result.errors.filter(e => e.type === 'warning').length,
      0
    );

    auditStore.addEntry({
      userId: 'current-user',
      action: 'validate',
      entityType: 'table',
      entityId: 'financial-table',
      field: 'validation',
      oldValue: JSON.stringify({ errors: 0, warnings: 0 }),
      newValue: JSON.stringify({ errors: totalErrors, warnings: totalWarnings }),
    });
  }, [tableData]);

  const handleCellEdit = (rowId: string, field: string, value: string) => {
    const oldRow = tableData.find((row) => row.id === rowId);
    const oldValue = oldRow?.[field];
    let newValue: string | number = value;

    if (field === 'amount') {
      const numValue = parseFloat(value.replace(/[^\d.-]/g, ''));
      newValue = isNaN(numValue) ? value : numValue;
    }

    const newData = tableData.map((row) => {
      if (row.id === rowId) {
        return { ...row, [field]: newValue };
      }
      return row;
    });
    
    setTableData(newData);
    onDataChange?.(newData);

    // Add to history
    const newHistory = history.slice(0, currentHistoryIndex + 1);
    newHistory.push(newData);
    setHistory(newHistory);
    setCurrentHistoryIndex(newHistory.length - 1);

    // Add audit entry
    auditStore.addEntry({
      userId: 'current-user',
      action: 'update',
      entityType: 'cell',
      entityId: rowId,
      field,
      oldValue: oldValue?.toString(),
      newValue: newValue.toString(),
    });
  };

  const addCorrectionNote = (note: string) => {
    if (!selectedCell) return;

    const newNote: CorrectionNote = {
      rowId: selectedCell.rowId,
      column: selectedCell.field,
      note,
      timestamp: new Date().toISOString(),
      userId: 'current-user',
    };

    setCorrectionNotes([...correctionNotes, newNote]);
    auditStore.addEntry({
      userId: 'current-user',
      action: 'add_note',
      entityType: 'cell',
      entityId: selectedCell.rowId,
      field: selectedCell.field,
      note,
    });
  };

  const handleCellRightClick = (e: React.MouseEvent, rowId: string, field: string) => {
    e.preventDefault();
    setSelectedCell({ rowId, field });
    // Show context menu or note input
  };

  const getFilteredData = () => {
    if (!showOnlyErrors) return tableData;
    return tableData.filter(row => {
      const result = validationStore.validationMap.get(row.id);
      return result ? !result.isValid : false;
    });
  };

  const handleUndo = () => {
    if (currentHistoryIndex > 0) {
      setCurrentHistoryIndex(currentHistoryIndex - 1);
      const previousState = history[currentHistoryIndex - 1];
      setTableData(previousState);
      onDataChange?.(previousState);
    }
  };

  const handleRedo = () => {
    if (currentHistoryIndex < history.length - 1) {
      setCurrentHistoryIndex(currentHistoryIndex + 1);
      const nextState = history[currentHistoryIndex + 1];
      setTableData(nextState);
      onDataChange?.(nextState);
    }
  };

  const getCellClassName = (rowId: string, field: string) => {
    const baseClasses = 'px-4 py-2 text-sm border-r border-gray-300 relative';
    const result = validationStore.validationMap.get(rowId);
    const hasError = result ? result.errors.some(e => e.field === field && e.type === 'error') : false;
    return `${baseClasses} ${hasError ? 'bg-red-50' : ''} ${
      editingCell?.rowId === rowId && editingCell?.field === field
        ? 'bg-blue-50'
        : ''
    }`;
  };

  const getErrorIcon = (rowId: string, field: string) => {
    const result = validationStore.validationMap.get(rowId);
    const hasError = result ? result.errors.some(e => e.field === field && e.type === 'error') : false;
    if (hasError) {
      return (
        <div className="absolute top-0 right-0 -mt-2 -mr-2">
          <ExclamationCircleIcon className="h-4 w-4 text-red-500" />
        </div>
      );
    }
    return null;
  };

  const formatCellValue = (value: string | number, field: string) => {
    if (field === 'amount') {
      const numValue = typeof value === 'string' ? parseFloat(value) : value;
      return new Intl.NumberFormat('en-IN', {
        style: 'currency',
        currency: 'INR',
      }).format(numValue);
    }
    return value;
  };

  const getValidationStatus = (entry: FinancialEntry) => {
    const validation = validationStore.validationMap.get(entry.id);
    if (!validation) return null;

    return {
      isValid: validation.isValid,
      errorCount: validation.errors.filter(e => e.type === 'error').length,
      warningCount: validation.errors.filter(e => e.type === 'warning').length,
    };
  };

  const handleCellClick = (entry: FinancialEntry, field: string) => {
    const validation = validationStore.validationMap.get(entry.id);
    if (!validation) return;

    const errors = validation.errors.filter(e => e.field === field);
    if (errors.length > 0) {
      // Add to audit trail
      auditStore.addEntry({
        userId: 'current-user', // Replace with actual user ID
        action: 'validate',
        entityType: 'cell',
        entityId: entry.id,
        field,
        validationResult: {
          isValid: errors.every(e => e.type === 'warning'),
          errorCount: errors.filter(e => e.type === 'error').length,
          warningCount: errors.filter(e => e.type === 'warning').length,
        },
      });
    }
  };

  return (
    <div className="space-y-4">
      {!readOnly && (
        <div className="flex justify-between items-center mb-4">
          <div className="flex items-center space-x-4">
            <label className="flex items-center space-x-2">
              <input
                type="checkbox"
                checked={showOnlyErrors}
                onChange={(e) => setShowOnlyErrors(e.target.checked)}
                className="form-checkbox h-4 w-4 text-blue-600"
              />
              <span>Show Only Rows with Errors</span>
            </label>
            <button
              onClick={() => setShowAuditTrail(true)}
              className="px-3 py-1 text-sm bg-gray-100 hover:bg-gray-200 rounded"
            >
              View Audit Trail
            </button>
          </div>
          <div className="flex space-x-2">
            <button
              onClick={handleUndo}
              disabled={currentHistoryIndex === 0}
              className="p-2 text-gray-600 hover:text-gray-900 disabled:text-gray-300"
              title="Undo"
            >
              <ArrowUturnLeftIcon className="h-5 w-5" />
            </button>
            <button
              onClick={handleRedo}
              disabled={currentHistoryIndex === history.length - 1}
              className="p-2 text-gray-600 hover:text-gray-900 disabled:text-gray-300"
              title="Redo"
            >
              <ArrowUturnRightIcon className="h-5 w-5" />
            </button>
          </div>
        </div>
      )}

      <div className="overflow-x-auto">
        <table className="min-w-full divide-y divide-gray-200 border border-gray-300 rounded-lg">
          <thead className="bg-gray-50">
            <tr>
              {Object.keys(tableData[0] || {}).map((header) => (
                <th
                  key={header}
                  className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider border-b border-gray-300"
                >
                  <Button
                    variant="ghost"
                    onClick={() => onSort?.(header as keyof FinancialEntry)}
                    className="h-8 flex items-center gap-2"
                  >
                    {header}
                    {!readOnly && <ArrowUpDown className="h-4 w-4" />}
                  </Button>
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-gray-200">
            {getFilteredData().map((row) => (
              <motion.tr
                key={row.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.05 * Array.from(getFilteredData()).indexOf(row) }}
                className={`border-b transition-colors hover:bg-muted/50 ${
                  !readOnly && getValidationStatus(row)?.isValid === false ? 'bg-red-50' :
                  !readOnly && getValidationStatus(row)?.isValid === true ? 'bg-green-50' :
                  ''
                }`}
              >
                {Object.entries(row).map(([field, value]) => (
                  <td
                    key={`${row.id}-${field}`}
                    className={readOnly ? 'px-4 py-2 text-sm border-r border-gray-300 relative' : getCellClassName(row.id, field)}
                    onClick={(e) => {
                      if (!readOnly) {
                        e.preventDefault();
                        handleCellClick(row, field);
                      }
                    }}
                  >
                    {formatCellValue(value, field)}
                    {!readOnly && getErrorIcon(row.id, field)}
                  </td>
                ))}
              </motion.tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default FinancialTable;