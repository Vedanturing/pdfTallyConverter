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
}

const FinancialTable: React.FC<FinancialTableProps> = ({
  data,
  readOnly = false,
  onDataChange,
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
  const { validationMap, setValidationMap } = useValidationStore();

  useEffect(() => {
    if (isInitialMount.current) {
      isInitialMount.current = false;
      return;
    }

    const newValidationMap = validateTable(tableData);
    setValidationMap(newValidationMap);

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
  }, [tableData, setValidationMap]);

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
      const result = validationMap.get(row.id);
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
    const result = validationMap.get(rowId);
    const hasError = result ? result.errors.some(e => e.field === field && e.type === 'error') : false;
    return `${baseClasses} ${hasError ? 'bg-red-50' : ''} ${
      editingCell?.rowId === rowId && editingCell?.field === field
        ? 'bg-blue-50'
        : ''
    }`;
  };

  const getErrorIcon = (rowId: string, field: string) => {
    const result = validationMap.get(rowId);
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

  return (
    <div className="space-y-4">
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

      <div className="overflow-x-auto">
        <table className="min-w-full divide-y divide-gray-200 border border-gray-300 rounded-lg">
          <thead className="bg-gray-50">
            <tr>
              {Object.keys(tableData[0] || {}).map((header) => (
                <th
                  key={header}
                  className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider border-b border-gray-300"
                >
                  {header}
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-gray-200">
            {getFilteredData().map((row) => (
              <tr key={row.id} className="hover:bg-gray-50">
                {Object.entries(row).map(([field, value]) => (
                  <td
                    key={`${row.id}-${field}`}
                    className={getCellClassName(row.id, field)}
                    onClick={() => !readOnly && setEditingCell({ rowId: row.id, field })}
                    onContextMenu={(e) => handleCellRightClick(e, row.id, field)}
                  >
                    {editingCell?.rowId === row.id && editingCell?.field === field ? (
                      <input
                        type={field === 'amount' ? 'number' : 'text'}
                        defaultValue={value}
                        className="w-full p-1 border border-blue-500 rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
                        onBlur={(e) => {
                          handleCellEdit(row.id, field, e.target.value);
                          setEditingCell(null);
                        }}
                        onKeyDown={(e) => {
                          if (e.key === 'Enter') {
                            handleCellEdit(row.id, field, e.currentTarget.value);
                            setEditingCell(null);
                          }
                        }}
                        autoFocus
                      />
                    ) : (
                      <div className="relative group">
                        {formatCellValue(value, field)}
                        {getErrorIcon(row.id, field)}
                        {correctionNotes.find(
                          note => note.rowId === row.id && note.column === field
                        ) && (
                          <div className="absolute top-0 right-0">
                            <span className="text-yellow-500">🟡</span>
                          </div>
                        )}
                      </div>
                    )}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {showAuditTrail && (
        <AuditTrail onClose={() => setShowAuditTrail(false)} />
      )}
    </div>
  );
};

export default FinancialTable; 