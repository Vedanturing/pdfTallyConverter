import React, { useState, useEffect } from 'react';
import { TableData, TableCell, ValidationError } from '../../types/DataValidation';
import { ExclamationCircleIcon, CheckCircleIcon, XCircleIcon, MagnifyingGlassIcon } from '@heroicons/react/24/outline';

interface Props {
  data: TableData;
  onCellEdit: (rowId: string, columnKey: string, newValue: string) => void;
  onStatusChange: (rowId: string, columnKey: string, status: 'corrected' | 'ignored' | 'needs-review') => void;
  validationErrors: ValidationError[];
  selectedError: ValidationError | null;
}

const EditableTable: React.FC<Props> = ({
  data,
  onCellEdit,
  onStatusChange,
  validationErrors,
  selectedError
}) => {
  const [editingCell, setEditingCell] = useState<{ rowId: string; columnKey: string } | null>(null);
  const [editValue, setEditValue] = useState('');

  useEffect(() => {
    if (selectedError) {
      const element = document.querySelector(`[data-row-id="${selectedError.rowId}"][data-column-key="${selectedError.columnKey}"]`);
      if (element) {
        element.scrollIntoView({ behavior: 'smooth', block: 'center' });
        setEditingCell({ rowId: selectedError.rowId, columnKey: selectedError.columnKey });
      }
    }
  }, [selectedError]);

  const getCellError = (rowId: string, columnKey: string) => {
    return validationErrors.find(
      error => error.rowId === rowId && error.columnKey === columnKey
    );
  };

  const handleCellClick = (rowId: string, columnKey: string, currentValue: string) => {
    if (columnKey === 'id') return;
    setEditingCell({ rowId, columnKey });
    setEditValue(currentValue);
  };

  const handleCellBlur = () => {
    if (editingCell) {
      onCellEdit(editingCell.rowId, editingCell.columnKey, editValue);
      setEditingCell(null);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleCellBlur();
    } else if (e.key === 'Escape') {
      setEditingCell(null);
    }
  };

  const getCellClassName = (cell: TableCell, error: ValidationError | undefined) => {
    const baseClasses = 'px-4 py-2 whitespace-nowrap text-sm';
    const statusClasses = cell.metadata?.status === 'corrected' ? 'bg-green-50' : error ? 'bg-red-50' : '';
    const selectedClasses = selectedError?.rowId === error?.rowId && selectedError?.columnKey === error?.columnKey
      ? 'ring-2 ring-blue-500'
      : '';
    return `${baseClasses} ${statusClasses} ${selectedClasses}`;
  };

  const renderStatusButtons = (rowId: string, columnKey: string, cell: TableCell) => (
    <div className="absolute top-0 right-0 flex space-x-1 opacity-0 group-hover:opacity-100 transition-opacity">
      <button
        onClick={() => onStatusChange(rowId, columnKey, 'corrected')}
        className={`p-1 rounded ${cell.metadata?.status === 'corrected' ? 'text-green-600' : 'text-gray-400 hover:text-green-600'}`}
        title="Mark as corrected"
      >
        <CheckCircleIcon className="w-4 h-4" />
      </button>
      <button
        onClick={() => onStatusChange(rowId, columnKey, 'ignored')}
        className={`p-1 rounded ${cell.metadata?.status === 'ignored' ? 'text-red-600' : 'text-gray-400 hover:text-red-600'}`}
        title="Mark as ignored"
      >
        <XCircleIcon className="w-4 h-4" />
      </button>
      <button
        onClick={() => onStatusChange(rowId, columnKey, 'needs-review')}
        className={`p-1 rounded ${cell.metadata?.status === 'needs-review' ? 'text-yellow-600' : 'text-gray-400 hover:text-yellow-600'}`}
        title="Mark for review"
      >
        <MagnifyingGlassIcon className="w-4 h-4" />
      </button>
    </div>
  );

  return (
    <div className="overflow-x-auto">
      <table className="min-w-full divide-y divide-gray-200">
        <thead className="bg-gray-50">
          <tr>
            {data.headers.map((header, index) => (
              <th
                key={index}
                className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider"
              >
                {header}
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="bg-white divide-y divide-gray-200">
          {data.rows.map((row) => (
            <tr key={row.id.value}>
              {data.headers.map((header) => {
                const cell = row[header];
                const error = getCellError(row.id.value, header);
                const isEditing = editingCell?.rowId === row.id.value && editingCell?.columnKey === header;

                return (
                  <td
                    key={`${row.id.value}-${header}`}
                    className={getCellClassName(cell, error)}
                    onClick={() => handleCellClick(row.id.value, header, cell.value.toString())}
                    data-row-id={row.id.value}
                    data-column-key={header}
                  >
                    <div className="flex items-center justify-between">
                      {isEditing ? (
                        <input
                          type="text"
                          value={editValue}
                          onChange={(e) => setEditValue(e.target.value)}
                          onBlur={handleCellBlur}
                          onKeyDown={handleKeyDown}
                          className="bg-transparent border-none focus:ring-0 w-full"
                        />
                      ) : (
                        <div className="flex items-center">
                          <span className="flex-grow">{cell.value.toString()}</span>
                          {error && (
                            <div className="ml-2 text-red-500 tooltip-trigger">
                              <ExclamationCircleIcon className="h-5 w-5" />
                              <div className="tooltip absolute hidden group-hover:block bg-white p-2 rounded shadow-lg border border-gray-200 z-10">
                                {error.message}
                              </div>
                            </div>
                          )}
                          {cell.metadata?.status === 'corrected' && !error && (
                            <div className="ml-2 text-green-500">
                              <CheckCircleIcon className="h-5 w-5" />
                            </div>
                          )}
                        </div>
                      )}
                      {renderStatusButtons(row.id.value, header, cell)}
                    </div>
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export default EditableTable; 