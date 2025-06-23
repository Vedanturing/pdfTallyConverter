import React, { useState } from 'react';
import { ExclamationCircleIcon, CheckCircleIcon, PencilIcon } from '@heroicons/react/24/outline';
import { Tooltip } from '../ui/Tooltip';

export interface ValidationIssue {
  message: string;
  severity: 'error' | 'warning';
  suggestedFix?: string;
  type: 'amount' | 'date' | 'gstin' | 'tax' | 'voucher' | 'other';
}

interface ValidationCellProps {
  value: string;
  field: string;
  rowIndex: number;
  issue?: ValidationIssue;
  isIgnored?: boolean;
  onEdit: (value: string) => void;
  onIgnore: () => void;
  onApplyFix: (fix: string) => void;
}

export const ValidationCell: React.FC<ValidationCellProps> = ({
  value,
  field,
  rowIndex,
  issue,
  isIgnored,
  onEdit,
  onIgnore,
  onApplyFix,
}) => {
  const [isEditing, setIsEditing] = useState(false);
  const [editValue, setEditValue] = useState(value);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      onEdit(editValue);
      setIsEditing(false);
    } else if (e.key === 'Escape') {
      setEditValue(value);
      setIsEditing(false);
    }
  };

  const getCellStyle = () => {
    if (isIgnored) return 'bg-gray-50';
    if (!issue) return '';
    return issue.severity === 'error' ? 'bg-red-50' : 'bg-yellow-50';
  };

  const getBorderStyle = () => {
    if (isIgnored) return 'border-gray-300';
    if (!issue) return 'border-transparent';
    return issue.severity === 'error' ? 'border-red-300' : 'border-yellow-300';
  };

  return (
    <td className={`px-4 py-2 relative ${getCellStyle()}`}>
      <div className="flex items-center group">
        {isEditing ? (
          <input
            type="text"
            value={editValue}
            onChange={(e) => setEditValue(e.target.value)}
            onKeyDown={handleKeyDown}
            onBlur={() => {
              onEdit(editValue);
              setIsEditing(false);
            }}
            className={`w-full p-1 border rounded ${getBorderStyle()} focus:outline-none focus:ring-2 focus:ring-blue-500`}
            autoFocus
          />
        ) : (
          <>
            <span className="flex-grow">{value}</span>
            {!isIgnored && (
              <button
                onClick={() => setIsEditing(true)}
                className="opacity-0 group-hover:opacity-100 ml-2"
              >
                <PencilIcon className="h-4 w-4 text-gray-500 hover:text-gray-700" />
              </button>
            )}
          </>
        )}

        {issue && !isIgnored && (
          <Tooltip
            content={
              <div className="p-2 max-w-xs">
                <div className="font-medium text-gray-900">{issue.message}</div>
                {issue.suggestedFix && (
                  <div className="mt-1 text-sm text-gray-600">
                    Suggested fix: {issue.suggestedFix}
                  </div>
                )}
                <div className="mt-2 flex gap-2">
                  {issue.suggestedFix && (
                    <button
                      onClick={() => onApplyFix(issue.suggestedFix!)}
                      className="px-2 py-1 text-xs bg-green-100 text-green-800 rounded hover:bg-green-200"
                    >
                      Apply Fix
                    </button>
                  )}
                  <button
                    onClick={onIgnore}
                    className="px-2 py-1 text-xs bg-gray-100 text-gray-800 rounded hover:bg-gray-200"
                  >
                    Ignore
                  </button>
                </div>
              </div>
            }
          >
            <div className="absolute right-2 top-1/2 transform -translate-y-1/2">
              <ExclamationCircleIcon
                className={`h-5 w-5 ${
                  issue.severity === 'error' ? 'text-red-500' : 'text-yellow-500'
                }`}
              />
            </div>
          </Tooltip>
        )}

        {isIgnored && (
          <Tooltip content="This issue has been ignored">
            <CheckCircleIcon className="h-5 w-5 text-gray-400 ml-2" />
          </Tooltip>
        )}
      </div>
    </td>
  );
}; 