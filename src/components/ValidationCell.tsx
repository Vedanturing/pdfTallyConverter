import React from 'react';
import { motion } from 'framer-motion';
import {
  ExclamationTriangleIcon,
  ExclamationCircleIcon,
  CheckIcon,
  XMarkIcon,
  PencilIcon,
} from '@heroicons/react/24/outline';
import ValidationTooltip from './ValidationTooltip';
import IgnoreButton from './IgnoreButton';

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

interface CellData {
  value: any;
  originalValue: any;
  isEdited: boolean;
  isIgnored: boolean;
  issues: ValidationIssue[];
}

interface ValidationCellProps {
  cellData: CellData;
  isEditing: boolean;
  editValue: string;
  onEditValueChange: (value: string) => void;
  onStartEdit: () => void;
  onSaveEdit: () => void;
  onCancelEdit: () => void;
  onApplyFix: (issueId: string, suggestedValue: any) => void;
  onIgnoreIssue: (issueId: string) => void;
  columnType?: string;
}

const ValidationCell: React.FC<ValidationCellProps> = ({
  cellData,
  isEditing,
  editValue,
  onEditValueChange,
  onStartEdit,
  onSaveEdit,
  onCancelEdit,
  onApplyFix,
  onIgnoreIssue,
  columnType = 'text'
}) => {
  const { value, originalValue, isEdited, isIgnored, issues } = cellData;
  
  // Filter out ignored and fixed issues
  const activeIssues = issues.filter(issue => !issue.ignored && !issue.fixed);
  const hasErrors = activeIssues.some(issue => issue.type === 'error');
  const hasWarnings = activeIssues.some(issue => issue.type === 'warning');
  
  // Determine cell styling based on issues
  const getCellStyles = () => {
    let baseClasses = "px-6 py-4 whitespace-nowrap text-sm relative group transition-colors";
    
    if (hasErrors) {
      baseClasses += " bg-red-50 border-l-4 border-red-400";
    } else if (hasWarnings) {
      baseClasses += " bg-yellow-50 border-l-4 border-yellow-400";
    } else if (isEdited) {
      baseClasses += " bg-blue-50 border-l-4 border-blue-400";
    } else {
      baseClasses += " text-gray-900";
    }
    
    return baseClasses;
  };

  // Format value for display
  const formatValue = (val: any) => {
    if (val === null || val === undefined || val === '') {
      return <span className="text-gray-400 italic">Empty</span>;
    }
    
    if (columnType === 'number' && typeof val === 'number') {
      return new Intl.NumberFormat('en-IN', {
        style: 'currency',
        currency: 'INR',
        minimumFractionDigits: 2
      }).format(val);
    }
    
    if (columnType === 'date' && val) {
      try {
        const date = new Date(val);
        return date.toLocaleDateString('en-IN');
      } catch {
        return String(val);
      }
    }
    
    return String(val);
  };

  if (isEditing) {
    return (
      <td className={getCellStyles()}>
        <div className="flex items-center space-x-2">
          <input
            type={columnType === 'number' ? 'number' : columnType === 'date' ? 'date' : 'text'}
            value={editValue}
            onChange={(e) => onEditValueChange(e.target.value)}
            className="flex-1 px-2 py-1 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500 text-sm"
            autoFocus
          />
          <button
            onClick={onSaveEdit}
            className="p-1 text-green-600 hover:text-green-800 hover:bg-green-100 rounded"
            title="Save changes"
          >
            <CheckIcon className="h-4 w-4" />
          </button>
          <button
            onClick={onCancelEdit}
            className="p-1 text-red-600 hover:text-red-800 hover:bg-red-100 rounded"
            title="Cancel editing"
          >
            <XMarkIcon className="h-4 w-4" />
          </button>
        </div>
      </td>
    );
  }

  return (
    <td className={getCellStyles()}>
      <div className="flex items-center justify-between min-h-[2rem]">
        <div className="flex-1">
          {formatValue(value)}
          {isEdited && (
            <span className="ml-2 text-xs text-blue-600 font-medium">
              (Modified)
            </span>
          )}
        </div>
        
        {/* Edit button (appears on hover) */}
        <button
          onClick={onStartEdit}
          className="opacity-0 group-hover:opacity-100 ml-2 p-1 text-gray-500 hover:text-gray-700 hover:bg-gray-100 rounded transition-opacity"
          title="Edit cell"
        >
          <PencilIcon className="h-4 w-4" />
        </button>
      </div>

      {/* Issue indicators */}
      {activeIssues.length > 0 && (
        <div className="absolute -top-1 -right-1 flex space-x-1">
          {hasErrors && (
            <motion.div
              initial={{ scale: 0 }}
              animate={{ scale: 1 }}
              className="h-3 w-3 bg-red-500 rounded-full flex items-center justify-center"
            >
              <ExclamationCircleIcon className="h-2 w-2 text-white" />
            </motion.div>
          )}
          {hasWarnings && !hasErrors && (
            <motion.div
              initial={{ scale: 0 }}
              animate={{ scale: 1 }}
              className="h-3 w-3 bg-yellow-500 rounded-full flex items-center justify-center"
            >
              <ExclamationTriangleIcon className="h-2 w-2 text-white" />
            </motion.div>
          )}
        </div>
      )}

      {/* Validation tooltip */}
      {activeIssues.length > 0 && (
        <ValidationTooltip
          issues={activeIssues}
          onApplyFix={onApplyFix}
          onIgnore={onIgnoreIssue}
        />
      )}
    </td>
  );
};

export default ValidationCell; 