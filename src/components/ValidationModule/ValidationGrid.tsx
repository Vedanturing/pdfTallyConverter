import React, { useState } from 'react';
import { FinancialEntry } from '../../types/financial';
import { ValidationGridProps, CellValidationIssue } from './types';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '../ui/table';
import { Button } from '../ui/button';
import { Input } from '../ui/input';
import { Tooltip } from '../ui/Tooltip';
import { PencilIcon, CheckIcon, XMarkIcon } from '@heroicons/react/24/outline';

const COLUMNS: Array<{ key: keyof FinancialEntry; label: string }> = [
  { key: 'date', label: 'Date' },
  { key: 'description', label: 'Description' },
  { key: 'amount', label: 'Amount' },
  { key: 'balance', label: 'Balance' },
  { key: 'gstin', label: 'GSTIN' },
  { key: 'taxRate', label: 'Tax Rate' },
  { key: 'voucherNo', label: 'Voucher No.' }
];

export function ValidationGrid({
  data,
  issues,
  onCellEdit,
  onIgnoreIssue,
  editableRows,
  onToggleEditable
}: ValidationGridProps) {
  const [editingCell, setEditingCell] = useState<{
    rowIndex: number;
    field: keyof FinancialEntry;
    value: any;
  } | null>(null);

  const getCellIssues = (rowIndex: number, field: keyof FinancialEntry): CellValidationIssue[] => {
    const cellId = `${rowIndex}-${field}`;
    return issues[cellId] || [];
  };

  const getCellStyle = (rowIndex: number, field: keyof FinancialEntry) => {
    const cellIssues = getCellIssues(rowIndex, field);
    if (cellIssues.length === 0) return {};

    const hasCritical = cellIssues.some(issue => issue.severity === 'error');
    const hasWarning = cellIssues.some(issue => issue.severity === 'warning');

    return {
      backgroundColor: hasCritical ? 'rgba(239, 68, 68, 0.1)' : hasWarning ? 'rgba(245, 158, 11, 0.1)' : undefined,
      borderColor: hasCritical ? 'rgb(239, 68, 68)' : hasWarning ? 'rgb(245, 158, 11)' : undefined
    };
  };

  const handleStartEdit = (rowIndex: number, field: keyof FinancialEntry, value: any) => {
    setEditingCell({ rowIndex, field, value });
  };

  const handleSaveEdit = () => {
    if (editingCell) {
      onCellEdit(editingCell.rowIndex, editingCell.field, editingCell.value);
      setEditingCell(null);
    }
  };

  const handleCancelEdit = () => {
    setEditingCell(null);
  };

  const renderCellContent = (row: FinancialEntry, rowIndex: number, field: keyof FinancialEntry) => {
    const isEditing = editingCell?.rowIndex === rowIndex && editingCell?.field === field;
    const isEditable = editableRows.has(`row-${rowIndex}`);
    const value = row[field];
    const cellIssues = getCellIssues(rowIndex, field);

    if (isEditing) {
      return (
        <div className="flex items-center space-x-2">
          <Input
            value={editingCell.value}
            onChange={(e) => setEditingCell({ ...editingCell, value: e.target.value })}
            className="w-full"
          />
          <Button size="sm" onClick={handleSaveEdit}>
            <CheckIcon className="h-4 w-4" />
          </Button>
          <Button size="sm" variant="ghost" onClick={handleCancelEdit}>
            <XMarkIcon className="h-4 w-4" />
          </Button>
        </div>
      );
    }

    return (
      <div className="group relative">
        <div className="flex items-center justify-between">
          <span>{value}</span>
          {isEditable && (
            <Button
              size="sm"
              variant="ghost"
              onClick={() => handleStartEdit(rowIndex, field, value)}
              className="opacity-0 group-hover:opacity-100"
            >
              <PencilIcon className="h-4 w-4" />
            </Button>
          )}
        </div>
        {cellIssues.length > 0 && (
          <Tooltip
            content={
              <div className="space-y-1">
                {cellIssues.map((issue, i) => (
                  <div key={i} className="flex items-start space-x-2">
                    <span className={issue.severity === 'error' ? 'text-red-500' : 'text-yellow-500'}>
                      •
                    </span>
                    <div>
                      <p className="font-medium">{issue.message}</p>
                      {issue.suggestedValue && (
                        <p className="text-sm text-gray-500">
                          Suggested: {issue.suggestedValue}
                        </p>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            }
          >
            <div
              className={`absolute -top-1 -right-1 h-2 w-2 rounded-full ${
                cellIssues.some(i => i.severity === 'error') ? 'bg-red-500' : 'bg-yellow-500'
              }`}
            />
          </Tooltip>
        )}
      </div>
    );
  };

  return (
    <div className="rounded-md border">
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead className="w-[50px]">Actions</TableHead>
            {COLUMNS.map(col => (
              <TableHead key={col.key}>{col.label}</TableHead>
            ))}
          </TableRow>
        </TableHeader>
        <TableBody>
          {data.map((row, rowIndex) => (
            <TableRow key={rowIndex}>
              <TableCell>
                <Button
                  size="sm"
                  variant={editableRows.has(`row-${rowIndex}`) ? 'default' : 'outline'}
                  onClick={() => onToggleEditable(rowIndex)}
                >
                  {editableRows.has(`row-${rowIndex}`) ? 'Lock' : 'Edit'}
                </Button>
              </TableCell>
              {COLUMNS.map(col => (
                <TableCell
                  key={col.key}
                  style={getCellStyle(rowIndex, col.key)}
                  className="relative"
                >
                  {renderCellContent(row, rowIndex, col.key)}
                </TableCell>
              ))}
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  );
} 