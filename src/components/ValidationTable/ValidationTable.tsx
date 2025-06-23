import React, { useRef } from 'react';
import { ValidationCell, ValidationIssue } from './ValidationCell';
import { FunnelIcon } from '@heroicons/react/24/outline';

interface ValidationTableProps {
  data: Record<string, any>[];
  issues: Record<number, Record<string, ValidationIssue>>;
  ignoredIssues: Set<string>;
  showOnlyIssues: boolean;
  onToggleShowOnlyIssues: () => void;
  onEdit: (rowIndex: number, field: string, value: string) => void;
  onIgnore: (rowIndex: number, field: string) => void;
  onApplyFix: (rowIndex: number, field: string, fix: string) => void;
}

export const ValidationTable: React.FC<ValidationTableProps> = ({
  data,
  issues,
  ignoredIssues,
  showOnlyIssues,
  onToggleShowOnlyIssues,
  onEdit,
  onIgnore,
  onApplyFix,
}) => {
  const tableRef = useRef<HTMLDivElement>(null);

  const scrollToCell = (rowIndex: number, field: string) => {
    const row = tableRef.current?.querySelector(`[data-row-index="${rowIndex}"]`);
    const cell = row?.querySelector(`[data-field="${field}"]`);
    cell?.scrollIntoView({ behavior: 'smooth', block: 'center' });
  };

  const hasIssuesInRow = (rowIndex: number) => {
    return issues[rowIndex] && Object.keys(issues[rowIndex]).length > 0;
  };

  const filteredData = showOnlyIssues
    ? data.filter((_, index) => hasIssuesInRow(index))
    : data;

  const columns = data.length > 0 ? Object.keys(data[0]) : [];

  return (
    <div className="flex-1 overflow-hidden flex flex-col">
      <div className="p-4 border-b border-gray-200 flex justify-between items-center">
        <h2 className="text-lg font-medium text-gray-900">Validation Results</h2>
        <button
          onClick={onToggleShowOnlyIssues}
          className={`flex items-center px-3 py-1 rounded ${
            showOnlyIssues
              ? 'bg-blue-100 text-blue-800'
              : 'bg-gray-100 text-gray-800'
          }`}
        >
          <FunnelIcon className="h-4 w-4 mr-2" />
          {showOnlyIssues ? 'Showing Issues Only' : 'Show All Rows'}
        </button>
      </div>

      <div
        ref={tableRef}
        className="flex-1 overflow-auto"
      >
        <table className="min-w-full divide-y divide-gray-200">
          <thead className="bg-gray-50 sticky top-0">
            <tr>
              {columns.map((column) => (
                <th
                  key={column}
                  className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider"
                >
                  {column}
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-gray-200">
            {filteredData.map((row, rowIndex) => (
              <tr
                key={rowIndex}
                data-row-index={rowIndex}
                className={hasIssuesInRow(rowIndex) ? 'bg-red-50/10' : ''}
              >
                {columns.map((field) => {
                  const issue = issues[rowIndex]?.[field];
                  const isIgnored = ignoredIssues.has(
                    `${rowIndex}-${field}`
                  );

                  return (
                    <ValidationCell
                      key={`${rowIndex}-${field}`}
                      value={String(row[field] ?? '')}
                      field={field}
                      rowIndex={rowIndex}
                      issue={issue}
                      isIgnored={isIgnored}
                      onEdit={(value) => onEdit(rowIndex, field, value)}
                      onIgnore={() => onIgnore(rowIndex, field)}
                      onApplyFix={(fix) => onApplyFix(rowIndex, field, fix)}
                    />
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};
