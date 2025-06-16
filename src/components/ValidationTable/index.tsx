import React from 'react';

interface ValidationTableProps {
  data: any[];
}

const ValidationTable: React.FC<ValidationTableProps> = ({ data }) => {
  if (!data || data.length === 0) {
    return (
      <div className="text-center py-4 text-gray-500">
        No data available
      </div>
    );
  }

  const headers = Object.keys(data[0]);

  const formatValue = (value: any, header: string) => {
    if (header.toLowerCase().includes('amount') || header.toLowerCase().includes('balance')) {
      const numValue = parseFloat(value);
      return isNaN(numValue) ? value : numValue.toLocaleString('en-IN', {
        maximumFractionDigits: 2,
        minimumFractionDigits: 2
      });
    }
    return value || '';
  };

  const getCellClassName = (value: any, header: string) => {
    const baseClasses = 'px-4 py-2 text-sm border border-gray-200';
    if (!value || value === '') {
      return `${baseClasses} bg-red-50`;
    }
    if (header.toLowerCase().includes('amount') || header.toLowerCase().includes('balance')) {
      const numValue = parseFloat(value);
      if (isNaN(numValue)) {
        return `${baseClasses} bg-red-50`;
      }
      return `${baseClasses} text-right font-mono`;
    }
    return baseClasses;
  };

  return (
    <div className="overflow-x-auto">
      <table className="min-w-full divide-y divide-gray-200 border border-gray-200">
        <thead className="bg-gray-50">
          <tr>
            {headers.map((header) => (
              <th
                key={header}
                scope="col"
                className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider border border-gray-200"
              >
                {header}
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="bg-white divide-y divide-gray-200">
          {data.map((row, rowIndex) => (
            <tr key={rowIndex} className="hover:bg-gray-50">
              {headers.map((header) => (
                <td
                  key={`${rowIndex}-${header}`}
                  className={getCellClassName(row[header], header)}
                >
                  {formatValue(row[header], header)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export default ValidationTable;
