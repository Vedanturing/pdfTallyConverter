import React from 'react';
import { TableData } from '../../types/DataValidation';

interface Props {
  originalData: TableData;
  modifiedData: TableData;
  onClose: () => void;
}

export default function DiffModal({ originalData, modifiedData, onClose }: Props) {
  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4">
      <div className="bg-white rounded-lg max-w-6xl w-full max-h-[90vh] overflow-hidden">
        <div className="p-4 border-b flex justify-between items-center">
          <h2 className="text-lg font-semibold">Modified vs Original Data</h2>
          <button
            onClick={onClose}
            className="text-gray-500 hover:text-gray-700"
          >
            ✕
          </button>
        </div>
        
        <div className="grid grid-cols-2 gap-4 p-4 overflow-auto max-h-[calc(90vh-4rem)]">
          <div>
            <h3 className="font-semibold mb-2">Modified Data</h3>
            <pre className="bg-gray-50 p-4 rounded overflow-auto">
              {JSON.stringify(modifiedData, null, 2)}
            </pre>
          </div>
          <div>
            <h3 className="font-semibold mb-2">Original Data</h3>
            <pre className="bg-gray-50 p-4 rounded overflow-auto">
              {JSON.stringify(originalData, null, 2)}
            </pre>
          </div>
        </div>
      </div>
    </div>
  );
} 