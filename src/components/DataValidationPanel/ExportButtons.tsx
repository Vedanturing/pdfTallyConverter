import React from 'react';
import { toast } from 'react-hot-toast';

interface Props {
  fileId: string;
  onSave: () => Promise<any>;
}

export default function ExportButtons({ fileId, onSave }: Props) {
  const handleExport = async (format: 'xlsx' | 'csv' | 'xml') => {
    try {
      // First save the current changes
      const saveResult = await onSave();
      
      // Then download the file
      const response = await fetch(`/api/download/${fileId}/${format}`);
      if (!response.ok) throw new Error(`Failed to download ${format} file`);
      
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      
      // Get filename from headers or generate default
      let filename = `corrected_data.${format}`;
      const contentDisposition = response.headers.get('content-disposition');
      if (contentDisposition) {
        const filenameMatch = contentDisposition.match(/filename[^;=\n]*=((['"]).*?\2|[^;\n]*)/);
        if (filenameMatch) {
          filename = filenameMatch[1].replace(/['"]/g, '');
        }
      }
      
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
      
      toast.success(`Exported as ${format.toUpperCase()}`);
    } catch (error) {
      toast.error(`Failed to export as ${format.toUpperCase()}`);
      console.error('Export error:', error);
    }
  };

  return (
    <div className="space-x-2">
      <button
        onClick={() => handleExport('xlsx')}
        className="px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700"
      >
        Export XLSX
      </button>
      <button
        onClick={() => handleExport('csv')}
        className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
      >
        Export CSV
      </button>
      <button
        onClick={() => handleExport('xml')}
        className="px-4 py-2 bg-purple-600 text-white rounded hover:bg-purple-700"
      >
        Export XML
      </button>
    </div>
  );
} 