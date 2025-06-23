import React, { useState, useEffect } from 'react';
import * as pdfjsLib from 'pdfjs-dist';
import { FileMetadata } from '../types/file';

interface FilePreviewProps {
  file: File;
  metadata?: FileMetadata;
  className?: string;
  previewUrl: string;
}

export const FilePreview: React.FC<FilePreviewProps> = ({
  file,
  metadata,
  className = '',
  previewUrl
}) => {
  const [preview, setPreview] = useState<string>('');
  const [currentPage, setCurrentPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const loadPreview = async () => {
      try {
        setLoading(true);
        setError(null);

        if (file.type === 'application/pdf') {
          const arrayBuffer = await file.arrayBuffer();
          const pdf = await pdfjsLib.getDocument(arrayBuffer).promise;
          setTotalPages(pdf.numPages);

          const page = await pdf.getPage(currentPage);
          const viewport = page.getViewport({ scale: 1.0 });
          
          const canvas = document.createElement('canvas');
          const context = canvas.getContext('2d');
          canvas.height = viewport.height;
          canvas.width = viewport.width;
          
          await page.render({
            canvasContext: context!,
            viewport: viewport
          }).promise;
          
          setPreview(canvas.toDataURL());
        } else if (file.type.startsWith('image/')) {
          setPreview(URL.createObjectURL(file));
        }
      } catch (err) {
        setError('Failed to load preview');
        console.error('Preview error:', err);
      } finally {
        setLoading(false);
      }
    };

    loadPreview();

    return () => {
      if (file.type.startsWith('image/')) {
        URL.revokeObjectURL(preview);
      }
    };
  }, [file, currentPage]);

  if (loading) {
    return (
      <div className={`flex items-center justify-center p-4 ${className}`}>
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-gray-900"></div>
      </div>
    );
  }

  if (error) {
    return (
      <div className={`flex items-center justify-center p-4 text-red-500 ${className}`}>
        {error}
      </div>
    );
  }

  return (
    <div className={`flex flex-col items-center ${className}`}>
      <img
        src={preview}
        alt={`Preview of ${file.name}`}
        className="max-w-full max-h-[500px] object-contain rounded-lg shadow-sm"
      />
      
      {file.type === 'application/pdf' && totalPages > 1 && (
        <div className="flex items-center gap-4 mt-4">
          <button
            onClick={() => setCurrentPage(p => Math.max(1, p - 1))}
            disabled={currentPage <= 1}
            className="px-3 py-1 bg-gray-100 rounded hover:bg-gray-200 disabled:opacity-50"
            aria-label="Previous page"
          >
            ←
          </button>
          <span className="text-sm text-gray-600">
            Page {currentPage} of {totalPages}
          </span>
          <button
            onClick={() => setCurrentPage(p => Math.min(totalPages, p + 1))}
            disabled={currentPage >= totalPages}
            className="px-3 py-1 bg-gray-100 rounded hover:bg-gray-200 disabled:opacity-50"
            aria-label="Next page"
          >
            →
          </button>
        </div>
      )}

      {metadata && (
        <div className="mt-4 text-sm text-gray-500 w-full">
          <p>File size: {(metadata.file_size / 1024).toFixed(2)} KB</p>
          {metadata.type === 'pdf' && metadata.num_pages && (
            <p>Pages: {metadata.num_pages}</p>
          )}
          {metadata.type === 'image' && (
            <p>Dimensions: {metadata.width}×{metadata.height}</p>
          )}
        </div>
      )}
    </div>
  );
};

export default FilePreview; 