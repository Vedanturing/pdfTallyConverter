import React, { useState } from 'react';
import { ArrowLeftIcon } from '@heroicons/react/24/outline';
import { Document, Page } from 'react-pdf';
import FinancialTable from './FinancialTable';
import { FinancialEntry } from '../types/financial';

interface ValidationComponentProps {
  data: FinancialEntry[];
  pdfUrl: string | null;
  onBack: () => void;
  onDataChange: (data: FinancialEntry[]) => void;
}

const ValidationComponent: React.FC<ValidationComponentProps> = ({
  data,
  pdfUrl,
  onBack,
  onDataChange,
}) => {
  const [currentPage, setCurrentPage] = useState(1);
  const [numPages, setNumPages] = useState(0);

  const onDocumentLoadSuccess = ({ numPages }: { numPages: number }) => {
    setNumPages(numPages);
  };

  return (
    <div className="space-y-4">
      <div className="flex justify-between items-center mb-4">
        <button
          onClick={onBack}
          className="flex items-center px-3 py-2 text-gray-600 hover:text-gray-900"
        >
          <ArrowLeftIcon className="h-5 w-5 mr-2" />
          Back to Conversion
        </button>
      </div>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div className="bg-white rounded-lg shadow-lg p-4">
          <Document file={pdfUrl} onLoadSuccess={onDocumentLoadSuccess}>
            <Page
              pageNumber={currentPage}
              width={400}
              renderTextLayer={true}
              renderAnnotationLayer={true}
            />
          </Document>
          {numPages > 1 && (
            <div className="flex justify-center mt-4 space-x-2">
              <button
                onClick={() => setCurrentPage(Math.max(1, currentPage - 1))}
                disabled={currentPage === 1}
                className="px-3 py-1 bg-blue-500 text-white rounded disabled:bg-gray-300"
              >
                Previous
              </button>
              <span className="px-3 py-1">
                Page {currentPage} of {numPages}
              </span>
              <button
                onClick={() => setCurrentPage(Math.min(numPages, currentPage + 1))}
                disabled={currentPage === numPages}
                className="px-3 py-1 bg-blue-500 text-white rounded disabled:bg-gray-300"
              >
                Next
              </button>
            </div>
          )}
        </div>
        <div className="bg-white rounded-lg shadow-lg p-4">
          <FinancialTable
            data={data}
            onDataChange={onDataChange}
          />
        </div>
      </div>
    </div>
  );
};

export default ValidationComponent; 