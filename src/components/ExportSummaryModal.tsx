import React from 'react';
import { Dialog } from '@headlessui/react';
import { DocumentDuplicateIcon, DocumentArrowDownIcon, XMarkIcon } from '@heroicons/react/24/outline';
import { FinancialEntry, ValidationSummary } from '../types/financial';
import html2pdf from 'html2pdf.js';

interface ExportSummaryModalProps {
  isOpen: boolean;
  onClose: () => void;
  exportData: {
    fileName: string;
    fileType: string;
    data: FinancialEntry[];
    validationSummary?: ValidationSummary;
    exportFormat: string;
    clientName: string;
    exportSuccess: boolean;
    downloadPath?: string;
  };
}

const ExportSummaryModal: React.FC<ExportSummaryModalProps> = ({
  isOpen,
  onClose,
  exportData
}) => {
  const {
    fileName,
    fileType,
    data,
    validationSummary,
    exportFormat,
    clientName,
    exportSuccess,
    downloadPath
  } = exportData;

  const dateRange = React.useMemo(() => {
    if (!data.length) return 'No transactions';
    const dates = data.map(entry => new Date(entry.date));
    const minDate = new Date(Math.min(...dates.map(d => d.getTime()))).toLocaleDateString();
    const maxDate = new Date(Math.max(...dates.map(d => d.getTime()))).toLocaleDateString();
    return `${minDate} to ${maxDate}`;
  }, [data]);

  const totals = React.useMemo(() => {
    return data.reduce((acc, entry) => {
      if (entry.amount > 0) {
        acc.credit += entry.amount;
      } else {
        acc.debit += Math.abs(entry.amount);
      }
      return acc;
    }, { credit: 0, debit: 0 });
  }, [data]);

  const handleCopySummary = () => {
    const summaryText = `
Export Summary for ${clientName}
File: ${fileName}
Type: ${fileType}
Format: ${exportFormat}
Date Range: ${dateRange}
Total Transactions: ${data.length}
Total Credit: ₹${totals.credit.toFixed(2)}
Total Debit: ₹${totals.debit.toFixed(2)}
${validationSummary ? `
Validation Summary:
- Critical Issues: ${validationSummary.critical}
- Warnings: ${validationSummary.warning}
- Info: ${validationSummary.info}
- Total Issues: ${validationSummary.total}
` : ''}
Status: ${exportSuccess ? 'Export Successful' : 'Export Failed'}
${downloadPath ? `Download Location: ${downloadPath}` : ''}
    `.trim();

    navigator.clipboard.writeText(summaryText);
  };

  const handleDownloadPDF = () => {
    const element = document.getElementById('export-summary');
    if (!element) return;

    const opt = {
      margin: 1,
      filename: `${clientName}_export_summary.pdf`,
      image: { type: 'jpeg', quality: 0.98 },
      html2canvas: { scale: 2 },
      jsPDF: { unit: 'in', format: 'letter', orientation: 'portrait' }
    };

    html2pdf().set(opt).from(element).save();
  };

  return (
    <Dialog
      open={isOpen}
      onClose={onClose}
      className="fixed inset-0 z-50 overflow-y-auto"
    >
      <div className="flex min-h-screen items-center justify-center">
        <Dialog.Overlay className="fixed inset-0 bg-black opacity-30" />

        <div className="relative bg-white rounded-lg shadow-xl max-w-2xl w-full mx-4">
          <div className="absolute right-4 top-4">
            <button
              onClick={onClose}
              className="text-gray-400 hover:text-gray-500"
            >
              <XMarkIcon className="h-6 w-6" />
            </button>
          </div>

          <div className="p-6" id="export-summary">
            <Dialog.Title className="text-2xl font-semibold text-gray-900 mb-6">
              Export Summary
            </Dialog.Title>

            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <h4 className="text-sm font-medium text-gray-500">Client Name</h4>
                  <p className="text-base text-gray-900">{clientName}</p>
                </div>
                <div>
                  <h4 className="text-sm font-medium text-gray-500">File Type</h4>
                  <p className="text-base text-gray-900">{fileType}</p>
                </div>
                <div>
                  <h4 className="text-sm font-medium text-gray-500">Export Format</h4>
                  <p className="text-base text-gray-900">{exportFormat}</p>
                </div>
                <div>
                  <h4 className="text-sm font-medium text-gray-500">Date Range</h4>
                  <p className="text-base text-gray-900">{dateRange}</p>
                </div>
                <div>
                  <h4 className="text-sm font-medium text-gray-500">Total Transactions</h4>
                  <p className="text-base text-gray-900">{data.length}</p>
                </div>
              </div>

              <div className="bg-gray-50 p-4 rounded-lg">
                <h4 className="text-sm font-medium text-gray-500 mb-2">Transaction Totals</h4>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <p className="text-sm text-gray-500">Total Credit</p>
                    <p className="text-lg font-semibold text-green-600">₹{totals.credit.toFixed(2)}</p>
                  </div>
                  <div>
                    <p className="text-sm text-gray-500">Total Debit</p>
                    <p className="text-lg font-semibold text-red-600">₹{totals.debit.toFixed(2)}</p>
                  </div>
                </div>
              </div>

              {validationSummary && (
                <div className="bg-gray-50 p-4 rounded-lg">
                  <h4 className="text-sm font-medium text-gray-500 mb-2">Validation Summary</h4>
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <p className="text-sm text-gray-500">Critical Issues</p>
                      <p className="text-base text-red-600">{validationSummary.critical}</p>
                    </div>
                    <div>
                      <p className="text-sm text-gray-500">Warnings</p>
                      <p className="text-base text-yellow-600">{validationSummary.warning}</p>
                    </div>
                    <div>
                      <p className="text-sm text-gray-500">Info</p>
                      <p className="text-base text-blue-600">{validationSummary.info}</p>
                    </div>
                    <div>
                      <p className="text-sm text-gray-500">Total Issues</p>
                      <p className="text-base text-gray-900">{validationSummary.total}</p>
                    </div>
                  </div>
                </div>
              )}

              <div className="bg-gray-50 p-4 rounded-lg">
                <h4 className="text-sm font-medium text-gray-500 mb-2">Export Status</h4>
                <p className={`text-base ${exportSuccess ? 'text-green-600' : 'text-red-600'}`}>
                  {exportSuccess ? 'Export Successful' : 'Export Failed'}
                </p>
                {downloadPath && (
                  <p className="text-sm text-gray-500 mt-1">
                    Download Location: {downloadPath}
                  </p>
                )}
              </div>
            </div>
          </div>

          <div className="px-6 py-4 bg-gray-50 rounded-b-lg flex justify-end space-x-4">
            <button
              onClick={handleCopySummary}
              className="flex items-center px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-md hover:bg-gray-50"
            >
              <DocumentDuplicateIcon className="h-5 w-5 mr-2" />
              Copy Summary
            </button>
            <button
              onClick={handleDownloadPDF}
              className="flex items-center px-4 py-2 text-sm font-medium text-white bg-blue-600 rounded-md hover:bg-blue-700"
            >
              <DocumentArrowDownIcon className="h-5 w-5 mr-2" />
              Download PDF
            </button>
          </div>
        </div>
      </div>
    </Dialog>
  );
};

export default ExportSummaryModal; 