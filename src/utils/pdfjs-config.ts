import { pdfjs } from 'react-pdf';

// Import the worker directly from node_modules
const pdfjsWorker = new URL(
  'pdfjs-dist/build/pdf.worker.min.js',
  import.meta.url
);

export const initPdfWorker = () => {
  try {
    // Set the worker source URL
    pdfjs.GlobalWorkerOptions.workerSrc = pdfjsWorker.href;
  } catch (error) {
    console.error('Error initializing PDF.js worker:', error);
    throw error;
  }
};

export const cleanupPdfWorker = () => {
  try {
    // No need to manually cleanup as the worker will be managed by react-pdf
    pdfjs.GlobalWorkerOptions.workerSrc = '';
  } catch (error) {
    console.error('Error cleaning up PDF.js worker:', error);
  }
}; 