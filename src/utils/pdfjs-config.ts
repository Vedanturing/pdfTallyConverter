import { pdfjs } from 'react-pdf';

// Initialize PDF.js worker
const workerSrc = '/pdf.worker.min.js';

// Configure react-pdf to use the worker
pdfjs.GlobalWorkerOptions.workerSrc = workerSrc;

export const initPdfWorker = () => {
  try {
    // Ensure react-pdf is using the worker
    if (!pdfjs.GlobalWorkerOptions.workerSrc) {
      pdfjs.GlobalWorkerOptions.workerSrc = workerSrc;
    }
    console.log('PDF.js worker initialized with URL:', workerSrc);
  } catch (error) {
    console.error('Error initializing PDF.js worker:', error);
    throw error;
  }
};

export const cleanupPdfWorker = () => {
  try {
    pdfjs.GlobalWorkerOptions.workerSrc = '';
  } catch (error) {
    console.error('Error cleaning up PDF.js worker:', error);
  }
}; 