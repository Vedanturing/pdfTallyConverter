import { pdfjs } from 'react-pdf';

export const setupPdfWorker = () => {
  try {
    console.log('Setting up PDF.js worker...'); // Debug log
    pdfjs.GlobalWorkerOptions.workerSrc = '/pdf.worker.min.js';
    console.log('Worker URL:', pdfjs.GlobalWorkerOptions.workerSrc); // Debug log
    return true;
  } catch (error) {
    console.error('Error setting up PDF.js worker:', error);
    return false;
  }
}; 