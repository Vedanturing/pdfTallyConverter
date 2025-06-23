import { pdfjs } from 'react-pdf';
import pdfjsWorker from 'pdfjs-dist/build/pdf.worker.entry';

export const setupPdfWorker = () => {
  try {
    console.log('Setting up PDF.js worker...'); // Debug log
    pdfjs.GlobalWorkerOptions.workerSrc = pdfjsWorker;
    console.log('Worker URL:', pdfjs.GlobalWorkerOptions.workerSrc); // Debug log
    return true;
  } catch (error) {
    console.error('Error setting up PDF.js worker:', error);
    return false;
  }
}; 