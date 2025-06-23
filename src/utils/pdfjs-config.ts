import { pdfjs as reactPdfJs } from 'react-pdf';
import * as pdfjsLib from 'pdfjs-dist';

// Initialize PDF.js worker
const workerSrc = '/pdf.worker.min.js';

// Configure both react-pdf and pdfjs-dist to use the same worker
reactPdfJs.GlobalWorkerOptions.workerSrc = workerSrc;
pdfjsLib.GlobalWorkerOptions.workerSrc = workerSrc;

export const initPdfWorker = () => {
  try {
    // Ensure both libraries are using the worker
    if (!reactPdfJs.GlobalWorkerOptions.workerSrc) {
      reactPdfJs.GlobalWorkerOptions.workerSrc = workerSrc;
    }
    if (!pdfjsLib.GlobalWorkerOptions.workerSrc) {
      pdfjsLib.GlobalWorkerOptions.workerSrc = workerSrc;
    }
    console.log('PDF.js worker initialized with URL:', workerSrc);
  } catch (error) {
    console.error('Error initializing PDF.js worker:', error);
    throw error;
  }
};

export const cleanupPdfWorker = () => {
  try {
    reactPdfJs.GlobalWorkerOptions.workerSrc = '';
    pdfjsLib.GlobalWorkerOptions.workerSrc = '';
  } catch (error) {
    console.error('Error cleaning up PDF.js worker:', error);
  }
};

export { pdfjsLib }; 