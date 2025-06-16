import React, { useState, useEffect } from 'react';
import PDFConverter from './components/PDFConverter';
import ValidationComponent from './components/ValidationComponent';
import Layout from './components/Layout';
import { FinancialEntry } from './types/financial';
import { initPdfWorker, cleanupPdfWorker } from './utils/pdfjs-config';

const App: React.FC = () => {
  const [convertedData, setConvertedData] = useState<FinancialEntry[] | null>(null);
  const [pdfUrl, setPdfUrl] = useState<string | null>(null);
  const [isValidating, setIsValidating] = useState(false);

  useEffect(() => {
    // Initialize PDF worker when app mounts
    initPdfWorker();

    // Cleanup worker when app unmounts
    return () => {
      cleanupPdfWorker();
    };
  }, []);

  const handleConvert = (data: FinancialEntry[]) => {
    setConvertedData(data);
  };

  const handlePdfUrl = (url: string) => {
    setPdfUrl(url);
  };

  const handleStartValidation = () => {
    setIsValidating(true);
  };

  const handleBackToConversion = () => {
    setIsValidating(false);
  };

  return (
    <Layout>
      {isValidating && convertedData ? (
        <ValidationComponent
          data={convertedData}
          pdfUrl={pdfUrl}
          onBack={handleBackToConversion}
          onDataChange={setConvertedData}
        />
      ) : (
        <PDFConverter
          onConvert={handleConvert}
          onPdfUrl={handlePdfUrl}
          onStartValidation={handleStartValidation}
        />
      )}
    </Layout>
  );
};

export default App; 