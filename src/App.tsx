import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, useLocation, useNavigate, Link } from 'react-router-dom';
import { Layout } from './components/Layout';
import PDFConverter from './components/PDFConverter';
import FileUpload from './components/FileUpload';
import ViewComponent from './components/ViewComponent';
import ConvertComponent from './components/ConvertComponent';
import ValidationComponent from './components/ValidationComponent';
import ExportComponent from './components/ExportComponent';
import MultipleFileUpload from './components/MultipleFileUpload';
import WorkflowStepper from './components/WorkflowStepper';
import { Toaster } from 'react-hot-toast';
import { initPdfWorker, cleanupPdfWorker } from './utils/pdfjs-config';
import { FinancialEntry } from './types/financial';
import {
  DocumentArrowUpIcon,
  DocumentMagnifyingGlassIcon,
  TableCellsIcon,
  CheckCircleIcon,
  ArrowDownTrayIcon,
} from '@heroicons/react/24/outline';
import BankMatcher from './components/BankMatcher';
import GSTHelper from './components/GSTHelper';
import GSTReconciliation from './components/GSTReconciliation';

// Initialize i18n
import './i18n/config';
import { useLanguageStore } from './store/languageStore';
import { useAuthStore } from './store/authStore';
import { AuthModal } from './components/auth/AuthModal';
import { HistoryList } from './components/history/HistoryList';
import { DeleteAccountModal } from './components/auth/DeleteAccountModal';

const steps = [
  { title: 'Upload', icon: DocumentArrowUpIcon },
  { title: 'Preview', icon: DocumentMagnifyingGlassIcon },
  { title: 'Convert', icon: TableCellsIcon },
  { title: 'Validate', icon: CheckCircleIcon },
  { title: 'Export', icon: ArrowDownTrayIcon },
];

function AppContent() {
  const [theme, setTheme] = useState<'light' | 'dark'>('light');
  const [currentStep, setCurrentStep] = useState(0);
  const [showAuthModal, setShowAuthModal] = useState(false);
  const [showHistoryModal, setShowHistoryModal] = useState(false);
  const [showDeleteAccountModal, setShowDeleteAccountModal] = useState(false);
  const location = useLocation();
  const navigate = useNavigate();
  const { initializeLanguage } = useLanguageStore();
  const { checkAuth } = useAuthStore();

  useEffect(() => {
    // Initialize language settings
    initializeLanguage();
    
    // Check authentication state on app load
    checkAuth();
    
    // Initialize PDF worker
    initPdfWorker();

    // Check system preference for dark mode
    const isDarkMode = window.matchMedia('(prefers-color-scheme: dark)').matches;
    setTheme(isDarkMode ? 'dark' : 'light');

    // Set current step based on route
    if (location.pathname === '/') {
      setCurrentStep(0);
    } else if (location.pathname === '/preview') {
      setCurrentStep(1);
    } else if (location.pathname === '/convert') {
      setCurrentStep(2);
    } else if (location.pathname === '/history') {
      setShowHistoryModal(true);
    }

    // Cleanup PDF worker on unmount
    return () => {
      cleanupPdfWorker();
    };
  }, [location.pathname, checkAuth, initializeLanguage]);

  const toggleTheme = () => {
    setTheme(prevTheme => {
      const newTheme = prevTheme === 'light' ? 'dark' : 'light';
      document.documentElement.classList.toggle('dark', newTheme === 'dark');
      return newTheme;
    });
  };

  const handleConvert = (data: FinancialEntry[]) => {
    console.log('Conversion result:', data);
    setCurrentStep(2);
    navigate('/convert', {
      state: {
        data,
        currentStep: 2
      }
    });
  };

  const handlePdfUrl = (url: string) => {
    console.log('PDF URL:', url);
    setCurrentStep(1);
  };

  const handleStartValidation = () => {
    console.log('Starting validation...');
    setCurrentStep(3);
  };

  return (
    <div className={theme}>
      <Layout 
        theme={theme} 
        onToggleTheme={toggleTheme}
        onShowAuth={() => setShowAuthModal(true)}
        onShowHistory={() => setShowHistoryModal(true)}
        onShowDeleteAccount={() => setShowDeleteAccountModal(true)}
      >
        <div className="space-y-8">
          <div className="prose dark:prose-invert max-w-none">
            <h1 className="text-4xl font-bold tracking-tight">PDF Tally Converter</h1>
            <p className="text-lg text-muted-foreground">
              Convert your PDF bank statements into structured financial data with ease.
            </p>
          </div>
          
          <div className="max-w-4xl mx-auto">
            <WorkflowStepper steps={steps} currentStep={currentStep} />
          </div>

          <Routes>
            <Route 
              path="/" 
              element={
                <FileUpload />
              } 
            />
            <Route 
              path="/preview" 
              element={
                <ViewComponent
                  onNext={handleConvert}
                  onBack={() => {
                    setCurrentStep(0);
                    navigate('/');
                  }}
                />
              } 
            />
            <Route 
              path="/convert" 
              element={
                <ConvertComponent />
              } 
            />
            <Route 
              path="/convert/:fileId" 
              element={
                <ConvertComponent />
              } 
            />
            <Route 
              path="/validate" 
              element={
                <ValidationComponent />
              } 
            />
            <Route 
              path="/export" 
              element={
                <ExportComponent />
              } 
            />
            <Route 
              path="/bank-matcher" 
              element={
                <BankMatcher />
              } 
            />
            <Route 
              path="/gst-helper" 
              element={
                <GSTHelper />
              } 
            />
            <Route 
              path="/gst-reconciliation" 
              element={
                <GSTReconciliation />
              } 
            />
            <Route 
              path="/multiple-upload" 
              element={
                <MultipleFileUpload
                  onComplete={(results) => {
                    console.log('Multiple upload completed:', results);
                  }}
                />
              } 
            />
            <Route 
              path="/history" 
              element={
                <HistoryList 
                  onClose={() => {
                    setShowHistoryModal(false);
                    navigate('/');
                  }}
                />
              } 
            />
          </Routes>

          {/* Authentication Modal */}
          <AuthModal
            isOpen={showAuthModal}
            onClose={() => setShowAuthModal(false)}
            onSuccess={() => {
              setShowAuthModal(false);
            }}
          />

          {/* History Modal */}
          {showHistoryModal && (
            <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-50">
              <div className="bg-white rounded-lg p-4 max-w-6xl w-full mx-4 max-h-[90vh] overflow-auto">
                <HistoryList 
                  onClose={() => {
                    setShowHistoryModal(false);
                    navigate('/');
                  }}
                />
              </div>
            </div>
          )}

          {/* Delete Account Modal */}
          <DeleteAccountModal
            isOpen={showDeleteAccountModal}
            onClose={() => setShowDeleteAccountModal(false)}
            onSuccess={() => {
              setShowDeleteAccountModal(false);
            }}
          />
        </div>
      </Layout>
      <Toaster
        position="top-right"
        toastOptions={{
          className: 'toast-enter',
          duration: 4000,
          style: {
            background: 'hsl(var(--background))',
            color: 'hsl(var(--foreground))',
            border: '1px solid hsl(var(--border))',
          },
        }}
      />
    </div>
  );
}

function App() {
  return (
    <Router>
      <AppContent />
    </Router>
  );
}

export default App; 