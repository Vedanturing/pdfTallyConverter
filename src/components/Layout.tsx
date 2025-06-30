import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { FileText, Settings, HelpCircle, Moon, Sun } from 'lucide-react';
import { Button } from './ui/button';
import { Link, useLocation } from 'react-router-dom';
import {
  DocumentTextIcon,
  DocumentArrowUpIcon,
  DocumentMagnifyingGlassIcon,
  ArrowPathIcon,
  CheckCircleIcon,
  ArrowDownTrayIcon,
  BanknotesIcon,
  CalculatorIcon,
  ScaleIcon,
} from '@heroicons/react/24/outline';
import { useTranslation } from 'react-i18next';
import SettingsModal from './Settings';
import { useAuthStore } from '../store/authStore';
import { UserCircleIcon, ClockIcon, ArrowRightStartOnRectangleIcon, CogIcon, TrashIcon } from '@heroicons/react/24/outline';

const getNavigation = (t: any) => [
  { name: t('navigation.upload'), href: '/', icon: DocumentTextIcon },
  { name: 'Multiple Upload', href: '/multiple-upload', icon: DocumentArrowUpIcon },
  { name: t('navigation.preview'), href: '/preview', icon: DocumentMagnifyingGlassIcon },
  { name: t('navigation.convert'), href: '/convert', icon: ArrowPathIcon },
  { name: t('navigation.validate'), href: '/validate', icon: CheckCircleIcon },
  { name: t('navigation.export'), href: '/export', icon: ArrowDownTrayIcon },
  { name: t('navigation.bankMatcher'), href: '/bank-matcher', icon: BanknotesIcon },
  { name: t('navigation.gstHelper'), href: '/gst-helper', icon: CalculatorIcon },
  { name: 'GST Reconciliation', href: '/gst-reconciliation', icon: ScaleIcon },
];

interface LayoutProps {
  children: React.ReactNode;
  theme: 'light' | 'dark';
  onToggleTheme: () => void;
  onShowAuth?: () => void;
  onShowHistory?: () => void;
  onShowDeleteAccount?: () => void;
}

export const Layout: React.FC<LayoutProps> = ({ 
  children, 
  theme, 
  onToggleTheme,
  onShowAuth,
  onShowHistory,
  onShowDeleteAccount
}) => {
  const location = useLocation();
  const { t } = useTranslation();
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [showUserMenu, setShowUserMenu] = useState(false);
  const { user, isAuthenticated, logout } = useAuthStore();

  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-50 to-gray-100 dark:from-gray-900 dark:to-gray-800">
      {/* Header */}
      <header className="sticky top-0 z-50 w-full border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
        <div className="container flex h-14 items-center">
          <div className="mr-4 flex">
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              className="flex items-center space-x-2"
            >
              <FileText className="h-6 w-6 text-primary" />
              <span className="font-bold text-xl">PDF Tally Converter</span>
            </motion.div>
          </div>
          <div className="flex flex-1 items-center justify-between space-x-2 md:justify-end">
            <nav className="flex items-center space-x-4">
              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                className="w-9 h-9 flex items-center justify-center rounded-md border border-input bg-background hover:bg-accent hover:text-accent-foreground"
                onClick={onToggleTheme}
              >
                {theme === 'dark' ? (
                  <Sun className="h-5 w-5" />
                ) : (
                  <Moon className="h-5 w-5" />
                )}
              </motion.button>
              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={() => setIsSettingsOpen(true)}
                className="w-9 h-9 flex items-center justify-center rounded-md border border-input bg-background hover:bg-accent hover:text-accent-foreground"
                title={t('common.settings')}
              >
                <Settings className="h-5 w-5" />
              </motion.button>
              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                className="w-9 h-9 flex items-center justify-center rounded-md border border-input bg-background hover:bg-accent hover:text-accent-foreground"
              >
                <HelpCircle className="h-5 w-5" />
              </motion.button>
              
              {/* Authentication Section */}
              {isAuthenticated ? (
                <div className="relative">
                  <motion.button
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    onClick={() => setShowUserMenu(!showUserMenu)}
                    className="flex items-center space-x-2 px-3 py-2 rounded-md border border-input bg-background hover:bg-accent hover:text-accent-foreground"
                  >
                    <UserCircleIcon className="h-5 w-5" />
                    <span className="text-sm hidden md:block">{user?.name}</span>
                  </motion.button>
                  
                  {/* User Menu Dropdown */}
                  {showUserMenu && (
                    <div className="absolute right-0 mt-2 w-48 bg-white dark:bg-gray-800 rounded-md shadow-lg ring-1 ring-black ring-opacity-5 z-50">
                      <div className="py-1">
                        <div className="px-3 py-2 text-sm text-gray-500 border-b">
                          {user?.email}
                        </div>
                        {onShowHistory && (
                          <button
                            onClick={() => {
                              onShowHistory();
                              setShowUserMenu(false);
                            }}
                            className="flex items-center w-full px-3 py-2 text-sm text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700"
                          >
                            <ClockIcon className="h-4 w-4 mr-2" />
                            My History
                          </button>
                        )}
                        <button
                          onClick={() => {
                            setIsSettingsOpen(true);
                            setShowUserMenu(false);
                          }}
                          className="flex items-center w-full px-3 py-2 text-sm text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700"
                        >
                          <CogIcon className="h-4 w-4 mr-2" />
                          Account Settings
                        </button>
                        {onShowDeleteAccount && (
                          <button
                            onClick={() => {
                              onShowDeleteAccount();
                              setShowUserMenu(false);
                            }}
                            className="flex items-center w-full px-3 py-2 text-sm text-red-600 hover:bg-red-50 dark:hover:bg-red-900"
                          >
                            <TrashIcon className="h-4 w-4 mr-2" />
                            Delete Account
                          </button>
                        )}
                        <button
                          onClick={() => {
                            logout();
                            setShowUserMenu(false);
                          }}
                          className="flex items-center w-full px-3 py-2 text-sm text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700"
                        >
                          <ArrowRightStartOnRectangleIcon className="h-4 w-4 mr-2" />
                          Logout
                        </button>
                      </div>
                    </div>
                  )}
                </div>
              ) : (
                onShowAuth && (
                  <motion.button
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    onClick={onShowAuth}
                    className="px-4 py-2 bg-primary text-primary-foreground rounded-md text-sm font-medium hover:bg-primary/90"
                  >
                    Sign In
                  </motion.button>
                )
              )}
            </nav>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto py-6 px-4 min-h-[calc(100vh-3.5rem)]">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
        >
          <nav className="bg-white shadow-sm">
            <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
              <div className="flex h-16 justify-between">
                <div className="flex">
                  <div className="flex flex-shrink-0 items-center">
                    <h1 className="text-xl font-semibold text-gray-900">PDF Tally Converter</h1>
                  </div>
                  <div className="hidden sm:ml-6 sm:flex sm:space-x-8">
                    {getNavigation(t).map((item) => {
                      const isActive = location.pathname === item.href;
                      return (
                        <Link
                          key={item.name}
                          to={item.href}
                          className={`inline-flex items-center px-1 pt-1 text-sm font-medium ${
                            isActive
                              ? 'border-b-2 border-indigo-500 text-gray-900'
                              : 'border-b-2 border-transparent text-gray-500 hover:border-gray-300 hover:text-gray-700'
                          }`}
                        >
                          <item.icon className="h-5 w-5 mr-1" />
                          {item.name}
                        </Link>
                      );
                    })}
                  </div>
                </div>
              </div>
            </div>
          </nav>
          {children}
        </motion.div>
      </main>

      {/* Footer */}
      <footer className="border-t bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
        <div className="container flex h-14 items-center justify-center text-sm text-muted-foreground">
          <p>© 2024 PDF Tally Converter. All rights reserved.</p>
        </div>
      </footer>

      {/* Settings Modal */}
      <SettingsModal
        isOpen={isSettingsOpen}
        onClose={() => setIsSettingsOpen(false)}
        theme={theme}
        onToggleTheme={onToggleTheme}
      />
    </div>
  );
};

export default Layout; 