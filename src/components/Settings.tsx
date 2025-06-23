import React, { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { 
  Cog6ToothIcon, 
  XMarkIcon,
  GlobeAltIcon,
  SunIcon,
  MoonIcon 
} from '@heroicons/react/24/outline';
import LanguageSelector from './LanguageSelector';

interface SettingsProps {
  isOpen: boolean;
  onClose: () => void;
  theme: 'light' | 'dark';
  onToggleTheme: () => void;
}

const Settings: React.FC<SettingsProps> = ({ 
  isOpen, 
  onClose, 
  theme, 
  onToggleTheme 
}) => {
  const { t } = useTranslation();

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 overflow-y-auto">
      <div className="fixed inset-0 bg-black bg-opacity-50" onClick={onClose} />
      
      <div className="relative min-h-screen flex items-center justify-center p-4">
        <div className="relative bg-white dark:bg-gray-800 rounded-lg shadow-xl max-w-md w-full">
          {/* Header */}
          <div className="flex items-center justify-between p-6 border-b border-gray-200 dark:border-gray-700">
            <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
              <Cog6ToothIcon className="w-5 h-5 inline mr-2" />
              {t('settings.title')}
            </h2>
            <button
              onClick={onClose}
              className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-300"
            >
              <XMarkIcon className="w-5 h-5" />
            </button>
          </div>

          {/* Content */}
          <div className="p-6 space-y-6">
            {/* Language Settings */}
            <div className="space-y-4">
              <h3 className="text-sm font-medium text-gray-900 dark:text-gray-100 flex items-center">
                <GlobeAltIcon className="w-4 h-4 mr-2" />
                {t('settings.language.title')}
              </h3>
              <LanguageSelector showLabel={false} />
            </div>

            {/* Theme Settings */}
            <div className="space-y-4">
              <h3 className="text-sm font-medium text-gray-900 dark:text-gray-100 flex items-center">
                {theme === 'light' ? (
                  <SunIcon className="w-4 h-4 mr-2" />
                ) : (
                  <MoonIcon className="w-4 h-4 mr-2" />
                )}
                {t('settings.theme')}
              </h3>
              <button
                onClick={onToggleTheme}
                className="
                  flex items-center justify-between w-full px-4 py-2
                  border border-gray-300 dark:border-gray-600
                  rounded-md shadow-sm
                  bg-white dark:bg-gray-700
                  text-gray-900 dark:text-gray-100
                  hover:bg-gray-50 dark:hover:bg-gray-600
                  focus:outline-none focus:ring-2 focus:ring-blue-500
                  text-sm
                "
              >
                <span>
                  {theme === 'light' ? t('settings.theme') : t('settings.theme')}
                </span>
                {theme === 'light' ? (
                  <MoonIcon className="w-4 h-4" />
                ) : (
                  <SunIcon className="w-4 h-4" />
                )}
              </button>
            </div>
          </div>

          {/* Footer */}
          <div className="px-6 py-4 border-t border-gray-200 dark:border-gray-700">
            <button
              onClick={onClose}
              className="
                w-full px-4 py-2
                bg-blue-600 hover:bg-blue-700
                text-white font-medium
                rounded-md
                focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2
                text-sm
              "
            >
              {t('common.close')}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Settings; 