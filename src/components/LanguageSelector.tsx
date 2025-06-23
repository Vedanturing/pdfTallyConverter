import React from 'react';
import { useTranslation } from 'react-i18next';
import { ChevronDownIcon, GlobeAltIcon } from '@heroicons/react/24/outline';
import { useLanguageStore } from '../store/languageStore';
import { languages, LanguageCode } from '../i18n/config';

interface LanguageSelectorProps {
  className?: string;
  showLabel?: boolean;
}

const LanguageSelector: React.FC<LanguageSelectorProps> = ({ 
  className = '', 
  showLabel = true 
}) => {
  const { t } = useTranslation();
  const { currentLanguage, setLanguage } = useLanguageStore();

  const handleLanguageChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
    const newLanguage = event.target.value as LanguageCode;
    setLanguage(newLanguage);
  };

  const currentLangData = languages.find(lang => lang.code === currentLanguage) || languages[0];

  return (
    <div className={`language-selector ${className}`}>
      {showLabel && (
        <label htmlFor="language-select" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
          <GlobeAltIcon className="w-4 h-4 inline mr-2" />
          {t('settings.language.select')}
        </label>
      )}
      
      <div className="relative">
        <select
          id="language-select"
          value={currentLanguage}
          onChange={handleLanguageChange}
          className="
            appearance-none block w-full px-3 py-2 pr-8
            border border-gray-300 dark:border-gray-600
            rounded-md shadow-sm
            bg-white dark:bg-gray-700
            text-gray-900 dark:text-gray-100
            focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500
            text-sm
          "
          aria-label={t('settings.language.select')}
        >
          {languages.map((language) => (
            <option key={language.code} value={language.code}>
              {language.nativeName}
            </option>
          ))}
        </select>
        
        <ChevronDownIcon className="absolute right-2 top-2.5 h-4 w-4 text-gray-400 pointer-events-none" />
      </div>
      
      <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">
        {t('common.language')}: {currentLangData.nativeName}
      </p>
    </div>
  );
};

export default LanguageSelector; 