import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import i18n from '../i18n/config';
import { LanguageCode } from '../i18n/config';

interface LanguageState {
  currentLanguage: LanguageCode;
  setLanguage: (language: LanguageCode) => void;
  initializeLanguage: () => void;
}

export const useLanguageStore = create<LanguageState>()(
  persist(
    (set, get) => ({
      currentLanguage: 'en',

      setLanguage: (language: LanguageCode) => {
        set({ currentLanguage: language });
        i18n.changeLanguage(language);
        
        // Update document lang attribute for better accessibility
        document.documentElement.lang = language;
        
        // Store in localStorage for persistence across sessions
        localStorage.setItem('i18nextLng', language);
      },

      initializeLanguage: () => {
        const stored = localStorage.getItem('i18nextLng') as LanguageCode;
        const browserLang = navigator.language.split('-')[0] as LanguageCode;
        const supportedLanguages: LanguageCode[] = ['en', 'hi', 'mr'];
        
        // Priority: stored > browser > default
        const language = stored && supportedLanguages.includes(stored) 
          ? stored 
          : supportedLanguages.includes(browserLang) 
            ? browserLang 
            : 'en';

        get().setLanguage(language);
      },
    }),
    {
      name: 'language-settings',
      partialize: (state) => ({ currentLanguage: state.currentLanguage }),
    }
  )
); 