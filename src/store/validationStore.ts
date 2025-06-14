import create from 'zustand';
import { ValidationResult } from '../types/ValidationTypes';

interface ValidationStore {
  validationResults: ValidationResult[];
  isValidating: boolean;
  error: string | null;
  setValidationResults: (results: ValidationResult[]) => void;
  setIsValidating: (status: boolean) => void;
  setError: (error: string | null) => void;
  clearValidation: () => void;
}

export const useValidationStore = create<ValidationStore>((set) => ({
  validationResults: [],
  isValidating: false,
  error: null,
  setValidationResults: (results) => set({ validationResults: results }),
  setIsValidating: (status) => set({ isValidating: status }),
  setError: (error) => set({ error }),
  clearValidation: () => set({ validationResults: [], error: null }),
}));
