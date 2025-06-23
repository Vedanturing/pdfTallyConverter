import { create } from 'zustand';
import { ValidationResult } from '../utils/validation';

interface ValidationState {
  validationMap: Map<string, ValidationResult>;
  setValidationMap: (map: Map<string, ValidationResult>) => void;
  clearValidation: () => void;
}

const useValidationStore = create<ValidationState>((set) => ({
  validationMap: new Map(),
  setValidationMap: (map) => set({ validationMap: map }),
  clearValidation: () => set({ validationMap: new Map() }),
}));

export default useValidationStore;
