import React, { createContext, useContext, useReducer, useCallback } from 'react';
import { FinancialEntry } from '../../types/financial';
import {
  ValidationState,
  ValidationContextType,
  ValidationRule,
  CellValidationIssue
} from './types';

type ValidationAction =
  | { type: 'SET_DATA'; payload: FinancialEntry[] }
  | { type: 'TOGGLE_EDITABLE'; payload: number }
  | { type: 'UPDATE_CELL'; payload: { rowIndex: number; field: keyof FinancialEntry; value: any } }
  | { type: 'IGNORE_ISSUE'; payload: string }
  | { type: 'ADD_RULE'; payload: ValidationRule }
  | { type: 'REMOVE_RULE'; payload: string }
  | { type: 'SET_ISSUES'; payload: Record<string, CellValidationIssue[]> }
  | { type: 'UPDATE_SUMMARY' }
  | { type: 'UPDATE_VALIDATION_STATE'; payload: Partial<ValidationState> }
  | { type: 'UPDATE_SEARCH_RESULTS'; payload: FinancialEntry[] };

const initialState: ValidationState = {
  data: [],
  issues: {},
  ignoredIssues: new Set(),
  customRules: [],
  editableRows: new Set(),
  fixedIssues: new Set(),
  summary: {
    total: 0,
    errors: 0,
    warnings: 0,
    fixed: 0,
    ignored: 0
  }
};

function validationReducer(state: ValidationState, action: ValidationAction): ValidationState {
  switch (action.type) {
    case 'SET_DATA':
      return {
        ...state,
        data: action.payload
      };

    case 'TOGGLE_EDITABLE':
      const newEditableRows = new Set(state.editableRows);
      const rowKey = `row-${action.payload}`;
      if (newEditableRows.has(rowKey)) {
        newEditableRows.delete(rowKey);
      } else {
        newEditableRows.add(rowKey);
      }
      return {
        ...state,
        editableRows: newEditableRows
      };

    case 'UPDATE_CELL':
      const { rowIndex, field, value } = action.payload;
      const newData = [...state.data];
      newData[rowIndex] = {
        ...newData[rowIndex],
        [field]: value
      };
      return {
        ...state,
        data: newData
      };

    case 'IGNORE_ISSUE':
      const newIgnoredIssues = new Set(state.ignoredIssues);
      newIgnoredIssues.add(action.payload);
      return {
        ...state,
        ignoredIssues: newIgnoredIssues
      };

    case 'ADD_RULE':
      return {
        ...state,
        customRules: [...state.customRules, action.payload]
      };

    case 'REMOVE_RULE':
      return {
        ...state,
        customRules: state.customRules.filter(rule => rule.id !== action.payload)
      };

    case 'SET_ISSUES':
      return {
        ...state,
        issues: action.payload
      };

    case 'UPDATE_SUMMARY':
      const allIssues = Object.values(state.issues).flat();
      const summary = {
        total: allIssues.length,
        errors: allIssues.filter(i => i.severity === 'error').length,
        warnings: allIssues.filter(i => i.severity === 'warning').length,
        fixed: state.fixedIssues.size,
        ignored: state.ignoredIssues.size
      };
      return {
        ...state,
        summary
      };

    case 'UPDATE_VALIDATION_STATE':
      return {
        ...state,
        ...action.payload
      };

    case 'UPDATE_SEARCH_RESULTS':
      return {
        ...state,
        data: action.payload
      };

    default:
      return state;
  }
}

const ValidationContext = createContext<ValidationContextType | null>(null);

export function ValidationProvider({ children }: { children: React.ReactNode }) {
  const [state, dispatch] = useReducer(validationReducer, initialState);

  const toggleRowEditable = useCallback((rowIndex: number) => {
    dispatch({ type: 'TOGGLE_EDITABLE', payload: rowIndex });
  }, []);

  const updateCell = useCallback((rowIndex: number, field: keyof FinancialEntry, value: any) => {
    dispatch({ type: 'UPDATE_CELL', payload: { rowIndex, field, value } });
  }, []);

  const ignoreIssue = useCallback((issueId: string) => {
    dispatch({ type: 'IGNORE_ISSUE', payload: issueId });
  }, []);

  const addCustomRule = useCallback((rule: Omit<ValidationRule, 'id'>) => {
    const newRule = {
      ...rule,
      id: `rule-${Date.now()}`
    };
    dispatch({ type: 'ADD_RULE', payload: newRule });
  }, []);

  const removeCustomRule = useCallback((ruleId: string) => {
    dispatch({ type: 'REMOVE_RULE', payload: ruleId });
  }, []);

  const updateValidationState = useCallback((newState: Partial<ValidationState>) => {
    dispatch({ type: 'UPDATE_VALIDATION_STATE', payload: newState });
  }, []);

  const updateSearchResults = useCallback((results: FinancialEntry[]) => {
    dispatch({ type: 'UPDATE_SEARCH_RESULTS', payload: results });
  }, []);

  const applyAutoFixes = useCallback(() => {
    // Implement auto-fix logic here
    const newData = [...state.data];
    const newFixedIssues = new Set(state.fixedIssues);

    Object.entries(state.issues).forEach(([cellId, issues]) => {
      const [rowIndex, field] = cellId.split('-');
      const issue = issues.find(i => i.severity === 'warning' && i.suggestedValue);
      
      if (issue && issue.suggestedValue) {
        newData[Number(rowIndex)] = {
          ...newData[Number(rowIndex)],
          [field]: issue.suggestedValue
        };
        newFixedIssues.add(cellId);
      }
    });

    dispatch({ type: 'UPDATE_VALIDATION_STATE', payload: {
      data: newData,
      fixedIssues: newFixedIssues
    }});
  }, [state.data, state.issues, state.fixedIssues]);

  const value = {
    state,
    actions: {
      toggleRowEditable,
      updateCell,
      ignoreIssue,
      addCustomRule,
      removeCustomRule,
      applyAutoFixes,
      updateValidationState,
      updateSearchResults
    }
  };

  return (
    <ValidationContext.Provider value={value}>
      {children}
    </ValidationContext.Provider>
  );
}

export function useValidation() {
  const context = useContext(ValidationContext);
  if (!context) {
    throw new Error('useValidation must be used within a ValidationProvider');
  }
  return context;
} 