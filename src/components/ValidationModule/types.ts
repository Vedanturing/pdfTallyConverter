import { FinancialEntry, ValidationIssue } from '../../types/financial';

export interface ValidationState {
  data: FinancialEntry[];
  issues: Record<string, CellValidationIssue[]>;
  ignoredIssues: Set<string>;
  customRules: ValidationRule[];
  editableRows: Set<string>;
  fixedIssues: Set<string>;
  summary: ValidationSummary;
}

export interface CellValidationIssue extends ValidationIssue {
  cellId: string;
  rowIndex: number;
  columnKey: keyof FinancialEntry;
  originalValue: any;
  suggestedValue?: any;
}

export interface ValidationRule {
  id: string;
  name: string;
  field: keyof FinancialEntry;
  operator: 'equals' | 'notEquals' | 'greaterThan' | 'lessThan' | 'contains' | 'before' | 'after';
  value: any;
  severity: 'error' | 'warning';
}

export interface ValidationSummary {
  total: number;
  errors: number;
  warnings: number;
  fixed: number;
  ignored: number;
}

export interface ValidationContextType {
  state: ValidationState;
  actions: {
    toggleRowEditable: (rowIndex: number) => void;
    updateCell: (rowIndex: number, field: keyof FinancialEntry, value: any) => void;
    ignoreIssue: (issueId: string) => void;
    addCustomRule: (rule: Omit<ValidationRule, 'id'>) => void;
    removeCustomRule: (ruleId: string) => void;
    applyAutoFixes: () => void;
    updateValidationState: (newState: Partial<ValidationState>) => void;
    updateSearchResults: (results: FinancialEntry[]) => void;
  };
}

export interface ValidationGridProps {
  data: FinancialEntry[];
  issues: Record<string, CellValidationIssue[]>;
  onCellEdit: (rowIndex: number, field: keyof FinancialEntry, value: any) => void;
  onIgnoreIssue: (issueId: string) => void;
  editableRows: Set<string>;
  onToggleEditable: (rowIndex: number) => void;
}

export interface ValidationToolbarProps {
  summary: ValidationSummary;
  onApplyAutoFixes: () => void;
  onAddRule: (rule: Omit<ValidationRule, 'id'>) => void;
  onSearch: (query: string) => void;
} 