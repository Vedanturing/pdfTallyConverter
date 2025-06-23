export interface FinancialEntry {
  date: string;
  description: string;
  amount: number;
  balance?: number;
  gstin?: string;
  taxRate?: number;
  voucherNo?: string;
  [key: string]: any;
}

export interface ValidationIssue {
  message: string;
  severity: 'error' | 'warning';
  type: 'amount' | 'date' | 'gstin' | 'tax' | 'voucher' | 'other';
  suggestedFix?: string;
}

export interface ValidationResult {
  row: number;
  issues: ValidationIssue[];
}

export interface ValidationSummary {
  critical: number;
  warning: number;
  info: number;
  total: number;
  empty_fields: Array<{
    row: number;
    fields: string[];
  }>;
  numeric_issues: Array<{
    row: number;
    field: string;
    value: string;
  }>;
  date_issues: Array<{
    row: number;
    value: string;
  }>;
  duplicate_entries: Array<{
    row: number;
    entry: {
      date: string;
      amount: number;
      description: string;
    };
  }>;
}

export interface ValidationResponse {
  status: string;
  message: string;
  validationResults: {
    results: ValidationResult[];
    financial: {
      is_valid: boolean;
      errors: Array<{
        row: number;
        errors: string[];
      }>;
    };
    summary: ValidationSummary;
  };
}

export interface AuditEntry {
  id: string;
  timestamp: string;
  userId: string;
  action: 'create' | 'update' | 'delete' | 'validate' | 'add_note';
  entityType: 'cell' | 'row' | 'table';
  entityId: string;
  field?: string;
  oldValue?: string;
  newValue?: string;
  note?: string;
}

export interface ValidationError {
  rowIndex: number;
  field: keyof FinancialEntry;
  message: string;
}

export interface CorrectionNote {
  rowIndex: number;
  field: keyof FinancialEntry;
  originalValue: string | number;
  correctedValue: string | number;
  note: string;
} 