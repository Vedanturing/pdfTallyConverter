export interface FinancialEntry {
  id: string;
  date: string;
  voucherNo: string;
  ledgerName: string;
  amount: number | string;
  narration: string;
  balance: number | string;
  [key: string]: string | number;
}

export interface ValidationSummary {
  totalErrors: number;
  totalWarnings: number;
  details: {
    [field: string]: {
      errors: number;
      warnings: number;
    };
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