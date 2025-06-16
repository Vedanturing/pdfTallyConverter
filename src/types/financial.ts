export interface FinancialEntry {
  id: string;
  date: string;
  voucherNo: string;
  ledgerName: string;
  amount: number;
  narration: string;
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