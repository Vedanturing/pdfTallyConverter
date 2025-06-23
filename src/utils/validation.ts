import { FinancialEntry } from '../types/financial';

export interface ValidationError {
  type: 'error' | 'warning';
  message: string;
  field: string;
}

export interface ValidationResult {
  isValid: boolean;
  errors: ValidationError[];
}

export const validateAmount = (amount: number | string): ValidationError[] => {
  const errors: ValidationError[] = [];
  const numAmount = typeof amount === 'string' ? parseFloat(amount) : amount;

  if (isNaN(numAmount)) {
    errors.push({
      type: 'error',
      message: 'Amount must be a valid number',
      field: 'amount',
    });
  } else if (numAmount < 0) {
    errors.push({
      type: 'warning',
      message: 'Negative amount detected',
      field: 'amount',
    });
  }

  return errors;
};

export const validateDate = (date: string): ValidationError[] => {
  const errors: ValidationError[] = [];
  const dateRegex = /^\d{2}-\d{2}-\d{4}$/;

  if (!dateRegex.test(date)) {
    errors.push({
      type: 'error',
      message: 'Date must be in DD-MM-YYYY format',
      field: 'date',
    });
    return errors;
  }

  const [day, month, year] = date.split('-').map(Number);
  const dateObj = new Date(year, month - 1, day);

  if (
    dateObj.getDate() !== day ||
    dateObj.getMonth() + 1 !== month ||
    dateObj.getFullYear() !== year
  ) {
    errors.push({
      type: 'error',
      message: 'Invalid date',
      field: 'date',
    });
  }

  return errors;
};

export const validateVoucherNo = (voucherNo: string): ValidationError[] => {
  const errors: ValidationError[] = [];
  const voucherRegex = /^[A-Z0-9-]+$/;

  if (!voucherRegex.test(voucherNo)) {
    errors.push({
      type: 'error',
      message: 'Voucher number must contain only uppercase letters, numbers, and hyphens',
      field: 'voucherNo',
    });
  }

  return errors;
};

export const validateLedgerName = (ledgerName: string): ValidationError[] => {
  const errors: ValidationError[] = [];

  if (ledgerName.trim().length === 0) {
    errors.push({
      type: 'error',
      message: 'Ledger name cannot be empty',
      field: 'ledgerName',
    });
  }

  return errors;
};

export const validateNarration = (narration: string): ValidationError[] => {
  const errors: ValidationError[] = [];

  if (narration.trim().length === 0) {
    errors.push({
      type: 'warning',
      message: 'Narration is empty',
      field: 'narration',
    });
  }

  return errors;
};

export const validateEntry = (entry: FinancialEntry): ValidationResult => {
  const errors: ValidationError[] = [
    ...validateDate(entry.date),
    ...validateVoucherNo(entry.voucherNo),
    ...validateLedgerName(entry.ledgerName),
    ...validateAmount(entry.amount),
    ...validateNarration(entry.narration),
  ];

  return {
    isValid: errors.filter(error => error.type === 'error').length === 0,
    errors,
  };
};

export const validateTable = (data: FinancialEntry[]): Map<string, ValidationResult> => {
  const validationMap = new Map<string, ValidationResult>();

  data.forEach(entry => {
    validationMap.set(entry.id, validateEntry(entry));
  });

  return validationMap;
};

export const isRowValid = (validationMap: Map<string, ValidationResult>, rowId: string): boolean => {
  const result = validationMap.get(rowId);
  return result ? result.isValid : false;
};

export const isCellValid = (
  validationMap: Map<string, ValidationResult>,
  rowId: string,
  field: string
): boolean => {
  const result = validationMap.get(rowId);
  if (!result) return false;
  return !result.errors.some(error => error.field === field && error.type === 'error');
}; 