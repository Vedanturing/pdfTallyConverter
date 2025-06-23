import { useState, useCallback } from 'react';
import { format, isValid, parse } from 'date-fns';

export interface ValidationError {
  field: string;
  message: string;
  type: 'error' | 'warning';
}

export interface ValidationResult {
  isValid: boolean;
  errors: ValidationError[];
}

export interface FieldValidation {
  [key: string]: ValidationResult;
}

export interface RowValidation {
  [rowId: string]: FieldValidation;
}

export interface ValidationSummary {
  totalErrors: number;
  totalWarnings: number;
  byField: {
    [field: string]: {
      errors: number;
      warnings: number;
    };
  };
}

const useValidation = () => {
  const [validations, setValidations] = useState<RowValidation>({});
  const [summary, setSummary] = useState<ValidationSummary>({
    totalErrors: 0,
    totalWarnings: 0,
    byField: {},
  });

  const validateDate = useCallback((value: string): ValidationResult => {
    const errors: ValidationError[] = [];
    
    if (!value) {
      errors.push({
        field: 'date',
        message: 'Date is required',
        type: 'error',
      });
      return { isValid: false, errors };
    }

    // Try to parse the date
    const parsedDate = parse(value, 'dd-MM-yyyy', new Date());
    if (!isValid(parsedDate)) {
      errors.push({
        field: 'date',
        message: 'Invalid date format. Use DD-MM-YYYY',
        type: 'error',
      });
    } else {
      // Check if date is in future
      if (parsedDate > new Date()) {
        errors.push({
          field: 'date',
          message: 'Future date detected',
          type: 'warning',
        });
      }
    }

    return { isValid: errors.length === 0, errors };
  }, []);

  const validateAmount = useCallback((value: number | string): ValidationResult => {
    const errors: ValidationError[] = [];
    const numValue = typeof value === 'string' ? parseFloat(value) : value;

    if (value === undefined || value === null || value === '') {
      errors.push({
        field: 'amount',
        message: 'Amount is required',
        type: 'error',
      });
    } else if (isNaN(numValue)) {
      errors.push({
        field: 'amount',
        message: 'Amount must be a number',
        type: 'error',
      });
    } else {
      if (numValue < 0) {
        errors.push({
          field: 'amount',
          message: 'Amount cannot be negative',
          type: 'error',
        });
      }
      if (numValue === 0) {
        errors.push({
          field: 'amount',
          message: 'Zero amount detected',
          type: 'warning',
        });
      }
      if (numValue > 1000000) {
        errors.push({
          field: 'amount',
          message: 'Large amount detected',
          type: 'warning',
        });
      }
    }

    return { isValid: errors.filter(e => e.type === 'error').length === 0, errors };
  }, []);

  const validateLedgerName = useCallback((value: string): ValidationResult => {
    const errors: ValidationError[] = [];

    if (!value) {
      errors.push({
        field: 'ledgerName',
        message: 'Ledger name is required',
        type: 'error',
      });
    } else if (value.length < 3) {
      errors.push({
        field: 'ledgerName',
        message: 'Ledger name must be at least 3 characters',
        type: 'error',
      });
    }

    return { isValid: errors.length === 0, errors };
  }, []);

  const validateNarration = useCallback((value: string): ValidationResult => {
    const errors: ValidationError[] = [];

    if (!value) {
      errors.push({
        field: 'narration',
        message: 'Narration is required',
        type: 'error',
      });
    }

    return { isValid: errors.length === 0, errors };
  }, []);

  const validateRow = useCallback((row: any) => {
    const fieldValidations: FieldValidation = {
      date: validateDate(row.date),
      amount: validateAmount(row.amount),
      ledgerName: validateLedgerName(row.ledgerName),
      narration: validateNarration(row.narration),
    };

    return fieldValidations;
  }, [validateDate, validateAmount, validateLedgerName, validateNarration]);

  const validateData = useCallback((data: any[]) => {
    const newValidations: RowValidation = {};
    let totalErrors = 0;
    let totalWarnings = 0;
    const byField: ValidationSummary['byField'] = {};

    data.forEach((row) => {
      const rowValidation = validateRow(row);
      newValidations[row.id] = rowValidation;

      // Update summary
      Object.entries(rowValidation).forEach(([field, validation]) => {
        if (!byField[field]) {
          byField[field] = { errors: 0, warnings: 0 };
        }

        validation.errors.forEach((error) => {
          if (error.type === 'error') {
            totalErrors++;
            byField[field].errors++;
          } else {
            totalWarnings++;
            byField[field].warnings++;
          }
        });
      });
    });

    return { validations: newValidations, summary: { totalErrors, totalWarnings, byField } };
  }, [validateRow]);

  const updateValidations = useCallback((data: any[]) => {
    const result = validateData(data);
    setValidations(result.validations);
    setSummary(result.summary);
  }, [validateData]);

  const getFieldValidation = useCallback((rowId: string, field: string): ValidationResult | null => {
    return validations[rowId]?.[field] || null;
  }, [validations]);

  const isRowValid = useCallback((rowId: string): boolean => {
    const rowValidation = validations[rowId];
    if (!rowValidation) return false;
    return Object.values(rowValidation).every(v => v.isValid);
  }, [validations]);

  const isDataValid = useCallback((): boolean => {
    return summary.totalErrors === 0;
  }, [summary]);

  return {
    validations,
    summary,
    validateData,
    updateValidations,
    validateRow,
    getFieldValidation,
    isRowValid,
    isDataValid,
  };
};

export default useValidation; 