export interface CellMetadata {
  status?: 'original' | 'edited' | 'corrected' | 'ignored' | 'needs-review';
  confidence?: number;
}

export interface TableCell {
  value: string;
  isEdited: boolean;
  metadata?: CellMetadata;
}

export interface TableRow {
  id: TableCell;
  [key: string]: TableCell;
}

export interface TableData {
  headers: string[];
  rows: TableRow[];
}

export interface EditHistory {
  timestamp: string;
  rowId: string;
  columnKey: string;
  oldValue: string;
  newValue: string;
}

export interface ValidationError {
  rowId: string;
  columnKey: string;
  message: string;
}

export interface ConversionResult {
  headers: string[];
  rows: {
    [key: string]: string;
  }[];
  validationData?: {
    errors: ValidationError[];
  };
}

export interface SavePayload {
  fileId: string;
  originalData: TableData;
  modifiedData: TableData;
  editHistory: EditHistory[];
} 