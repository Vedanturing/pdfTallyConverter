export interface ValidationResult {
  field: string;
  value: string;
  isValid: boolean;
  message?: string;
}

export interface PDFValidationResult {
  isProtected: boolean;
  needsPassword: boolean;
  pageCount: number;
  metadata: {
    title?: string;
    author?: string;
    subject?: string;
    keywords?: string[];
    creationDate?: string;
    modificationDate?: string;
  };
}

export interface PDFProcessingOptions {
  password?: string;
  skipPasswordCheck?: boolean;
  extractText?: boolean;
  extractMetadata?: boolean;
}

export interface ValidationError {
  code: string;
  message: string;
  field?: string;
}
