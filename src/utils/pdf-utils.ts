import { API_URL } from '../config';

// Cache for converted data to avoid re-processing
const convertedDataCache = new Map<string, any>();
const CACHE_EXPIRY = 30 * 60 * 1000; // 30 minutes

interface CachedData {
  data: any;
  timestamp: number;
}

// Helper function to generate cache key
function getCacheKey(fileData: Blob): string {
  return `${fileData.size}_${fileData.type}_${Date.now().toString(36)}`;
}

// Helper function to check if cache is valid
function isCacheValid(cachedData: CachedData): boolean {
  return Date.now() - cachedData.timestamp < CACHE_EXPIRY;
}

// Optimized retry function
async function retryOperation<T>(
  operation: () => Promise<T>,
  maxRetries: number = 3,
  delay: number = 1000
): Promise<T> {
  let lastError: Error;
  
  for (let i = 0; i < maxRetries; i++) {
    try {
      return await operation();
    } catch (error) {
      lastError = error as Error;
      if (i === maxRetries - 1) break;
      
      // Exponential backoff
      await new Promise(resolve => setTimeout(resolve, delay * Math.pow(2, i)));
    }
  }
  
  throw lastError!;
}

// Progress callback type
type ProgressCallback = (progress: number, stage: string) => void;

export async function convertPDF(
  pdfUrl: string, 
  onProgress?: ProgressCallback
): Promise<any> {
  try {
    onProgress?.(10, 'Downloading PDF...');
    
    // Download PDF with timeout and better error handling
    const response = await fetch(pdfUrl, {
      signal: AbortSignal.timeout(60000) // 1 minute timeout
    });
    
    if (!response.ok) {
      throw new Error(`Failed to download PDF: ${response.status} ${response.statusText}`);
    }
    
    const blob = await response.blob();
    
    // Check cache first
    const cacheKey = getCacheKey(blob);
    const cachedData = convertedDataCache.get(cacheKey);
    
    if (cachedData && isCacheValid(cachedData)) {
      onProgress?.(100, 'Using cached data');
      return cachedData.data;
    }
    
    onProgress?.(20, 'Uploading file...');
    
    // Upload with optimized FormData
    const formData = new FormData();
    formData.append('file', blob, 'document.pdf');

    const uploadResponse = await retryOperation(async () => {
      const response = await fetch(`${API_URL}/upload`, {
        method: 'POST',
        body: formData,
        signal: AbortSignal.timeout(120000) // 2 minute timeout for upload
      });
      
      if (!response.ok) {
        throw new Error(`Upload failed: ${response.status} ${response.statusText}`);
      }
      
      return response;
    });

    const uploadResult = await uploadResponse.json();
    const fileId = uploadResult.file_id || uploadResult.id;
    
    if (!fileId) {
      throw new Error('No file ID returned from upload');
    }

    onProgress?.(50, 'Converting document...');

    // Convert with retry logic
    const convertResponse = await retryOperation(async () => {
      const response = await fetch(`${API_URL}/convert/${fileId}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        signal: AbortSignal.timeout(180000) // 3 minute timeout for conversion
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Conversion failed: ${response.status} - ${errorText}`);
      }

      return response;
    });

    onProgress?.(80, 'Processing data...');

    const convertResult = await convertResponse.json();
    
    // Handle different response formats
    let processedData;
    
    if (convertResult.data?.rows) {
      processedData = convertResult.data.rows;
    } else if (convertResult.rows) {
      processedData = convertResult.rows;
    } else if (Array.isArray(convertResult)) {
      processedData = convertResult;
    } else {
      throw new Error('Invalid response format from conversion API');
    }

    onProgress?.(90, 'Formatting data...');

    // Transform and normalize the data
    const transformedData = processedData.map((row: any, index: number) => ({
      id: row.id || `row-${index}`,
      date: normalizeField(row, ['date', 'Date', 'DATE']),
      voucherNo: normalizeField(row, ['voucherNo', 'voucher_no', 'Voucher No', 'VOUCHER NO']),
      ledgerName: normalizeField(row, ['ledgerName', 'ledger_name', 'Ledger Name', 'LEDGER NAME']),
      amount: normalizeAmount(row, ['amount', 'Amount', 'AMOUNT']),
      narration: normalizeField(row, ['narration', 'Narration', 'NARRATION', 'description', 'Description']),
      balance: normalizeAmount(row, ['balance', 'Balance', 'BALANCE'])
    }));

    // Cache the result
    convertedDataCache.set(cacheKey, {
      data: transformedData,
      timestamp: Date.now()
    });

    onProgress?.(100, 'Complete');
    
    return transformedData;

  } catch (error) {
    console.error('Error in convertPDF:', error);
    
    // Enhanced error handling with user-friendly messages
    if (error instanceof TypeError && error.message.includes('fetch')) {
      throw new Error('Network error. Please check your internet connection and try again.');
    } else if (error instanceof Error) {
      if (error.message.includes('timeout') || error.name === 'TimeoutError') {
        throw new Error('Request timed out. The file might be too large or the server is busy. Please try again.');
      } else if (error.message.includes('404')) {
        throw new Error('Service not found. Please make sure the backend server is running.');
      } else if (error.message.includes('500')) {
        throw new Error('Server error. Please try again in a few moments.');
      }
    }
    
    throw error;
  }
}

// Helper function to normalize field values
function normalizeField(row: any, possibleKeys: string[]): string {
  for (const key of possibleKeys) {
    if (row[key] !== undefined && row[key] !== null) {
      return String(row[key]).trim();
    }
  }
  return '';
}

// Helper function to normalize amount values
function normalizeAmount(row: any, possibleKeys: string[]): number {
  for (const key of possibleKeys) {
    if (row[key] !== undefined && row[key] !== null) {
      const value = row[key];
      if (typeof value === 'number') {
        return value;
      }
      if (typeof value === 'string') {
        const parsed = parseFloat(value.replace(/[^\d.-]/g, ''));
        return isNaN(parsed) ? 0 : parsed;
      }
    }
  }
  return 0;
}

// Clean up expired cache entries
export function cleanupCache(): void {
  const now = Date.now();
  for (const [key, cachedData] of convertedDataCache.entries()) {
    if (!isCacheValid(cachedData)) {
      convertedDataCache.delete(key);
    }
  }
}

// Get cache stats
export function getCacheStats(): { size: number; entries: number } {
  let totalSize = 0;
  for (const [key, cachedData] of convertedDataCache.entries()) {
    totalSize += JSON.stringify(cachedData.data).length;
  }
  
  return {
    size: totalSize,
    entries: convertedDataCache.size
  };
} 