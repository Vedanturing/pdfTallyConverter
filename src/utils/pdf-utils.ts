import { API_URL } from '../config';

export async function convertPDF(pdfUrl: string) {
  try {
    const formData = new FormData();
    const response = await fetch(pdfUrl);
    const blob = await response.blob();
    formData.append('file', blob, 'document.pdf');

    const uploadResponse = await fetch(`${API_URL}/upload`, {
      method: 'POST',
      body: formData,
    });

    if (!uploadResponse.ok) {
      throw new Error('Upload failed');
    }

    const { file_id } = await uploadResponse.json();

    const convertResponse = await fetch(`${API_URL}/convert/${file_id}`, {
      method: 'POST',
    });

    if (!convertResponse.ok) {
      throw new Error('Conversion failed');
    }

    const dataResponse = await fetch(`${API_URL}/get-data/${file_id}`);
    if (!dataResponse.ok) {
      throw new Error('Failed to get converted data');
    }

    const data = await dataResponse.json();

    if (data && data.rows) {
      // Transform the data to match the expected format
      return data.rows.map((row: any, index: number) => ({
        id: `row-${index}`,
        date: row.Date || row.DATE || '',
        voucherNo: row['Voucher No'] || row['VOUCHER NO'] || '',
        ledgerName: row['Ledger Name'] || row['LEDGER NAME'] || '',
        amount: typeof row.Amount === 'string' ? parseFloat(row.Amount.replace(/[^0-9.-]+/g, '')) || 0 : row.Amount || 0,
        narration: row.Narration || row.NARRATION || '',
        balance: typeof row.Balance === 'string' ? parseFloat(row.Balance.replace(/[^0-9.-]+/g, '')) || 0 : row.Balance || 0
      }));
    }

    throw new Error('Invalid data format');
  } catch (error) {
    console.error('Error in convertPDF:', error);
    throw error;
  }
} 