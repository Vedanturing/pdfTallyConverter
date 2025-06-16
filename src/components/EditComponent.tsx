import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { API_URL } from '../config';
import {
  ArrowPathIcon,
  DocumentArrowDownIcon,
  EyeIcon,
  PencilIcon,
  ArrowDownTrayIcon,
  ExclamationCircleIcon,
  CheckCircleIcon,
  AdjustmentsHorizontalIcon,
} from '@heroicons/react/24/outline';
import toast from 'react-hot-toast';
import FinancialTable from './FinancialTable';
import WorkflowStepper from './WorkflowStepper';
import { useLocation, useNavigate } from 'react-router-dom';
import ValidationTable from './ValidationTable/ValidationTable';

interface ValidationRule {
  id: string;
  name: string;
  field: string;
  type: 'required' | 'format' | 'range' | 'custom';
  condition?: string;
  value?: string | number;
  enabled: boolean;
}

interface FinancialEntry {
  id: string;
  date: string;
  voucherNo: string;
  ledgerName: string;
  amount: number | string;
  narration: string;
  errors?: string[];
}

const DEFAULT_RULES: ValidationRule[] = [
  {
    id: 'required-date',
    name: 'Date Required',
    field: 'date',
    type: 'required',
    enabled: true
  },
  {
    id: 'date-format',
    name: 'Valid Date Format',
    field: 'date',
    type: 'format',
    condition: 'YYYY-MM-DD',
    enabled: true
  },
  {
    id: 'required-voucher',
    name: 'Voucher Number Required',
    field: 'voucherNo',
    type: 'required',
    enabled: true
  },
  {
    id: 'required-ledger',
    name: 'Ledger Name Required',
    field: 'ledgerName',
    type: 'required',
    enabled: true
  },
  {
    id: 'amount-required',
    name: 'Amount Required',
    field: 'amount',
    type: 'required',
    enabled: true
  },
  {
    id: 'amount-range',
    name: 'Amount Range',
    field: 'amount',
    type: 'range',
    condition: 'between',
    value: '0,1000000',
    enabled: true
  }
];

interface LocationState {
  fileId: string;
  convertedFormats: string[];
}

const EditComponent: React.FC = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const [loading, setLoading] = useState(true);
  const [data, setData] = useState<any[]>([]);
  const { fileId } = location.state as LocationState || {};

  useEffect(() => {
    const fetchData = async () => {
      if (!fileId) {
        navigate('/');
        return;
      }

      try {
        setLoading(true);
        const response = await axios.get(`${API_URL}/get-data/${fileId}`);
        
        // Transform the data to match the ValidationTable format
        const transformedData = response.data.rows.map((row: any) => {
          const transformedRow: Record<string, any> = {};
          Object.entries(row).forEach(([key, value]) => {
            transformedRow[key] = {
              value: value || '',
              metadata: {
                error: false,
                confidence: 1.0,
                status: 'original'
              }
            };
          });
          return transformedRow;
        });

        setData(transformedData);
      } catch (error) {
        console.error('Error fetching data:', error);
        toast.error('Failed to fetch data');
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, [fileId, navigate]);

  if (!fileId) {
    return null;
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="animate-spin rounded-full h-32 w-32 border-t-2 border-b-2 border-blue-500"></div>
      </div>
    );
  }

  if (!data.length) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-xl text-red-500">No data available</div>
      </div>
    );
  }

  return (
    <div className="container mx-auto px-4 py-8">
      <h1 className="text-2xl font-bold mb-6">Validate and Edit Data</h1>
      <ValidationTable fileId={fileId} data={data} />
    </div>
  );
};

export default EditComponent; 