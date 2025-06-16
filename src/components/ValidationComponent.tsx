import React, { useState, useEffect } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import axios from 'axios';
import { API_URL } from '../config';
import { FinancialEntry } from '../types/financial';
import toast from 'react-hot-toast';
import {
  ArrowLeftIcon,
  CheckCircleIcon,
  ExclamationCircleIcon,
  ArrowPathIcon,
} from '@heroicons/react/24/outline';

interface ValidationRule {
  field: keyof FinancialEntry;
  type: 'required' | 'numeric' | 'date' | 'gstin' | 'tax';
  message: string;
}

const VALIDATION_RULES: ValidationRule[] = [
  { field: 'date', type: 'date', message: 'Invalid date format' },
  { field: 'amount', type: 'numeric', message: 'Amount must be a number' },
  { field: 'balance', type: 'numeric', message: 'Balance must be a number' },
  { field: 'gstin', type: 'gstin', message: 'Invalid GSTIN format' },
  { field: 'taxRate', type: 'tax', message: 'Invalid tax rate' },
];

const ValidationComponent: React.FC = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const [data, setData] = useState<FinancialEntry[]>([]);
  const [errors, setErrors] = useState<Record<number, Record<string, string>>>({});
  const [loading, setLoading] = useState(false);
  const [fileId, setFileId] = useState<string>('');
  const [convertedFormats, setConvertedFormats] = useState<string[]>([]);

  useEffect(() => {
    const state = location.state as { fileId?: string; data?: FinancialEntry[]; convertedFormats?: string[] };
    if (!state?.fileId || !state?.data) {
      toast.error('No data to validate');
      navigate('/', { replace: true });
      return;
    }
    setFileId(state.fileId);
    setData(state.data);
    setConvertedFormats(state.convertedFormats || []);
    validateData(state.data);
  }, [location.state, navigate]);

  const validateField = (value: any, rule: ValidationRule): string | null => {
    if (!value && rule.type === 'required') return rule.message;
    
    switch (rule.type) {
      case 'numeric':
        return isNaN(Number(value)) ? rule.message : null;
      case 'date':
        return !Date.parse(value) ? rule.message : null;
      case 'gstin':
        return !/^[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[1-9A-Z]{1}Z[0-9A-Z]{1}$/.test(value) ? rule.message : null;
      case 'tax':
        const tax = Number(value);
        return isNaN(tax) || tax < 0 || tax > 100 ? rule.message : null;
      default:
        return null;
    }
  };

  const validateData = (entries: FinancialEntry[]) => {
    const newErrors: Record<number, Record<string, string>> = {};
    
    entries.forEach((entry, index) => {
      const rowErrors: Record<string, string> = {};
      
      VALIDATION_RULES.forEach(rule => {
        const error = validateField(entry[rule.field], rule);
        if (error) rowErrors[rule.field] = error;
      });
      
      if (Object.keys(rowErrors).length > 0) {
        newErrors[index] = rowErrors;
      }
    });
    
    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleCellEdit = (index: number, field: keyof FinancialEntry, value: any) => {
    const newData = [...data];
    newData[index] = { ...newData[index], [field]: value };
    setData(newData);
    
    // Validate the edited cell
    const rule = VALIDATION_RULES.find(rule => rule.field === field);
    const error = rule ? validateField(value, rule) : null;
    
    setErrors(prev => {
      const newErrors = { ...prev };
      if (!newErrors[index]) newErrors[index] = {};
      
      if (error) {
        newErrors[index][field] = error;
      } else {
        delete newErrors[index][field];
        if (Object.keys(newErrors[index]).length === 0) {
          delete newErrors[index];
        }
      }
      
      return newErrors;
    });
  };

  const handleValidateAndConfirm = async () => {
    if (!validateData(data)) {
      toast.error('Please fix all validation errors before proceeding');
      return;
    }

    setLoading(true);
    const toastId = toast.loading('Saving validated data...');

    try {
      await axios.post(`${API_URL}/validate/${fileId}`, {
        data,
        convertedFormats
      });

      toast.success('Data validated and saved successfully', { id: toastId });
      navigate('/export', { 
        state: { 
          fileId,
          data,
          convertedFormats
        }
      });
    } catch (error) {
      console.error('Validation error:', error);
      toast.error('Failed to save validated data', { id: toastId });
    } finally {
      setLoading(false);
    }
  };

  const handleBack = () => {
    navigate('/convert', { 
      state: { 
        fileId,
        data,
        convertedFormats
      }
    });
  };

  return (
    <div className="container mx-auto p-6 space-y-8">
      <div className="flex justify-between items-center">
        <button
          onClick={handleBack}
          className="flex items-center text-gray-600 hover:text-gray-900"
        >
          <ArrowLeftIcon className="h-5 w-5 mr-2" />
          Back
        </button>
        <h2 className="text-2xl font-semibold text-gray-900">Validate Data</h2>
        <button
          onClick={handleValidateAndConfirm}
          disabled={loading || Object.keys(errors).length > 0}
          className="flex items-center px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:bg-gray-400"
        >
          {loading ? (
            <ArrowPathIcon className="h-5 w-5 animate-spin" />
          ) : (
            <>
              <CheckCircleIcon className="h-5 w-5 mr-2" />
              Validate & Confirm
            </>
          )}
        </button>
      </div>

      <div className="bg-white rounded-lg shadow overflow-hidden">
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                {Object.keys(data[0] || {}).map((field) => (
                  <th
                    key={field}
                    className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider"
                  >
                    {field}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {data.map((entry, index) => (
                <tr key={index} className={errors[index] ? 'bg-red-50' : ''}>
                  {Object.entries(entry).map(([field, value]) => (
                    <td
                      key={field}
                      className={`px-6 py-4 whitespace-nowrap ${
                        errors[index]?.[field] ? 'relative' : ''
                      }`}
                    >
                      <input
                        type="text"
                        value={value as string}
                        onChange={(e) => handleCellEdit(index, field as keyof FinancialEntry, e.target.value)}
                        className={`w-full bg-transparent border-b border-transparent focus:border-blue-500 outline-none ${
                          errors[index]?.[field] ? 'text-red-600' : ''
                        }`}
                      />
                      {errors[index]?.[field] && (
                        <div className="absolute inset-y-0 right-0 pr-3 flex items-center pointer-events-none">
                          <ExclamationCircleIcon className="h-5 w-5 text-red-500" />
                        </div>
                      )}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {Object.keys(errors).length > 0 && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <h3 className="text-red-800 font-medium mb-2">Validation Errors</h3>
          <ul className="list-disc list-inside text-red-700">
            {Object.entries(errors).map(([rowIndex, rowErrors]) =>
              Object.entries(rowErrors).map(([field, error]) => (
                <li key={`${rowIndex}-${field}`}>
                  Row {Number(rowIndex) + 1}, {field}: {error}
                </li>
              ))
            )}
          </ul>
        </div>
      )}
    </div>
  );
};

export default ValidationComponent; 