import React, { useState } from 'react';
import axios from 'axios';
import { API_URL } from '../config';
import {
  DocumentArrowDownIcon,
  DocumentCheckIcon,
  ArrowPathIcon,
  ExclamationCircleIcon,
} from '@heroicons/react/24/outline';
import toast from 'react-hot-toast';

interface GSTInvoice {
  invoice_no: string;
  invoice_date: string;
  customer_gstin: string;
  customer_name: string;
  place_of_supply: string;
  taxable_value: number;
  gst_rate: number;
  invoice_type: 'B2B' | 'B2C' | 'EXPORT';
  reverse_charge: boolean;
  export_type?: string;
  shipping_bill_no?: string;
  shipping_bill_date?: string;
  port_code?: string;
}

const GSTHelper: React.FC = () => {
  const [loading, setLoading] = useState(false);
  const [period, setPeriod] = useState(getCurrentPeriod());
  const [invoice, setInvoice] = useState<GSTInvoice>({
    invoice_no: '',
    invoice_date: new Date().toISOString().split('T')[0],
    customer_gstin: '',
    customer_name: '',
    place_of_supply: '',
    taxable_value: 0,
    gst_rate: 18,
    invoice_type: 'B2B',
    reverse_charge: false
  });

  function getCurrentPeriod() {
    const date = new Date();
    return `${date.getFullYear()}${(date.getMonth() + 1).toString().padStart(2, '0')}`;
  }

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value, type } = e.target;
    setInvoice(prev => ({
      ...prev,
      [name]: type === 'number' ? parseFloat(value) : 
              type === 'checkbox' ? (e.target as HTMLInputElement).checked : 
              value
    }));
  };

  const validateGSTIN = async () => {
    try {
      const response = await axios.post(`${API_URL}/gst/validate-gstin`, {
        gstin: invoice.customer_gstin
      });
      
      if (response.data.is_valid) {
        toast.success('GSTIN is valid');
      } else {
        toast.error('Invalid GSTIN format');
      }
    } catch (error) {
      console.error('Error validating GSTIN:', error);
      toast.error('Failed to validate GSTIN');
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);

    try {
      const response = await axios.post(`${API_URL}/gst/add-invoice`, invoice);
      toast.success('Invoice added successfully');
      
      // Clear form
      setInvoice({
        invoice_no: '',
        invoice_date: new Date().toISOString().split('T')[0],
        customer_gstin: '',
        customer_name: '',
        place_of_supply: '',
        taxable_value: 0,
        gst_rate: 18,
        invoice_type: 'B2B',
        reverse_charge: false
      });
    } catch (error) {
      console.error('Error adding invoice:', error);
      toast.error('Failed to add invoice');
    } finally {
      setLoading(false);
    }
  };

  const generateReport = async (type: 'gstr1' | 'gstr3b' | 'excel') => {
    setLoading(true);
    try {
      const response = await axios.post(
        `${API_URL}/gst/generate-${type}`,
        { period },
        { responseType: 'blob' }
      );

      // Create download link
      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', `${type.toUpperCase()}_${period}.${type === 'excel' ? 'xlsx' : 'json'}`);
      document.body.appendChild(link);
      link.click();
      link.remove();

      toast.success(`${type.toUpperCase()} report generated successfully`);
    } catch (error) {
      console.error(`Error generating ${type}:`, error);
      toast.error(`Failed to generate ${type} report`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container mx-auto p-6 space-y-8">
      <div className="bg-white p-6 rounded-lg shadow space-y-6">
        <h2 className="text-2xl font-semibold">GST Document Assistant</h2>

        {/* Invoice Form */}
        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700">Invoice No</label>
              <input
                type="text"
                name="invoice_no"
                value={invoice.invoice_no}
                onChange={handleInputChange}
                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
                required
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700">Invoice Date</label>
              <input
                type="date"
                name="invoice_date"
                value={invoice.invoice_date}
                onChange={handleInputChange}
                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
                required
              />
            </div>

            <div className="relative">
              <label className="block text-sm font-medium text-gray-700">Customer GSTIN</label>
              <div className="mt-1 flex rounded-md shadow-sm">
                <input
                  type="text"
                  name="customer_gstin"
                  value={invoice.customer_gstin}
                  onChange={handleInputChange}
                  className="block w-full rounded-l-md border-gray-300 focus:border-blue-500 focus:ring-blue-500"
                  required
                />
                <button
                  type="button"
                  onClick={validateGSTIN}
                  className="inline-flex items-center px-3 rounded-r-md border border-l-0 border-gray-300 bg-gray-50 text-gray-500 hover:bg-gray-100"
                >
                  <DocumentCheckIcon className="h-5 w-5" />
                </button>
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700">Customer Name</label>
              <input
                type="text"
                name="customer_name"
                value={invoice.customer_name}
                onChange={handleInputChange}
                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
                required
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700">Place of Supply</label>
              <input
                type="text"
                name="place_of_supply"
                value={invoice.place_of_supply}
                onChange={handleInputChange}
                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
                required
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700">Taxable Value</label>
              <input
                type="number"
                name="taxable_value"
                value={invoice.taxable_value}
                onChange={handleInputChange}
                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
                required
                min="0"
                step="0.01"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700">GST Rate (%)</label>
              <select
                name="gst_rate"
                value={invoice.gst_rate}
                onChange={handleInputChange}
                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
                required
              >
                <option value="0">0%</option>
                <option value="5">5%</option>
                <option value="12">12%</option>
                <option value="18">18%</option>
                <option value="28">28%</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700">Invoice Type</label>
              <select
                name="invoice_type"
                value={invoice.invoice_type}
                onChange={handleInputChange}
                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
                required
              >
                <option value="B2B">B2B</option>
                <option value="B2C">B2C</option>
                <option value="EXPORT">Export</option>
              </select>
            </div>

            <div className="flex items-center">
              <input
                type="checkbox"
                name="reverse_charge"
                checked={invoice.reverse_charge}
                onChange={handleInputChange}
                className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
              />
              <label className="ml-2 block text-sm text-gray-900">
                Reverse Charge Applicable
              </label>
            </div>
          </div>

          {invoice.invoice_type === 'EXPORT' && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4">
              <div>
                <label className="block text-sm font-medium text-gray-700">Export Type</label>
                <select
                  name="export_type"
                  value={invoice.export_type}
                  onChange={handleInputChange}
                  className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
                >
                  <option value="With Payment">With Payment</option>
                  <option value="Without Payment">Without Payment</option>
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700">Shipping Bill No</label>
                <input
                  type="text"
                  name="shipping_bill_no"
                  value={invoice.shipping_bill_no}
                  onChange={handleInputChange}
                  className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700">Shipping Bill Date</label>
                <input
                  type="date"
                  name="shipping_bill_date"
                  value={invoice.shipping_bill_date}
                  onChange={handleInputChange}
                  className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700">Port Code</label>
                <input
                  type="text"
                  name="port_code"
                  value={invoice.port_code}
                  onChange={handleInputChange}
                  className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
                />
              </div>
            </div>
          )}

          <div className="flex justify-end">
            <button
              type="submit"
              disabled={loading}
              className="inline-flex items-center px-4 py-2 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
            >
              {loading ? (
                <ArrowPathIcon className="h-5 w-5 mr-2 animate-spin" />
              ) : (
                'Add Invoice'
              )}
            </button>
          </div>
        </form>
      </div>

      {/* Report Generation */}
      <div className="bg-white p-6 rounded-lg shadow space-y-6">
        <h3 className="text-xl font-semibold">Generate Reports</h3>
        
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700">Tax Period</label>
            <input
              type="text"
              value={period}
              onChange={(e) => setPeriod(e.target.value)}
              placeholder="YYYYMM"
              pattern="\d{6}"
              className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
              required
            />
          </div>

          <div className="flex flex-wrap gap-4">
            <button
              onClick={() => generateReport('gstr1')}
              disabled={loading}
              className="inline-flex items-center px-4 py-2 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-green-600 hover:bg-green-700"
            >
              <DocumentArrowDownIcon className="h-5 w-5 mr-2" />
              Generate GSTR-1
            </button>

            <button
              onClick={() => generateReport('gstr3b')}
              disabled={loading}
              className="inline-flex items-center px-4 py-2 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-green-600 hover:bg-green-700"
            >
              <DocumentArrowDownIcon className="h-5 w-5 mr-2" />
              Generate GSTR-3B
            </button>

            <button
              onClick={() => generateReport('excel')}
              disabled={loading}
              className="inline-flex items-center px-4 py-2 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-green-600 hover:bg-green-700"
            >
              <DocumentArrowDownIcon className="h-5 w-5 mr-2" />
              Generate Excel Report
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default GSTHelper; 