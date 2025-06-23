import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { API_URL } from '../config';
import {
  ArrowPathIcon,
  CheckCircleIcon,
  XCircleIcon,
  QuestionMarkCircleIcon,
} from '@heroicons/react/24/outline';
import toast from 'react-hot-toast';

interface Transaction {
  date: string;
  amount: number;
  description: string;
  transaction_type: 'bank' | 'invoice';
  status: 'pending' | 'matched' | 'unmatched';
  match_id?: string;
  confidence_score: number;
  source_id?: string;
}

interface Match {
  match_id: string;
  bank_transaction: Transaction;
  invoice_transaction: Transaction;
  confidence: number;
}

interface UnmatchedTransactions {
  bank: Transaction[];
  invoice: Transaction[];
}

const BankMatcher: React.FC = () => {
  const [matches, setMatches] = useState<Match[]>([]);
  const [unmatched, setUnmatched] = useState<UnmatchedTransactions>({ bank: [], invoice: [] });
  const [loading, setLoading] = useState(false);
  const [bankFile, setBankFile] = useState<File | null>(null);
  const [invoiceData, setInvoiceData] = useState<any[]>([]);

  const handleBankFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setBankFile(file);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post(`${API_URL}/bank-statement/upload`, formData);
      toast.success('Bank statement uploaded successfully');
    } catch (error) {
      console.error('Error uploading bank statement:', error);
      toast.error('Failed to upload bank statement');
    }
  };

  const handleInvoiceDataUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post(`${API_URL}/invoice-data/upload`, formData);
      setInvoiceData(response.data);
      toast.success('Invoice data uploaded successfully');
    } catch (error) {
      console.error('Error uploading invoice data:', error);
      toast.error('Failed to upload invoice data');
    }
  };

  const findMatches = async () => {
    if (!bankFile || invoiceData.length === 0) {
      toast.error('Please upload both bank statement and invoice data');
      return;
    }

    setLoading(true);
    try {
      const response = await axios.post(`${API_URL}/bank-matcher/match`);
      setMatches(response.data.matches);
      setUnmatched(response.data.unmatched);
      toast.success('Matching completed');
    } catch (error) {
      console.error('Error finding matches:', error);
      toast.error('Failed to find matches');
    } finally {
      setLoading(false);
    }
  };

  const updateMatchStatus = async (matchId: string, newStatus: 'pending' | 'matched' | 'unmatched') => {
    try {
      await axios.post(`${API_URL}/bank-matcher/update-status`, {
        match_id: matchId,
        status: newStatus
      });

      // Update local state
      setMatches(prevMatches => prevMatches.map(match => {
        if (match.match_id === matchId) {
          return {
            ...match,
            bank_transaction: { ...match.bank_transaction, status: newStatus },
            invoice_transaction: { ...match.invoice_transaction, status: newStatus }
          };
        }
        return match;
      }));

      toast.success('Match status updated');
    } catch (error) {
      console.error('Error updating match status:', error);
      toast.error('Failed to update match status');
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'matched':
        return <CheckCircleIcon className="h-5 w-5 text-green-500" />;
      case 'unmatched':
        return <XCircleIcon className="h-5 w-5 text-red-500" />;
      default:
        return <QuestionMarkCircleIcon className="h-5 w-5 text-yellow-500" />;
    }
  };

  const formatAmount = (amount: number) => {
    return new Intl.NumberFormat('en-IN', {
      style: 'currency',
      currency: 'INR'
    }).format(amount);
  };

  return (
    <div className="container mx-auto p-6 space-y-8">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="space-y-4">
          <h2 className="text-lg font-semibold">Upload Bank Statement</h2>
          <input
            type="file"
            accept=".pdf,.csv"
            onChange={handleBankFileUpload}
            className="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
          />
        </div>
        <div className="space-y-4">
          <h2 className="text-lg font-semibold">Upload Invoice Data</h2>
          <input
            type="file"
            accept=".pdf,.csv"
            onChange={handleInvoiceDataUpload}
            className="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
          />
        </div>
      </div>

      <div className="flex justify-center">
        <button
          onClick={findMatches}
          disabled={loading || !bankFile || invoiceData.length === 0}
          className="flex items-center px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-400"
        >
          {loading ? (
            <ArrowPathIcon className="h-5 w-5 mr-2 animate-spin" />
          ) : (
            'Find Matches'
          )}
        </button>
      </div>

      {/* Matches */}
      <div className="space-y-6">
        <h2 className="text-xl font-semibold">Matches</h2>
        <div className="grid gap-4">
          {matches.map(match => (
            <div
              key={match.match_id}
              className="bg-white p-4 rounded-lg shadow border border-gray-200 space-y-4"
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-2">
                  {getStatusIcon(match.bank_transaction.status)}
                  <span className="font-medium">Match ID: {match.match_id}</span>
                </div>
                <span className="text-sm text-gray-500">
                  Confidence: {(match.confidence * 100).toFixed(1)}%
                </span>
              </div>

              <div className="grid md:grid-cols-2 gap-4">
                <div className="space-y-2">
                  <h3 className="font-medium">Bank Transaction</h3>
                  <p>Date: {new Date(match.bank_transaction.date).toLocaleDateString()}</p>
                  <p>Amount: {formatAmount(match.bank_transaction.amount)}</p>
                  <p className="text-sm text-gray-600">{match.bank_transaction.description}</p>
                </div>

                <div className="space-y-2">
                  <h3 className="font-medium">Invoice Transaction</h3>
                  <p>Date: {new Date(match.invoice_transaction.date).toLocaleDateString()}</p>
                  <p>Amount: {formatAmount(match.invoice_transaction.amount)}</p>
                  <p className="text-sm text-gray-600">{match.invoice_transaction.description}</p>
                </div>
              </div>

              <div className="flex justify-end space-x-2">
                <button
                  onClick={() => updateMatchStatus(match.match_id, 'matched')}
                  className="px-3 py-1 bg-green-100 text-green-700 rounded hover:bg-green-200"
                >
                  Confirm Match
                </button>
                <button
                  onClick={() => updateMatchStatus(match.match_id, 'unmatched')}
                  className="px-3 py-1 bg-red-100 text-red-700 rounded hover:bg-red-200"
                >
                  Reject Match
                </button>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Unmatched Transactions */}
      <div className="grid md:grid-cols-2 gap-6">
        <div className="space-y-4">
          <h2 className="text-xl font-semibold">Unmatched Bank Transactions</h2>
          <div className="space-y-2">
            {unmatched.bank.map(tx => (
              <div
                key={tx.source_id}
                className="bg-white p-3 rounded shadow-sm border border-gray-200"
              >
                <p>Date: {new Date(tx.date).toLocaleDateString()}</p>
                <p>Amount: {formatAmount(tx.amount)}</p>
                <p className="text-sm text-gray-600">{tx.description}</p>
              </div>
            ))}
          </div>
        </div>

        <div className="space-y-4">
          <h2 className="text-xl font-semibold">Unmatched Invoice Transactions</h2>
          <div className="space-y-2">
            {unmatched.invoice.map(tx => (
              <div
                key={tx.source_id}
                className="bg-white p-3 rounded shadow-sm border border-gray-200"
              >
                <p>Date: {new Date(tx.date).toLocaleDateString()}</p>
                <p>Amount: {formatAmount(tx.amount)}</p>
                <p className="text-sm text-gray-600">{tx.description}</p>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

export default BankMatcher; 