import React, { useState, useEffect } from 'react';
import { useAuthStore } from '../../store/authStore';
import { Button } from '../ui/button';
import { 
  DocumentTextIcon, 
  ArrowDownTrayIcon, 
  TrashIcon,
  ClockIcon,
  EyeIcon,
  FolderOpenIcon
} from '@heroicons/react/24/outline';
import axios from 'axios';
import { API_URL } from '../../config';
import toast from 'react-hot-toast';
import { formatDistanceToNow } from 'date-fns';

interface ConversionHistoryItem {
  id: string;
  filename: string;
  original_filename: string;
  file_size: number;
  conversion_type: string;
  status: string;
  created_at: string;
  processing_time?: number;
}

interface HistoryListProps {
  onClose?: () => void;
}

export const HistoryList: React.FC<HistoryListProps> = ({ onClose }) => {
  const [history, setHistory] = useState<ConversionHistoryItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedItem, setSelectedItem] = useState<string | null>(null);
  const { isAuthenticated } = useAuthStore();

  useEffect(() => {
    if (isAuthenticated) {
      fetchHistory();
    }
  }, [isAuthenticated]);

  const fetchHistory = async () => {
    try {
      setLoading(true);
      const response = await axios.get(`${API_URL}/user/history`);
      setHistory(response.data);
    } catch (error: any) {
      const errorMessage = error.response?.data?.detail || 'Failed to fetch history';
      setError(errorMessage);
      toast.error(errorMessage);
    } finally {
      setLoading(false);
    }
  };

  const handleDeleteItem = async (itemId: string) => {
    if (!confirm('Are you sure you want to delete this item?')) {
      return;
    }

    try {
      await axios.delete(`${API_URL}/user/history/${itemId}`);
      setHistory(prev => prev.filter(item => item.id !== itemId));
      toast.success('Item deleted successfully');
    } catch (error: any) {
      const errorMessage = error.response?.data?.detail || 'Failed to delete item';
      toast.error(errorMessage);
    }
  };

  const handleClearAllHistory = async () => {
    if (!confirm('Are you sure you want to clear all history? This action cannot be undone.')) {
      return;
    }

    try {
      await axios.delete(`${API_URL}/user/history`);
      setHistory([]);
      toast.success('History cleared successfully');
    } catch (error: any) {
      const errorMessage = error.response?.data?.detail || 'Failed to clear history';
      toast.error(errorMessage);
    }
  };

  const handleViewDetails = async (itemId: string) => {
    try {
      const response = await axios.get(`${API_URL}/user/history/${itemId}`);
      setSelectedItem(itemId);
      // You can show details in a modal or expand the item
      console.log('Item details:', response.data);
    } catch (error: any) {
      const errorMessage = error.response?.data?.detail || 'Failed to fetch details';
      toast.error(errorMessage);
    }
  };

  const formatFileSize = (bytes: number) => {
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    if (bytes === 0) return '0 Bytes';
    const i = Math.floor(Math.log(bytes) / Math.log(1024));
    return Math.round(bytes / Math.pow(1024, i) * 100) / 100 + ' ' + sizes[i];
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'success':
        return 'text-green-600 bg-green-50 border-green-200';
      case 'failed':
        return 'text-red-600 bg-red-50 border-red-200';
      case 'processing':
        return 'text-yellow-600 bg-yellow-50 border-yellow-200';
      default:
        return 'text-gray-600 bg-gray-50 border-gray-200';
    }
  };

  if (!isAuthenticated) {
    return (
      <div className="text-center py-8">
        <FolderOpenIcon className="mx-auto h-12 w-12 text-gray-400" />
        <h3 className="mt-2 text-sm font-medium text-gray-900">Authentication Required</h3>
        <p className="mt-1 text-sm text-gray-500">
          Please sign in to view your conversion history.
        </p>
      </div>
    );
  }

  if (loading) {
    return (
      <div className="text-center py-8">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto"></div>
        <p className="mt-2 text-sm text-gray-500">Loading history...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="text-center py-8">
        <p className="text-red-600">{error}</p>
        <Button onClick={fetchHistory} className="mt-2">
          Retry
        </Button>
      </div>
    );
  }

  if (history.length === 0) {
    return (
      <div className="text-center py-8">
        <DocumentTextIcon className="mx-auto h-12 w-12 text-gray-400" />
        <h3 className="mt-2 text-sm font-medium text-gray-900">No History Yet</h3>
        <p className="mt-1 text-sm text-gray-500">
          Your conversion history will appear here once you start converting files.
        </p>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-lg shadow-lg p-6 max-w-4xl mx-auto">
      <div className="flex justify-between items-center mb-6">
        <h2 className="text-2xl font-bold text-gray-900">Conversion History</h2>
        <div className="flex space-x-2">
          {history.length > 0 && (
            <Button
              variant="outline"
              onClick={handleClearAllHistory}
              className="text-red-600 border-red-300 hover:bg-red-50"
            >
              <TrashIcon className="h-4 w-4 mr-2" />
              Clear All
            </Button>
          )}
          {onClose && (
            <Button variant="outline" onClick={onClose}>
              Close
            </Button>
          )}
        </div>
      </div>

      <div className="space-y-4">
        {history.map((item) => (
          <div
            key={item.id}
            className="border border-gray-200 rounded-lg p-4 hover:shadow-md transition-shadow"
          >
            <div className="flex items-center justify-between">
              <div className="flex-1">
                <div className="flex items-center space-x-3">
                  <DocumentTextIcon className="h-5 w-5 text-gray-400" />
                  <div>
                    <h3 className="text-sm font-medium text-gray-900">
                      {item.original_filename}
                    </h3>
                    <div className="flex items-center space-x-4 mt-1 text-xs text-gray-500">
                      <span>{formatFileSize(item.file_size)}</span>
                      <span className="capitalize">{item.conversion_type.replace('_', ' to ')}</span>
                      <span className="flex items-center">
                        <ClockIcon className="h-3 w-3 mr-1" />
                        {formatDistanceToNow(new Date(item.created_at), { addSuffix: true })}
                      </span>
                      {item.processing_time && (
                        <span>{item.processing_time.toFixed(2)}s</span>
                      )}
                    </div>
                  </div>
                </div>
              </div>

              <div className="flex items-center space-x-3">
                <span className={`px-2 py-1 text-xs rounded-full border ${getStatusColor(item.status)}`}>
                  {item.status}
                </span>
                
                <div className="flex space-x-2">
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => handleViewDetails(item.id)}
                  >
                    <EyeIcon className="h-4 w-4" />
                  </Button>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => handleDeleteItem(item.id)}
                    className="text-red-600 hover:text-red-700"
                  >
                    <TrashIcon className="h-4 w-4" />
                  </Button>
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>

      <div className="mt-6 text-center text-sm text-gray-500">
        Showing {history.length} conversion{history.length !== 1 ? 's' : ''}
      </div>
    </div>
  );
}; 