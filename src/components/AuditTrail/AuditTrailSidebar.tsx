import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { format } from 'date-fns';
import { ChevronRightIcon, ChevronDownIcon, XMarkIcon } from '@heroicons/react/24/outline';
import { API_URL } from '../../config';

interface AuditLog {
  timestamp: string;
  action_type: string;
  summary: string;
  user_id: string;
  metadata: Record<string, any>;
}

interface AuditTrailSidebarProps {
  isOpen: boolean;
  onClose: () => void;
}

const AuditTrailSidebar: React.FC<AuditTrailSidebarProps> = ({ isOpen, onClose }) => {
  const [logs, setLogs] = useState<AuditLog[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [expandedLogs, setExpandedLogs] = useState<Set<string>>(new Set());

  useEffect(() => {
    if (isOpen) {
      fetchLogs();
    }
  }, [isOpen]);

  const fetchLogs = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await axios.get(`${API_URL}/audit-logs`);
      setLogs(response.data);
    } catch (err: any) {
      setError(err.message || 'Failed to fetch audit logs');
    } finally {
      setLoading(false);
    }
  };

  const toggleLogExpansion = (timestamp: string) => {
    const newExpandedLogs = new Set(expandedLogs);
    if (expandedLogs.has(timestamp)) {
      newExpandedLogs.delete(timestamp);
    } else {
      newExpandedLogs.add(timestamp);
    }
    setExpandedLogs(newExpandedLogs);
  };

  const formatTimestamp = (timestamp: string) => {
    return format(new Date(timestamp), 'MMM d, yyyy HH:mm:ss');
  };

  const getActionTypeColor = (actionType: string) => {
    const colors: Record<string, string> = {
      upload: 'bg-blue-100 text-blue-800',
      convert: 'bg-green-100 text-green-800',
      validate: 'bg-yellow-100 text-yellow-800',
      export: 'bg-purple-100 text-purple-800',
      api_request: 'bg-gray-100 text-gray-800',
      default: 'bg-gray-100 text-gray-800'
    };
    return colors[actionType] || colors.default;
  };

  if (!isOpen) return null;

  return (
    <div className="fixed right-0 top-0 h-full w-96 bg-white shadow-lg z-50 overflow-hidden flex flex-col">
      {/* Header */}
      <div className="p-4 border-b flex justify-between items-center bg-gray-50">
        <h2 className="text-lg font-semibold">Audit Trail</h2>
        <button
          onClick={onClose}
          className="p-1 hover:bg-gray-200 rounded-full"
        >
          <XMarkIcon className="h-5 w-5" />
        </button>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {loading && (
          <div className="flex justify-center items-center h-32">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500" />
          </div>
        )}

        {error && (
          <div className="bg-red-100 text-red-700 p-3 rounded">
            {error}
          </div>
        )}

        {logs.map((log) => (
          <div key={log.timestamp} className="border rounded-lg overflow-hidden">
            <div
              className="flex items-center p-3 cursor-pointer hover:bg-gray-50"
              onClick={() => toggleLogExpansion(log.timestamp)}
            >
              {expandedLogs.has(log.timestamp) ? (
                <ChevronDownIcon className="h-5 w-5 mr-2" />
              ) : (
                <ChevronRightIcon className="h-5 w-5 mr-2" />
              )}
              
              <div className="flex-1">
                <div className="flex items-center gap-2">
                  <span className={`text-xs px-2 py-1 rounded ${getActionTypeColor(log.action_type)}`}>
                    {log.action_type}
                  </span>
                  <span className="text-sm text-gray-500">
                    {formatTimestamp(log.timestamp)}
                  </span>
                </div>
                <p className="text-sm mt-1">{log.summary}</p>
              </div>
            </div>

            {expandedLogs.has(log.timestamp) && (
              <div className="p-3 bg-gray-50 border-t">
                <div className="text-sm">
                  <p><span className="font-medium">User:</span> {log.user_id}</p>
                  {Object.entries(log.metadata).map(([key, value]) => (
                    <p key={key}>
                      <span className="font-medium">{key}:</span>{' '}
                      {typeof value === 'object' ? JSON.stringify(value) : String(value)}
                    </p>
                  ))}
                </div>
              </div>
            )}
          </div>
        ))}
      </div>

      {/* Footer */}
      <div className="p-4 border-t bg-gray-50">
        <button
          onClick={fetchLogs}
          className="w-full py-2 px-4 bg-blue-600 text-white rounded hover:bg-blue-700"
        >
          Refresh Logs
        </button>
      </div>
    </div>
  );
};

export default AuditTrailSidebar; 