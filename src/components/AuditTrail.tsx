import React, { useState } from 'react';
import { format } from 'date-fns';
import {
  Clock,
  User,
  FileText,
  Pencil,
  Trash,
  CheckCircle,
  X
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import useAuditStore, { AuditEntry } from '../store/auditStore';

interface AuditTrailProps {
  entityId?: string;
  userId?: string;
  startDate?: Date;
  endDate?: Date;
  onClose?: () => void;
}

const AuditTrail: React.FC<AuditTrailProps> = ({
  entityId,
  userId,
  startDate,
  endDate,
  onClose,
}) => {
  const [filter, setFilter] = useState<'all' | 'changes' | 'validations'>('all');
  const auditStore = useAuditStore();

  let entries = entityId
    ? auditStore.getEntriesForEntity(entityId)
    : userId
    ? auditStore.getEntriesForUser(userId)
    : startDate && endDate
    ? auditStore.getEntriesByDateRange(startDate, endDate)
    : auditStore.entries;

  if (filter === 'changes') {
    entries = entries.filter((entry) => entry.action !== 'validate');
  } else if (filter === 'validations') {
    entries = entries.filter((entry) => entry.action === 'validate');
  }

  const getActionIcon = (action: AuditEntry['action']) => {
    switch (action) {
      case 'create':
        return <FileText className="h-5 w-5 text-green-500" />;
      case 'update':
        return <Pencil className="h-5 w-5 text-blue-500" />;
      case 'delete':
        return <Trash className="h-5 w-5 text-red-500" />;
      case 'validate':
        return <CheckCircle className="h-5 w-5 text-purple-500" />;
      case 'add_note':
        return <FileText className="h-5 w-5 text-yellow-500" />;
    }
  };

  const getActionText = (entry: AuditEntry) => {
    switch (entry.action) {
      case 'create':
        return `Created new ${entry.entityType}`;
      case 'update':
        return `Updated ${entry.entityType} ${entry.field ? `field "${entry.field}"` : ''}`;
      case 'delete':
        return `Deleted ${entry.entityType}`;
      case 'validate':
        return `Validated ${entry.entityType}`;
      case 'add_note':
        return `Added note to ${entry.entityType}`;
    }
  };

  const formatValue = (value: any) => {
    if (value === undefined || value === null) return 'N/A';
    if (typeof value === 'object') return JSON.stringify(value);
    return String(value);
  };

  return (
    <div className="bg-white rounded-lg shadow-xl max-w-4xl w-full max-h-[80vh] flex flex-col">
      <div className="p-6 border-b border-gray-200">
        <div className="flex justify-between items-center">
          <h2 className="text-2xl font-semibold text-gray-900">Audit Trail</h2>
          {onClose && (
            <button
              onClick={onClose}
              className="text-gray-400 hover:text-gray-500"
            >
              <span className="sr-only">Close</span>
              <X className="h-6 w-6" />
            </button>
          )}
        </div>

        <div className="mt-4 flex space-x-2">
          <button
            onClick={() => setFilter('all')}
            className={`px-4 py-2 rounded-lg ${
              filter === 'all'
                ? 'bg-blue-500 text-white'
                : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
            }`}
          >
            All
          </button>
          <button
            onClick={() => setFilter('changes')}
            className={`px-4 py-2 rounded-lg ${
              filter === 'changes'
                ? 'bg-blue-500 text-white'
                : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
            }`}
          >
            Changes
          </button>
          <button
            onClick={() => setFilter('validations')}
            className={`px-4 py-2 rounded-lg ${
              filter === 'validations'
                ? 'bg-blue-500 text-white'
                : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
            }`}
          >
            Validations
          </button>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto p-6">
        <div className="space-y-6">
          <AnimatePresence>
            {entries.map((entry) => (
              <motion.div
                key={entry.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                className="bg-gray-50 rounded-lg p-4 shadow"
              >
                <div className="flex items-start space-x-4">
                  <div className="flex-shrink-0">
                    {getActionIcon(entry.action)}
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center justify-between">
                      <p className="text-sm font-medium text-gray-900">
                        {getActionText(entry)}
                      </p>
                      <div className="flex items-center space-x-2 text-sm text-gray-500">
                        <Clock className="h-4 w-4" />
                        <span>{format(new Date(entry.timestamp), 'MMM d, yyyy HH:mm:ss')}</span>
                      </div>
                    </div>

                    <div className="mt-2">
                      <div className="flex items-center space-x-2 text-sm text-gray-500">
                        <User className="h-4 w-4" />
                        <span>User: {entry.userId}</span>
                      </div>

                      {entry.field && (
                        <div className="mt-2 text-sm">
                          <p className="text-gray-600">Field: {entry.field}</p>
                          {entry.oldValue !== undefined && (
                            <p className="text-red-600">
                              - {formatValue(entry.oldValue)}
                            </p>
                          )}
                          {entry.newValue !== undefined && (
                            <p className="text-green-600">
                              + {formatValue(entry.newValue)}
                            </p>
                          )}
                        </div>
                      )}

                      {entry.validationResult && (
                        <div className="mt-2 text-sm">
                          <div className={`flex items-center space-x-2 ${
                            entry.validationResult.isValid
                              ? 'text-green-600'
                              : 'text-red-600'
                          }`}>
                            <CheckCircle className="h-4 w-4" />
                            <span>
                              {entry.validationResult.isValid
                                ? 'Valid'
                                : `Invalid (${entry.validationResult.errorCount} errors, ${entry.validationResult.warningCount} warnings)`}
                            </span>
                          </div>
                        </div>
                      )}

                      {entry.note && (
                        <div className="mt-2 text-sm text-gray-600">
                          <p className="italic">{entry.note}</p>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              </motion.div>
            ))}
          </AnimatePresence>

          {entries.length === 0 && (
            <div className="text-center text-gray-500 py-8">
              No audit entries found
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default AuditTrail; 