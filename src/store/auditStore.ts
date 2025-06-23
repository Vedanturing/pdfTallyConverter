import { create } from 'zustand';

export interface AuditEntry {
  id: string;
  timestamp: string;
  userId: string;
  action: 'create' | 'update' | 'delete' | 'validate' | 'add_note';
  entityType: 'cell' | 'row' | 'table';
  entityId: string;
  field?: string;
  oldValue?: string;
  newValue?: string;
  validationResult?: {
    isValid: boolean;
    errorCount: number;
    warningCount: number;
  };
  note?: string;
}

interface AuditStore {
  entries: AuditEntry[];
  addEntry: (entry: Omit<AuditEntry, 'id' | 'timestamp'>) => void;
  getEntriesForEntity: (entityId: string) => AuditEntry[];
  getEntriesForUser: (userId: string) => AuditEntry[];
  getEntriesByDateRange: (startDate: Date, endDate: Date) => AuditEntry[];
  clearEntries: () => void;
}

const useAuditStore = create<AuditStore>((set, get) => ({
  entries: [],

  addEntry: (entry) => set((state) => ({
    entries: [
      ...state.entries,
      {
        ...entry,
        id: crypto.randomUUID(),
        timestamp: new Date().toISOString(),
      },
    ],
  })),

  getEntriesForEntity: (entityId) => {
    return get().entries.filter((entry) => entry.entityId === entityId);
  },

  getEntriesForUser: (userId) => {
    return get().entries.filter((entry) => entry.userId === userId);
  },

  getEntriesByDateRange: (startDate, endDate) => {
    return get().entries.filter((entry) =>
      entry.timestamp >= startDate.toISOString() && entry.timestamp <= endDate.toISOString()
    );
  },

  clearEntries: () => set({ entries: [] }),
}));

export default useAuditStore; 