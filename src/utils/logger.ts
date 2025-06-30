const logger = {
  info: (message: string, ...args: any[]) => {
    console.log(`[INFO] ${message}`, ...args);
  },
  error: (message: string, ...args: any[]) => {
    console.error(`[ERROR] ${message}`, ...args);
  },
  warn: (message: string, ...args: any[]) => {
    console.warn(`[WARN] ${message}`, ...args);
  },
  debug: (message: string, ...args: any[]) => {
    console.debug(`[DEBUG] ${message}`, ...args);
  }
};

export const logAction = async (action: string, message: string, data?: any) => {
  // Simple logging for now - could be extended to send to backend
  console.log(`[ACTION] ${action.toUpperCase()}: ${message}`, data);
  
  // In a real application, you might want to send this to your backend
  // try {
  //   await axios.post('/api/audit-log', { action, message, data, timestamp: new Date().toISOString() });
  // } catch (error) {
  //   console.error('Failed to log action:', error);
  // }
};

export default logger; 