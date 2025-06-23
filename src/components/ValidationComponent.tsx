import React from 'react';
import { useLocation } from 'react-router-dom';
import { ValidationModule } from './ValidationModule';

export default function ValidationComponent() {
  const location = useLocation();
  const fileId = location.state?.fileId;

  if (!fileId) {
    return (
      <div className="flex items-center justify-center h-full">
        <p className="text-gray-500">No file selected for validation</p>
      </div>
    );
  }

  return (
    <div className="container mx-auto py-6">
      <h1 className="text-2xl font-bold mb-6">Data Validation</h1>
      <ValidationModule fileId={fileId} />
    </div>
  );
} 