import React, { useState } from 'react';
import { useAuthStore } from '../../store/authStore';
import { Button } from '../ui/button';
import { Input } from '../ui/input';
import { XMarkIcon, ExclamationTriangleIcon } from '@heroicons/react/24/outline';
import toast from 'react-hot-toast';

interface DeleteAccountModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSuccess?: () => void;
}

export const DeleteAccountModal: React.FC<DeleteAccountModalProps> = ({ 
  isOpen, 
  onClose, 
  onSuccess 
}) => {
  const [confirmText, setConfirmText] = useState('');
  const [isDeleting, setIsDeleting] = useState(false);
  const { deleteAccount, user } = useAuthStore();

  const handleDelete = async () => {
    if (confirmText !== 'DELETE') {
      toast.error('Please type DELETE to confirm account deletion');
      return;
    }

    setIsDeleting(true);
    
    try {
      const success = await deleteAccount();
      if (success) {
        toast.success('Account deleted successfully');
        onSuccess?.();
        onClose();
      }
    } catch (error) {
      // Error is handled in the store
    } finally {
      setIsDeleting(false);
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      {/* Backdrop */}
      <div 
        className="absolute inset-0 bg-black bg-opacity-50 transition-opacity"
        onClick={onClose}
      />
      
      {/* Modal */}
      <div className="relative bg-white rounded-lg shadow-xl p-6 w-full max-w-md mx-4">
        {/* Close button */}
        <button
          onClick={onClose}
          className="absolute top-4 right-4 text-gray-400 hover:text-gray-600"
          disabled={isDeleting}
        >
          <XMarkIcon className="h-5 w-5" />
        </button>

        <div className="flex items-center mb-4">
          <ExclamationTriangleIcon className="h-8 w-8 text-red-600 mr-3" />
          <h3 className="text-lg font-semibold text-gray-900">Delete Account</h3>
        </div>

        <div className="mb-6">
          <p className="text-gray-700 mb-4">
            Are you sure you want to delete your account? This action will:
          </p>
          <ul className="list-disc list-inside text-sm text-gray-600 space-y-1 mb-4">
            <li>Permanently delete your account and profile</li>
            <li>Remove all your conversion history</li>
            <li>Cannot be undone</li>
          </ul>
          <p className="text-gray-700 mb-4">
            <strong>Account:</strong> {user?.email}
          </p>
        </div>

        <div className="mb-6">
          <label htmlFor="confirm-delete" className="block text-sm font-medium text-gray-700 mb-2">
            Type <strong>DELETE</strong> to confirm:
          </label>
          <Input
            id="confirm-delete"
            type="text"
            value={confirmText}
            onChange={(e) => setConfirmText(e.target.value)}
            placeholder="Type DELETE to confirm"
            className="w-full"
            disabled={isDeleting}
          />
        </div>

        <div className="flex justify-end space-x-3">
          <Button
            variant="outline"
            onClick={onClose}
            disabled={isDeleting}
          >
            Cancel
          </Button>
          <Button
            onClick={handleDelete}
            disabled={isDeleting || confirmText !== 'DELETE'}
            className="bg-red-600 hover:bg-red-700 text-white"
          >
            {isDeleting ? 'Deleting...' : 'Delete Account'}
          </Button>
        </div>
      </div>
    </div>
  );
}; 