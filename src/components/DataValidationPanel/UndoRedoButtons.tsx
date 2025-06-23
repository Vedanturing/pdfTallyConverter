import React from 'react';
import { ArrowUturnLeftIcon, ArrowUturnRightIcon } from '@heroicons/react/24/outline';

interface Props {
  canUndo: boolean;
  canRedo: boolean;
  onUndo: () => void;
  onRedo: () => void;
}

export default function UndoRedoButtons({ canUndo, canRedo, onUndo, onRedo }: Props) {
  return (
    <div className="space-x-2">
      <button
        onClick={onUndo}
        disabled={!canUndo}
        className={`p-2 rounded ${
          canUndo
            ? 'text-gray-700 hover:bg-gray-100'
            : 'text-gray-400 cursor-not-allowed'
        }`}
        title="Undo"
      >
        <ArrowUturnLeftIcon className="w-5 h-5" />
      </button>
      <button
        onClick={onRedo}
        disabled={!canRedo}
        className={`p-2 rounded ${
          canRedo
            ? 'text-gray-700 hover:bg-gray-100'
            : 'text-gray-400 cursor-not-allowed'
        }`}
        title="Redo"
      >
        <ArrowUturnRightIcon className="w-5 h-5" />
      </button>
    </div>
  );
} 