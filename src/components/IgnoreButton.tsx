import React from 'react';
import { EyeSlashIcon } from '@heroicons/react/24/outline';

interface IgnoreButtonProps {
  onIgnore: () => void;
  size?: 'sm' | 'md' | 'lg';
  variant?: 'button' | 'icon';
  disabled?: boolean;
  className?: string;
}

const IgnoreButton: React.FC<IgnoreButtonProps> = ({
  onIgnore,
  size = 'sm',
  variant = 'button',
  disabled = false,
  className = ''
}) => {
  const sizeClasses = {
    sm: 'px-2 py-1 text-xs',
    md: 'px-3 py-2 text-sm',
    lg: 'px-4 py-2 text-base'
  };

  const iconSizes = {
    sm: 'h-3 w-3',
    md: 'h-4 w-4',
    lg: 'h-5 w-5'
  };

  const baseClasses = `
    inline-flex items-center font-medium rounded transition-colors
    focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2
    ${disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}
  `;

  const buttonClasses = `
    ${baseClasses}
    border border-gray-300 text-gray-700 bg-white hover:bg-gray-50
    ${sizeClasses[size]}
    ${className}
  `;

  const iconClasses = `
    ${baseClasses}
    text-gray-500 hover:text-gray-700 p-1
    ${className}
  `;

  const handleClick = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (!disabled) {
      onIgnore();
    }
  };

  if (variant === 'icon') {
    return (
      <button
        onClick={handleClick}
        disabled={disabled}
        className={iconClasses}
        title="Ignore this issue"
      >
        <EyeSlashIcon className={iconSizes[size]} />
      </button>
    );
  }

  return (
    <button
      onClick={handleClick}
      disabled={disabled}
      className={buttonClasses}
      title="Ignore this issue"
    >
      <EyeSlashIcon className={`${iconSizes[size]} mr-1`} />
      Ignore
    </button>
  );
};

export default IgnoreButton; 