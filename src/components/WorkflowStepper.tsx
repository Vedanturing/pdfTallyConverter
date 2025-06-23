import React from 'react';
import { type LucideIcon } from 'lucide-react';

interface Step {
  title: string;
  icon: React.ForwardRefExoticComponent<any>;
}

interface WorkflowStepperProps {
  steps: Step[];
  currentStep: number;
}

const WorkflowStepper: React.FC<WorkflowStepperProps> = ({ steps, currentStep }) => {
  return (
    <nav aria-label="Progress">
      <ol role="list" className="flex items-center">
        {steps.map((step, index) => {
          const Icon = step.icon;
          return (
            <li
              key={step.title}
              className={`relative ${index !== steps.length - 1 ? 'pr-8 sm:pr-20' : ''}`}
            >
              <div className="flex items-center">
                <div
                  className={`${
                    index <= currentStep
                      ? 'bg-blue-600 text-white'
                      : 'bg-gray-200 text-gray-500'
                  } h-10 w-10 rounded-full flex items-center justify-center transition-colors`}
                >
                  <Icon className="h-6 w-6" />
                </div>
                {index !== steps.length - 1 && (
                  <div
                    className={`${
                      index < currentStep ? 'bg-blue-600' : 'bg-gray-200'
                    } h-0.5 absolute top-5 left-10 -right-4 sm:right-0 transition-colors`}
                  />
                )}
              </div>
              <div className="mt-2">
                <span
                  className={`text-sm font-medium ${
                    index <= currentStep ? 'text-blue-600' : 'text-gray-500'
                  }`}
                >
                  {step.title}
                </span>
              </div>
            </li>
          );
        })}
      </ol>
    </nav>
  );
};

export default WorkflowStepper; 