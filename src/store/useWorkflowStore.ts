import { create } from 'zustand';
import { StateCreator } from 'zustand';

export type WorkflowStep = 'upload' | 'preview' | 'convert' | 'validate';

interface WorkflowState {
  currentStep: WorkflowStep;
  fileId: string | null;
  setStep: (step: WorkflowStep) => void;
  setFileId: (id: string | null) => void;
  resetWorkflow: () => void;
}

const createWorkflowStore: StateCreator<WorkflowState> = (set) => ({
  currentStep: 'upload',
  fileId: null,
  setStep: (step: WorkflowStep) => set({ currentStep: step }),
  setFileId: (id: string | null) => set({ fileId: id }),
  resetWorkflow: () => set({ currentStep: 'upload', fileId: null }),
});

export const useWorkflowStore = create<WorkflowState>(createWorkflowStore); 