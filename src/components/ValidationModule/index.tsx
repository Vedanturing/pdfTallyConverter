import React, { useEffect } from 'react';
import { ValidationProvider, useValidation } from './ValidationContext';
import { ValidationGrid } from './ValidationGrid';
import { ValidationToolbar } from './ValidationToolbar';
import { FinancialEntry } from '../../types/financial';
import { API_URL } from '../../config';
import axios from 'axios';
import toast from 'react-hot-toast';

interface ValidationModuleProps {
  fileId: string;
  initialData?: FinancialEntry[];
  validationResults?: any;
}

function ValidationModuleContent({ fileId, initialData, validationResults }: ValidationModuleProps) {
  const { state, actions } = useValidation();

  useEffect(() => {
    if (initialData) {
      // Load initial data directly instead of fetching
      loadInitialData();
    } else {
      fetchValidationData();
    }
  }, [fileId, initialData]);

  const loadInitialData = () => {
    try {
      if (initialData && initialData.length > 0) {
        // Convert initial data to the format expected by ValidationGrid
        const convertedData = initialData.map((entry, index) => ({
          ...entry,
          id: entry.id || `row-${index}`,
          rowIndex: index
        }));

        actions.updateValidationState({
          data: convertedData,
          issues: {},
          summary: {
            totalIssues: 0,
            errors: 0,
            warnings: 0,
            ignoredIssues: 0,
            fixedIssues: 0,
            byType: {},
            bySeverity: {}
          }
        });

        // Process validation results if available
        if (validationResults?.results) {
          const issues: Record<string, any> = {};
          validationResults.results.forEach((result: any) => {
            result.issues.forEach((issue: any) => {
              const cellId = `${result.row}-${issue.field}`;
              if (!issues[cellId]) {
                issues[cellId] = [];
              }
              issues[cellId].push({
                ...issue,
                cellId,
                rowIndex: result.row,
                columnKey: issue.field
              });
            });
          });

          actions.updateValidationState({
            data: convertedData,
            issues,
            summary: validationResults.summary || state.summary
          });
        }

        toast.success('Data loaded successfully');
      }
    } catch (error) {
      console.error('Error loading initial data:', error);
      toast.error('Failed to load validation data');
    }
  };

  const fetchValidationData = async () => {
    try {
      // Fetch validation results
      const [validationRes, excelRes] = await Promise.all([
        axios.get(`${API_URL}/validate/${fileId}`),
        axios.get(`${API_URL}/excel-data/${fileId}`)
      ]);

      if (validationRes.data?.validationResults?.results) {
        const issues: Record<string, any> = {};
        validationRes.data.validationResults.results.forEach((result: any) => {
          result.issues.forEach((issue: any) => {
            const cellId = `${result.row}-${issue.field}`;
            if (!issues[cellId]) {
              issues[cellId] = [];
            }
            issues[cellId].push({
              ...issue,
              cellId,
              rowIndex: result.row,
              columnKey: issue.field
            });
          });
        });

        // Update state with validation data
        actions.updateValidationState({
          data: excelRes.data,
          issues,
          summary: validationRes.data.validationResults.summary
        });
      }
    } catch (error) {
      console.error('Error fetching validation data:', error);
      toast.error('Failed to load validation data');
    }
  };

  const handleCellEdit = async (rowIndex: number, field: keyof FinancialEntry, value: any) => {
    try {
      await axios.post(`${API_URL}/update-cell/${fileId}`, {
        rowIndex,
        field,
        value
      });

      // Refresh validation data after edit
      await fetchValidationData();
      toast.success('Cell updated successfully');
    } catch (error) {
      console.error('Error updating cell:', error);
      toast.error('Failed to update cell');
    }
  };

  const handleApplyAutoFixes = async () => {
    try {
      await axios.post(`${API_URL}/auto-fix/${fileId}`);
      await fetchValidationData();
      toast.success('Auto-fixes applied successfully');
    } catch (error) {
      console.error('Error applying auto-fixes:', error);
      toast.error('Failed to apply auto-fixes');
    }
  };

  const handleSearch = (query: string) => {
    if (!query.trim()) {
      return;
    }

    try {
      // Simple query parser for basic operations
      const searchResults = state.data.filter(row => {
        const parts = query.split(/\s+(and|or)\s+/);
        return parts.some(part => {
          const [field, operator, value] = part.split(/\s+/);
          if (!field || !operator || !value) return false;

          const fieldValue = row[field as keyof FinancialEntry];
          switch (operator) {
            case '>':
              return Number(fieldValue) > Number(value);
            case '<':
              return Number(fieldValue) < Number(value);
            case '=':
            case '==':
              return fieldValue === value;
            case 'contains':
              return String(fieldValue).toLowerCase().includes(value.toLowerCase());
            default:
              return false;
          }
        });
      });

      // Update UI with search results
      actions.updateSearchResults(searchResults);
    } catch (error) {
      console.error('Error in search:', error);
      toast.error('Invalid search query');
    }
  };

  return (
    <div className="space-y-6">
      <ValidationToolbar
        summary={state.summary}
        onApplyAutoFixes={handleApplyAutoFixes}
        onAddRule={actions.addCustomRule}
        onSearch={handleSearch}
      />

      <ValidationGrid
        data={state.data}
        issues={state.issues}
        onCellEdit={handleCellEdit}
        onIgnoreIssue={actions.ignoreIssue}
        editableRows={state.editableRows}
        onToggleEditable={actions.toggleRowEditable}
      />
    </div>
  );
}

export function ValidationModule(props: ValidationModuleProps) {
  return (
    <ValidationProvider>
      <ValidationModuleContent {...props} />
    </ValidationProvider>
  );
} 