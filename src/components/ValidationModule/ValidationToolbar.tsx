import React, { useState } from 'react';
import { ValidationToolbarProps, ValidationRule } from './types';
import { Button } from '../ui/button';
import { Input } from '../ui/input';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '../ui/select';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '../ui/dialog';
import {
  MagnifyingGlassIcon,
  PlusIcon,
  WrenchIcon
} from '@heroicons/react/24/outline';

export function ValidationToolbar({
  summary,
  onApplyAutoFixes,
  onAddRule,
  onSearch
}: ValidationToolbarProps) {
  const [searchQuery, setSearchQuery] = useState('');
  const [showRuleBuilder, setShowRuleBuilder] = useState(false);
  const [newRule, setNewRule] = useState<Partial<ValidationRule>>({
    name: '',
    field: 'amount',
    operator: 'equals',
    value: '',
    severity: 'warning'
  });

  const handleSearch = () => {
    onSearch(searchQuery);
  };

  const handleAddRule = () => {
    if (newRule.name && newRule.field && newRule.operator && newRule.value !== undefined) {
      onAddRule(newRule as Omit<ValidationRule, 'id'>);
      setShowRuleBuilder(false);
      setNewRule({
        name: '',
        field: 'amount',
        operator: 'equals',
        value: '',
        severity: 'warning'
      });
    }
  };

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between bg-white p-4 rounded-lg shadow">
        <div className="flex space-x-4">
          <div className="text-center">
            <div className="text-2xl font-bold">{summary.total}</div>
            <div className="text-sm text-gray-500">Total Issues</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-red-600">{summary.errors}</div>
            <div className="text-sm text-gray-500">Errors</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-yellow-600">{summary.warnings}</div>
            <div className="text-sm text-gray-500">Warnings</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-green-600">{summary.fixed}</div>
            <div className="text-sm text-gray-500">Fixed</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-blue-600">{summary.ignored}</div>
            <div className="text-sm text-gray-500">Ignored</div>
          </div>
        </div>

        <div className="flex items-center space-x-2">
          <Button onClick={onApplyAutoFixes} className="flex items-center">
            <WrenchIcon className="h-5 w-5 mr-2" />
            Apply Auto-Fixes
          </Button>
          <Dialog open={showRuleBuilder} onOpenChange={setShowRuleBuilder}>
            <DialogTrigger asChild>
              <Button variant="outline" className="flex items-center">
                <PlusIcon className="h-5 w-5 mr-2" />
                Add Rule
              </Button>
            </DialogTrigger>
            <DialogContent>
              <DialogHeader>
                <DialogTitle>Add Custom Validation Rule</DialogTitle>
              </DialogHeader>
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium mb-1">Rule Name</label>
                  <Input
                    value={newRule.name}
                    onChange={(e) => setNewRule({ ...newRule, name: e.target.value })}
                    placeholder="e.g., High Value Transaction"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium mb-1">Field</label>
                  <Select
                    value={newRule.field}
                    onValueChange={(value: any) => setNewRule({ ...newRule, field: value })}
                  >
                    <SelectTrigger>
                      <SelectValue placeholder="Select field" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="amount">Amount</SelectItem>
                      <SelectItem value="date">Date</SelectItem>
                      <SelectItem value="description">Description</SelectItem>
                      <SelectItem value="balance">Balance</SelectItem>
                      <SelectItem value="gstin">GSTIN</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div>
                  <label className="block text-sm font-medium mb-1">Operator</label>
                  <Select
                    value={newRule.operator}
                    onValueChange={(value: any) => setNewRule({ ...newRule, operator: value })}
                  >
                    <SelectTrigger>
                      <SelectValue placeholder="Select operator" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="equals">Equals</SelectItem>
                      <SelectItem value="notEquals">Not Equals</SelectItem>
                      <SelectItem value="greaterThan">Greater Than</SelectItem>
                      <SelectItem value="lessThan">Less Than</SelectItem>
                      <SelectItem value="contains">Contains</SelectItem>
                      <SelectItem value="before">Before</SelectItem>
                      <SelectItem value="after">After</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div>
                  <label className="block text-sm font-medium mb-1">Value</label>
                  <Input
                    value={newRule.value}
                    onChange={(e) => setNewRule({ ...newRule, value: e.target.value })}
                    placeholder="Enter value"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium mb-1">Severity</label>
                  <Select
                    value={newRule.severity}
                    onValueChange={(value: any) => setNewRule({ ...newRule, severity: value })}
                  >
                    <SelectTrigger>
                      <SelectValue placeholder="Select severity" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="error">Error</SelectItem>
                      <SelectItem value="warning">Warning</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <Button onClick={handleAddRule} className="w-full">
                  Add Rule
                </Button>
              </div>
            </DialogContent>
          </Dialog>
        </div>
      </div>

      <div className="flex items-center space-x-2">
        <div className="flex-1">
          <Input
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search rows (e.g., amount > 1000 or description contains 'tax')"
            className="w-full"
          />
        </div>
        <Button onClick={handleSearch} variant="outline">
          <MagnifyingGlassIcon className="h-5 w-5" />
        </Button>
      </div>
    </div>
  );
} 