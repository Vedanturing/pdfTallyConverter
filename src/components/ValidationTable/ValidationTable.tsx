import React, { useState, useEffect } from 'react';
import { toast } from 'react-hot-toast';
import axios from 'axios';
import { API_URL } from '../../config';
import { BsCheckCircle, BsXCircle, BsSearch, BsInfoCircle, BsArrowLeft } from 'react-icons/bs';
import { MdUndo, MdRedo } from 'react-icons/md';
import { FaFileCsv, FaFileExcel, FaFileCode } from 'react-icons/fa';
import { useNavigate, useLocation } from 'react-router-dom';
import {
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  IconButton,
  Tooltip,
  Button,
  Box,
  Typography,
  Chip,
} from '@mui/material';
import {
  CheckCircle as CheckCircleIcon,
  Warning as WarningIcon,
  Error as ErrorIcon,
  Info as InfoIcon,
  Undo as UndoIcon,
  Redo as RedoIcon,
  AutoFixHigh as AutoFixIcon,
} from '@mui/icons-material';

interface CellMetadata {
  error?: boolean;
  confidence?: number;
  status?: 'corrected' | 'ignored' | 'needs-review' | 'original';
}

interface CellData {
  value: string | number;
  metadata: CellMetadata;
}

interface TableData {
  [key: string]: CellData;
}

interface ValidationIssue {
  column: string;
  type: string;
  severity: 'critical' | 'warning' | 'info';
  fix: string | null;
  original?: string;
}

interface ValidationResult {
  row: number;
  issues: ValidationIssue[];
}

interface ValidationSummary {
  critical: number;
  warning: number;
  info: number;
}

const ValidationTable: React.FC = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const [data, setData] = useState<any[]>([]);
  const [validationResults, setValidationResults] = useState<ValidationResult[]>([]);
  const [summary, setSummary] = useState<ValidationSummary>({ critical: 0, warning: 0, info: 0 });
  const [showErrors, setShowErrors] = useState(true);
  const [history, setHistory] = useState<any[][]>([]);
  const [historyIndex, setHistoryIndex] = useState(0);
  const [editedCells, setEditedCells] = useState<Set<string>>(new Set());

  useEffect(() => {
    if (location.state?.data) {
      setData(location.state.data);
      setHistory([location.state.data]);
    } else {
      navigate('/convert');
    }
  }, [location.state, navigate]);

  useEffect(() => {
    if (data.length > 0) {
      validateData();
    }
  }, [data]);

  const validateData = async () => {
    try {
      const response = await fetch('http://localhost:5175/validate-data', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(data),
      });
      const result = await response.json();
      setValidationResults(result.validation_results);
      setSummary(result.summary);
    } catch (error) {
      console.error('Error validating data:', error);
    }
  };

  const handleCellEdit = (rowIndex: number, column: string, value: string) => {
    const newData = [...data];
    newData[rowIndex][column] = value;
    
    // Add to history
    const newHistory = history.slice(0, historyIndex + 1);
    newHistory.push(newData);
    setHistory(newHistory);
    setHistoryIndex(newHistory.length - 1);
    
    // Mark cell as edited
    setEditedCells(prev => new Set([...prev, `${rowIndex}-${column}`]));
    
    setData(newData);
  };

  const handleUndo = () => {
    if (historyIndex > 0) {
      setHistoryIndex(historyIndex - 1);
      setData(history[historyIndex - 1]);
    }
  };

  const handleRedo = () => {
    if (historyIndex < history.length - 1) {
      setHistoryIndex(historyIndex + 1);
      setData(history[historyIndex + 1]);
    }
  };

  const handleAutoFix = (rowIndex: number, issue: ValidationIssue) => {
    if (issue.fix) {
      handleCellEdit(rowIndex, issue.column, issue.fix);
    }
  };

  const getCellStyle = (rowIndex: number, column: string) => {
    const issue = validationResults.find(r => r.row === rowIndex)?.issues.find(i => i.column === column);
    const isEdited = editedCells.has(`${rowIndex}-${column}`);
    
    if (isEdited) {
      return { backgroundColor: '#e3f2fd' };
    }
    
    if (issue && showErrors) {
      switch (issue.severity) {
        case 'critical':
          return { backgroundColor: '#ffebee' };
        case 'warning':
          return { backgroundColor: '#fff3e0' };
        case 'info':
          return { backgroundColor: '#e1f5fe' };
        default:
          return {};
      }
    }
    
    return {};
  };

  const getTooltipContent = (rowIndex: number, column: string) => {
    const issue = validationResults.find(r => r.row === rowIndex)?.issues.find(i => i.column === column);
    if (issue) {
      return (
        <div>
          <Typography variant="body2">{issue.type}</Typography>
          {issue.fix && (
            <Typography variant="body2">
              Suggested fix: {issue.fix}
            </Typography>
          )}
        </div>
      );
    }
    return '';
  };

  if (data.length === 0) {
    return null;
  }

  return (
    <Box sx={{ width: '100%' }}>
      <Box sx={{ mb: 2, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Button variant="outlined" onClick={() => navigate('/convert')}>
          ← Back to Convert
        </Button>
        <Box sx={{ display: 'flex', gap: 1 }}>
          <Chip
            icon={<ErrorIcon />}
            label={`${summary.critical} Critical`}
            color="error"
          />
          <Chip
            icon={<WarningIcon />}
            label={`${summary.warning} Warning`}
            color="warning"
          />
          <Chip
            icon={<InfoIcon />}
            label={`${summary.info} Info`}
            color="info"
          />
        </Box>
        <Box>
          <IconButton onClick={handleUndo} disabled={historyIndex === 0}>
            <UndoIcon />
          </IconButton>
          <IconButton onClick={handleRedo} disabled={historyIndex === history.length - 1}>
            <RedoIcon />
          </IconButton>
          <Button
            variant="outlined"
            onClick={() => setShowErrors(!showErrors)}
            sx={{ ml: 1 }}
          >
            {showErrors ? 'Hide Errors' : 'Show Errors'}
          </Button>
        </Box>
      </Box>

      <TableContainer component={Paper}>
        <Table>
          <TableHead>
            <TableRow>
              {Object.keys(data[0] || {}).map((column) => (
                <TableCell key={column}>{column}</TableCell>
              ))}
            </TableRow>
          </TableHead>
          <TableBody>
            {data.map((row, rowIndex) => (
              <TableRow key={rowIndex}>
                {Object.entries(row).map(([column, value]) => {
                  const issue = validationResults.find(r => r.row === rowIndex)?.issues.find(i => i.column === column);
                  return (
                    <TableCell
                      key={`${rowIndex}-${column}`}
                      style={getCellStyle(rowIndex, column)}
                      onClick={() => {
                        const newValue = prompt('Edit value:', String(value));
                        if (newValue !== null) {
                          handleCellEdit(rowIndex, column, newValue);
                        }
                      }}
                    >
                      <Tooltip title={getTooltipContent(rowIndex, column)} arrow>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                          <span>{String(value)}</span>
                          {issue && issue.fix && (
                            <IconButton
                              size="small"
                              onClick={(e: React.MouseEvent) => {
                                e.stopPropagation();
                                handleAutoFix(rowIndex, issue);
                              }}
                            >
                              <AutoFixIcon fontSize="small" />
                            </IconButton>
                          )}
                        </Box>
                      </Tooltip>
                    </TableCell>
                  );
                })}
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>
    </Box>
  );
};

export default ValidationTable;
