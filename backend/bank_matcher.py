from typing import List, Dict, Optional, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from bank_statement_parser import process_bank_statement

logger = logging.getLogger(__name__)

@dataclass
class Transaction:
    date: datetime
    amount: float
    description: str
    transaction_type: str  # 'bank' or 'invoice'
    status: str = 'pending'  # 'pending', 'matched', 'unmatched'
    match_id: Optional[str] = None
    confidence_score: float = 0.0
    source_id: Optional[str] = None

class BankMatcher:
    def __init__(self):
        self.bank_transactions: List[Transaction] = []
        self.invoice_transactions: List[Transaction] = []
        self.matches: List[Dict[str, Any]] = []
        
    def load_bank_statement(self, file_path: str) -> List[Transaction]:
        """Load and parse bank statement data"""
        try:
            df = process_bank_statement(file_path)
            
            transactions = []
            for _, row in df.iterrows():
                try:
                    date = pd.to_datetime(row.get('date'))
                    amount = float(row.get('amount', 0))
                    description = str(row.get('description', ''))
                    
                    transaction = Transaction(
                        date=date,
                        amount=amount,
                        description=description,
                        transaction_type='bank',
                        source_id=str(row.get('id', ''))
                    )
                    transactions.append(transaction)
                except Exception as e:
                    logger.warning(f"Error processing bank transaction: {e}")
                    continue
            
            self.bank_transactions = transactions
            return transactions
            
        except Exception as e:
            logger.error(f"Error loading bank statement: {e}")
            raise
    
    def load_invoice_data(self, invoice_data: List[Dict[str, Any]]) -> List[Transaction]:
        """Load invoice data for matching"""
        try:
            transactions = []
            for invoice in invoice_data:
                try:
                    date = pd.to_datetime(invoice.get('date'))
                    amount = float(invoice.get('amount', 0))
                    description = str(invoice.get('description', ''))
                    
                    transaction = Transaction(
                        date=date,
                        amount=amount,
                        description=description,
                        transaction_type='invoice',
                        source_id=str(invoice.get('id', ''))
                    )
                    transactions.append(transaction)
                except Exception as e:
                    logger.warning(f"Error processing invoice: {e}")
                    continue
            
            self.invoice_transactions = transactions
            return transactions
            
        except Exception as e:
            logger.error(f"Error loading invoice data: {e}")
            raise
    
    def find_matches(self, date_tolerance_days: int = 5, amount_tolerance: float = 0.01) -> List[Dict[str, Any]]:
        """Find matches between bank transactions and invoices"""
        matches = []
        
        for bank_tx in self.bank_transactions:
            potential_matches = []
            
            # Look for matches within date range
            date_min = bank_tx.date - timedelta(days=date_tolerance_days)
            date_max = bank_tx.date + timedelta(days=date_tolerance_days)
            
            for invoice_tx in self.invoice_transactions:
                if invoice_tx.status == 'matched':
                    continue
                    
                # Check if date is within range
                if not (date_min <= invoice_tx.date <= date_max):
                    continue
                
                # Check if amounts match within tolerance
                amount_diff = abs(bank_tx.amount - invoice_tx.amount)
                if amount_diff > amount_tolerance:
                    continue
                
                # Calculate confidence score
                confidence = self._calculate_confidence(bank_tx, invoice_tx, amount_diff)
                
                potential_matches.append({
                    'invoice': invoice_tx,
                    'confidence': confidence
                })
            
            # Sort potential matches by confidence
            potential_matches.sort(key=lambda x: x['confidence'], reverse=True)
            
            if potential_matches:
                best_match = potential_matches[0]
                match_id = f"match_{len(matches)}"
                
                # Update transaction statuses
                bank_tx.status = 'matched'
                bank_tx.match_id = match_id
                bank_tx.confidence_score = best_match['confidence']
                
                invoice_tx = best_match['invoice']
                invoice_tx.status = 'matched'
                invoice_tx.match_id = match_id
                invoice_tx.confidence_score = best_match['confidence']
                
                matches.append({
                    'match_id': match_id,
                    'bank_transaction': bank_tx,
                    'invoice_transaction': invoice_tx,
                    'confidence': best_match['confidence']
                })
            else:
                bank_tx.status = 'unmatched'
        
        self.matches = matches
        return matches
    
    def _calculate_confidence(self, bank_tx: Transaction, invoice_tx: Transaction, amount_diff: float) -> float:
        """Calculate confidence score for a potential match"""
        confidence = 1.0
        
        # Reduce confidence based on amount difference
        confidence *= (1 - amount_diff)
        
        # Reduce confidence based on date difference
        days_diff = abs((bank_tx.date - invoice_tx.date).days)
        confidence *= (1 - (days_diff * 0.1))  # Reduce by 10% per day difference
        
        # Add text similarity score if descriptions match
        if bank_tx.description and invoice_tx.description:
            similarity = self._text_similarity(bank_tx.description, invoice_tx.description)
            confidence *= (0.5 + 0.5 * similarity)  # Weight text similarity as 50%
        
        return max(0.0, min(1.0, confidence))
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity score"""
        # Convert to lowercase and split into words
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        # Calculate Jaccard similarity
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def get_unmatched_transactions(self) -> Dict[str, List[Transaction]]:
        """Get lists of unmatched transactions"""
        return {
            'bank': [tx for tx in self.bank_transactions if tx.status == 'unmatched'],
            'invoice': [tx for tx in self.invoice_transactions if tx.status == 'unmatched']
        }
    
    def update_match_status(self, match_id: str, new_status: str) -> bool:
        """Update the status of a match"""
        try:
            for match in self.matches:
                if match['match_id'] == match_id:
                    match['bank_transaction'].status = new_status
                    match['invoice_transaction'].status = new_status
                    return True
            return False
        except Exception as e:
            logger.error(f"Error updating match status: {e}")
            return False 