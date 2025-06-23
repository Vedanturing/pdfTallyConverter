from typing import List, Dict, Any, Optional
import pandas as pd
from datetime import datetime
import logging
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)

@dataclass
class GSTInvoice:
    invoice_no: str
    invoice_date: datetime
    customer_gstin: str
    customer_name: str
    place_of_supply: str
    taxable_value: float
    gst_rate: float
    igst: float = 0.0
    cgst: float = 0.0
    sgst: float = 0.0
    total_amount: float = 0.0
    invoice_type: str = "B2B"  # B2B, B2C, Export, etc.
    reverse_charge: bool = False
    export_type: Optional[str] = None  # With Payment, Without Payment
    shipping_bill_no: Optional[str] = None
    shipping_bill_date: Optional[datetime] = None
    port_code: Optional[str] = None

class GSTHelper:
    def __init__(self):
        self.invoices: List[GSTInvoice] = []
    
    def clear_invoices(self):
        """Clear all invoices from the collection"""
        self.invoices.clear()
    
    def get_invoice_count(self) -> int:
        """Get the number of invoices in the collection"""
        return len(self.invoices)
        
    def add_invoice(self, invoice_data: Dict[str, Any]) -> GSTInvoice:
        """Add an invoice to the collection"""
        try:
            # Calculate GST amounts based on rate and taxable value
            taxable_value = float(invoice_data.get('taxable_value', 0))
            gst_rate = float(invoice_data.get('gst_rate', 0))
            place_of_supply = invoice_data.get('place_of_supply', '')
            
            # Calculate GST components
            total_gst = (taxable_value * gst_rate) / 100
            
            # For inter-state supply, use IGST
            if self._is_interstate_supply(place_of_supply):
                igst = total_gst
                cgst = sgst = 0.0
            else:
                # For intra-state supply, split between CGST and SGST
                igst = 0.0
                cgst = sgst = total_gst / 2
            
            invoice = GSTInvoice(
                invoice_no=invoice_data.get('invoice_no', ''),
                invoice_date=pd.to_datetime(invoice_data.get('invoice_date')),
                customer_gstin=invoice_data.get('customer_gstin', ''),
                customer_name=invoice_data.get('customer_name', ''),
                place_of_supply=place_of_supply,
                taxable_value=taxable_value,
                gst_rate=gst_rate,
                igst=igst,
                cgst=cgst,
                sgst=sgst,
                total_amount=taxable_value + total_gst,
                invoice_type=invoice_data.get('invoice_type', 'B2B'),
                reverse_charge=invoice_data.get('reverse_charge', False),
                export_type=invoice_data.get('export_type'),
                shipping_bill_no=invoice_data.get('shipping_bill_no'),
                shipping_bill_date=pd.to_datetime(invoice_data.get('shipping_bill_date')) if invoice_data.get('shipping_bill_date') else None,
                port_code=invoice_data.get('port_code')
            )
            
            self.invoices.append(invoice)
            return invoice
            
        except Exception as e:
            logger.error(f"Error adding invoice: {str(e)}")
            raise
    
    def _is_interstate_supply(self, place_of_supply: str) -> bool:
        """Determine if supply is interstate based on place of supply"""
        # TODO: Implement proper state code comparison logic
        # For now, assuming all supplies are intra-state
        return False
    
    def generate_gstr1_json(self, period: str) -> Dict[str, Any]:
        """Generate GSTR-1 format JSON"""
        try:
            b2b_invoices = []
            b2cl_invoices = []
            export_invoices = []
            
            for invoice in self.invoices:
                if invoice.invoice_type == "B2B":
                    b2b_invoices.append({
                        "inum": invoice.invoice_no,
                        "idt": invoice.invoice_date.strftime("%d-%m-%Y"),
                        "val": invoice.total_amount,
                        "pos": invoice.place_of_supply,
                        "rchrg": "Y" if invoice.reverse_charge else "N",
                        "inv_typ": "R",
                        "itms": [{
                            "num": 1,
                            "itm_det": {
                                "txval": invoice.taxable_value,
                                "rt": invoice.gst_rate,
                                "iamt": invoice.igst,
                                "camt": invoice.cgst,
                                "samt": invoice.sgst
                            }
                        }]
                    })
                elif invoice.invoice_type == "EXPORT":
                    export_invoices.append({
                        "inum": invoice.invoice_no,
                        "idt": invoice.invoice_date.strftime("%d-%m-%Y"),
                        "val": invoice.total_amount,
                        "sbpcode": invoice.port_code,
                        "sbnum": invoice.shipping_bill_no,
                        "sbdt": invoice.shipping_bill_date.strftime("%d-%m-%Y") if invoice.shipping_bill_date else "",
                        "itms": [{
                            "txval": invoice.taxable_value,
                            "rt": invoice.gst_rate,
                            "iamt": invoice.igst if invoice.export_type == "With Payment" else 0
                        }]
                    })
            
            return {
                "gstin": "XXXXXXXXXXXX",  # To be filled by user
                "fp": period,
                "version": "GST3.0.4",
                "hash": "hash",  # To be generated
                "b2b": b2b_invoices,
                "exp": export_invoices
            }
            
        except Exception as e:
            logger.error(f"Error generating GSTR-1 JSON: {str(e)}")
            raise
    
    def generate_gstr3b_json(self, period: str) -> Dict[str, Any]:
        """Generate GSTR-3B format JSON"""
        try:
            # Calculate totals
            total_taxable = sum(inv.taxable_value for inv in self.invoices)
            total_igst = sum(inv.igst for inv in self.invoices)
            total_cgst = sum(inv.cgst for inv in self.invoices)
            total_sgst = sum(inv.sgst for inv in self.invoices)
            
            return {
                "gstin": "XXXXXXXXXXXX",  # To be filled by user
                "ret_period": period,
                "sup_details": {
                    "osup_det": {
                        "txval": total_taxable,
                        "iamt": total_igst,
                        "camt": total_cgst,
                        "samt": total_sgst,
                        "csamt": 0
                    },
                    "osup_zero": {
                        "txval": sum(inv.taxable_value for inv in self.invoices if inv.invoice_type == "EXPORT"),
                        "iamt": 0,
                        "csamt": 0
                    },
                    "osup_nil_exmp": {
                        "txval": 0
                    },
                    "isup_rev": {
                        "txval": sum(inv.taxable_value for inv in self.invoices if inv.reverse_charge),
                        "iamt": sum(inv.igst for inv in self.invoices if inv.reverse_charge),
                        "camt": sum(inv.cgst for inv in self.invoices if inv.reverse_charge),
                        "samt": sum(inv.sgst for inv in self.invoices if inv.reverse_charge),
                        "csamt": 0
                    }
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating GSTR-3B JSON: {str(e)}")
            raise
    
    def generate_excel_report(self, output_path: str):
        """Generate Excel report with GST details"""
        try:
            if not self.invoices:
                # Create empty DataFrame with headers if no invoices
                df = pd.DataFrame(columns=[
                    'Invoice No', 'Invoice Date', 'Customer GSTIN', 'Customer Name',
                    'Place of Supply', 'Invoice Type', 'Taxable Value', 'GST Rate',
                    'IGST', 'CGST', 'SGST', 'Total Amount', 'Reverse Charge'
                ])
            else:
                # Create DataFrame from invoices
                df = pd.DataFrame([{
                    'Invoice No': inv.invoice_no,
                    'Invoice Date': inv.invoice_date.strftime('%Y-%m-%d') if inv.invoice_date else '',
                    'Customer GSTIN': inv.customer_gstin,
                    'Customer Name': inv.customer_name,
                    'Place of Supply': inv.place_of_supply,
                    'Invoice Type': inv.invoice_type,
                    'Taxable Value': inv.taxable_value,
                    'GST Rate': inv.gst_rate,
                    'IGST': inv.igst,
                    'CGST': inv.cgst,
                    'SGST': inv.sgst,
                    'Total Amount': inv.total_amount,
                    'Reverse Charge': 'Yes' if inv.reverse_charge else 'No'
                } for inv in self.invoices])
            
            # Add summary sheet
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Invoices', index=False)
                
                # Create summary sheet
                if not df.empty:
                    summary = pd.DataFrame([{
                        'Description': 'Total',
                        'Taxable Value': df['Taxable Value'].sum(),
                        'IGST': df['IGST'].sum(),
                        'CGST': df['CGST'].sum(),
                        'SGST': df['SGST'].sum(),
                        'Total Amount': df['Total Amount'].sum()
                    }])
                else:
                    summary = pd.DataFrame([{
                        'Description': 'Total',
                        'Taxable Value': 0,
                        'IGST': 0,
                        'CGST': 0,
                        'SGST': 0,
                        'Total Amount': 0
                    }])
                summary.to_excel(writer, sheet_name='Summary', index=False)
                
        except Exception as e:
            logger.error(f"Error generating Excel report: {str(e)}")
            raise
    
    def validate_gstin(self, gstin: str) -> bool:
        """Validate GSTIN format"""
        # Basic GSTIN format validation
        if not gstin or len(gstin) != 15:
            return False
        
        # Check if first two characters are digits (state code)
        if not gstin[:2].isdigit():
            return False
        
        # Check if characters 3-7 are letters (PAN first 5)
        if not gstin[2:7].isalpha():
            return False
        
        # Check if characters 8-11 are digits (PAN next 4)
        if not gstin[7:11].isdigit():
            return False
        
        # Check if character 12 is letter (PAN last char)
        if not gstin[11].isalpha():
            return False
        
        # Check if character 13 is digit (entity number)
        if not gstin[12].isdigit():
            return False
        
        # Check if character 14 is Z (default for normal taxpayers)
        if gstin[13] != 'Z':
            return False
        
        # Check if last character is alphanumeric (checksum)
        if not gstin[14].isalnum():
            return False
        
        return True 