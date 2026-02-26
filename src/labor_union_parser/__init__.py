"""
Labor Union Parser - Extract union fields from union names.

Example:
    >>> from labor_union_parser import Extractor
    >>> extractor = Extractor()
    >>> extractor.extract("SEIU Local 1199")
    {'is_union': True, 'union_score': 0.9997, 'union_name': 'SERVICE EMPLOYEES',
     'desig_name': 'LU', 'desig_num': '1199', ...}
"""

from .extractor import Extractor

__version__ = "0.2.0"
__all__ = ["Extractor"]
