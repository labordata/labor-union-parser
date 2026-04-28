"""
Labor Union Parser - Match union name text to OLMS filing numbers.

Example:
    >>> from labor_union_parser import Extractor
    >>> extractor = Extractor()
    >>> extractor.extract("SEIU Local 1199")
    {'is_union': True, 'is_union_score': 0.9997, 'union_name': 'SERVICE EMPLOYEES',
     'union_name_score': 0.92, 'f_num': 509111, 'f_num_score': 0.85}
"""

from .extractor import Extractor

__version__ = "0.3.0"
__all__ = ["Extractor"]
