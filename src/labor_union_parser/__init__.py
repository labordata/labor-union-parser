"""
Labor Union Parser - Extract affiliation and designation from union names.

Example:
    >>> from labor_union_parser import extract
    >>> extract("SEIU Local 1199")
    {'affiliation': 'SEIU', 'designation': '1199'}

    >>> from labor_union_parser import Extractor
    >>> extractor = Extractor()
    >>> extractor.extract("Teamsters Local 705")
    {'affiliation': 'IBT', 'designation': '705'}

    >>> from labor_union_parser import lookup_fnum
    >>> lookup_fnum("SEIU", "1199")
    [31847, 69557, ...]
"""

from .extractor import Extractor, extract, lookup_fnum

__version__ = "0.1.0"
__all__ = ["Extractor", "extract", "lookup_fnum"]
