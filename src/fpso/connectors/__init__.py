"""Read-only connectors (L1): one uniform contract per source system."""

from fpso.connectors.base import Connector, Record
from fpso.connectors.sources import CONNECTORS, open_all

__all__ = ["CONNECTORS", "Connector", "Record", "open_all"]
