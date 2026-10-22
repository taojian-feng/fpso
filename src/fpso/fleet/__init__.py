"""Phase 4: a fleet of mock assets, onboarded mostly by agents, with supplier agents over A2A.

- ``assets``: the fleet (FPSO Kestrel, FPSO Merlin, FPSO Osprey), each built by the same
  world builder with its own seed;
- ``dialects``: a Maximo-style export, so a sister asset arrives in a different source
  system convention from Kestrel's SAP-style extracts;
- ``mapper``: the Onboarding agent's rule mapper (no model), which maps raw fields onto the
  canonical roles the existing agents read, with a confidence each;
- ``adapter``: presents a mapped asset through the existing connectors;
- ``a2a``: supplier-side agents and the Supplier Liaison client that asks them first;
- ``cli``: ``fpso-fleet``, the whole Phase 4 check.
"""
