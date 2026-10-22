"""The mock fleet: Northwind Offshore's three FPSOs (all fictional).

Every asset's world comes from ``build_world`` with the asset's own seed, so the builder's
random draws are untouched. Only the asset code in schedule and sailing ids is changed
afterwards, on a copy.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from fpso.synthetic.truth import ASSET_CODE, World, build_world


@dataclass(frozen=True)
class Asset:
    code: str
    name: str
    seed: int
    dialect: str  # export convention of its source systems, see ``fpso.fleet.dialects``
    renames: dict[str, str] = field(default_factory=dict)  # site-specific raw name changes


KESTREL = Asset("KST", "FPSO Kestrel", 7, "sap")
MERLIN = Asset("MRL", "FPSO Merlin", 11, "maximo")
# Osprey runs the same Maximo set-up as Merlin, with two local differences.
OSPREY = Asset(
    "OSP",
    "FPSO Osprey",
    23,
    "maximo",
    renames={"udf:MAXIMO WONUM": "WO REF", "logistics/cargo_lines.csv:PO_REFERENCE": "CUST_PO"},
)
FLEET = {a.code: a for a in (KESTREL, MERLIN, OSPREY)}


def asset_world(asset: Asset, scale: float = 1.0) -> World:
    """The asset's clean world: ``build_world`` with its seed, ids carrying its code."""
    world = build_world(seed=asset.seed, scale=scale)
    if asset.code == ASSET_CODE:
        return world

    def recode(text: str) -> str:
        return text.replace(f"{ASSET_CODE}-", f"{asset.code}-").replace(
            f"{ASSET_CODE}.", f"{asset.code}."
        )

    return replace(
        world,
        activities=[
            replace(a, act_id=recode(a.act_id), wbs=recode(a.wbs)) for a in world.activities
        ],
        sailings=[replace(s, sailing_id=recode(s.sailing_id)) for s in world.sailings],
        manifest_lines=[replace(m, sailing_id=recode(m.sailing_id)) for m in world.manifest_lines],
    )
