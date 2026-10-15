"""Clean ground truth for the mock asset FPSO Kestrel (operator: Northwind Offshore, fictional).

This module builds the "real world": what equipment exists, which work orders need
which parts, which purchase order lines buy them, which sailings carry them and which
supplier emails talk about them. Nothing here is messy; ``corrupt.py`` and ``export.py``
turn it into the messy extracts the agents actually see.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import date, timedelta

OPERATOR = "Northwind Offshore"
ASSET = "FPSO Kestrel"
ASSET_CODE = "KST"

# Equipment classes: tag prefix, class name, typical manufacturers and the spares
# each class consumes as (spare name, lead time range in days, unit weight kg).
CLASSES: dict[str, dict] = {
    "P": {
        "name": "PUMP, CENTRIFUGAL",
        "makers": ["HARLOW HX-200", "HARLOW HX-350", "CASTELL CP-80"],
        "spares": [
            ("MECHANICAL SEAL KIT", (60, 140), 12.0),
            ("BEARING SET", (20, 60), 4.0),
            ("IMPELLER", (90, 200), 35.0),
            ("COUPLING ELEMENT", (15, 45), 2.0),
            ("GASKET SET", (10, 30), 1.0),
            ("WEAR RING SET", (30, 80), 3.0),
        ],
    },
    "K": {
        "name": "COMPRESSOR, CENTRIFUGAL",
        "makers": ["NORDVIK NC-7", "BRAYTON BMC-40"],
        "spares": [
            ("DRY GAS SEAL CARTRIDGE", (120, 210), 45.0),
            ("JOURNAL BEARING KIT", (60, 150), 18.0),
            ("SUCTION FILTER ELEMENT", (20, 45), 6.0),
            ("GASKET SET", (10, 30), 2.0),
            ("THRUST BEARING KIT", (60, 150), 22.0),
        ],
    },
    "XV": {
        "name": "VALVE, BALL, ACTUATED",
        "makers": ["KESSLER KV-4", "KESSLER KV-8", "ORMOND QB-6"],
        "spares": [
            ("ACTUATOR SEAL KIT", (20, 60), 1.5),
            ("TRIM KIT", (45, 120), 9.0),
            ("STEM PACKING SET", (10, 30), 0.5),
            ("SEAT RING", (40, 100), 4.0),
        ],
    },
    "E": {
        "name": "HEAT EXCHANGER, PLATE",
        "makers": ["TALLIS PX-90", "TALLIS PX-140"],
        "spares": [
            ("PLATE GASKET SET", (30, 75), 15.0),
            ("HEAT TRANSFER PLATE", (60, 140), 8.0),
            ("SACRIFICIAL ANODE", (15, 40), 5.0),
        ],
    },
    "G": {
        "name": "GENERATOR, GAS TURBINE DRIVEN",
        "makers": ["AURELIA GT-25"],
        "spares": [
            ("AVR CONTROL CARD", (45, 120), 0.8),
            ("AIR INTAKE FILTER SET", (20, 50), 30.0),
            ("FUEL NOZZLE", (90, 180), 3.0),
            ("BORESCOPE PLUG", (30, 90), 0.4),
        ],
    },
    "CR": {
        "name": "CRANE, PEDESTAL, HYDRAULIC",
        "makers": ["MARSTEN PC-50"],
        "spares": [
            ("WIRE ROPE, 36MM", (60, 150), 900.0),
            ("SLEW BEARING SEAL KIT", (45, 110), 20.0),
            ("HYDRAULIC HOSE KIT", (15, 40), 25.0),
        ],
    },
}

# Systems on the asset with the tag area numbers used in them.
SYSTEMS = [
    ("CRUDE EXPORT", 11),
    ("HP COMPRESSION", 23),
    ("LP COMPRESSION", 24),
    ("GAS DEHYDRATION", 26),
    ("WATER INJECTION", 31),
    ("SEAWATER LIFT", 41),
    ("PRODUCED WATER", 44),
    ("FIRE WATER", 71),
    ("POWER GENERATION", 81),
    ("CARGO HANDLING", 91),
]

VENDORS = [
    ("Harlow Pumps Ltd", "harlowpumps.example"),
    ("Castell Fluid Systems", "castell-fluid.example"),
    ("Nordvik Compression AS", "nordvik.example"),
    ("Brayton Marine Rotating", "braytonmarine.example"),
    ("Kessler Valve Works", "kesslervalve.example"),
    ("Ormond Flow Control", "ormondflow.example"),
    ("Tallis Thermal", "tallis-thermal.example"),
    ("Aurelia Power Systems", "aureliapower.example"),
    ("Marsten Lifting", "marsten.example"),
    ("North Sea Seals & Gaskets", "nsseals.example"),
    ("Atlantic Bearing Supply", "atlanticbearing.example"),
    ("Coastline Industrial Distributors", "coastline-ind.example"),
]

# Which vendor supplies each maker's parts; generic spares may go to distributors instead.
MAKER_VENDOR = {
    "HARLOW": 0,
    "CASTELL": 1,
    "NORDVIK": 2,
    "BRAYTON": 3,
    "KESSLER": 4,
    "ORMOND": 5,
    "TALLIS": 6,
    "AURELIA": 7,
    "MARSTEN": 8,
}
DISTRIBUTOR_SPARES = {
    "GASKET SET": 9,
    "PLATE GASKET SET": 9,
    "STEM PACKING SET": 9,
    "BEARING SET": 10,
    "COUPLING ELEMENT": 11,
    "HYDRAULIC HOSE KIT": 11,
    "SACRIFICIAL ANODE": 11,
}

VESSELS = ["MV Skerry Supporter", "MV Northwind Tern"]


@dataclass
class Vendor:
    vendor_id: str
    name: str
    domain: str


@dataclass
class Equipment:
    equip_no: str
    tag: str
    group: str  # tag without train letter; train members share a model and spares
    eq_class: str
    system: str
    maker_model: str
    criticality: str


@dataclass
class Material:
    mat_no: str
    description: str
    spare: str
    maker_model: str
    eq_class: str
    vendor_id: str
    vendor_part: str
    lead_time_days: int
    unit_weight_kg: float
    unit: str = "EA"


@dataclass
class WorkOrder:
    wo_no: str
    equip_no: str
    wo_type: str  # PM01 corrective, PM02 preventive, PM03 turnaround
    description: str
    planned_start: date
    planned_finish: date
    in_turnaround: bool


@dataclass
class Component:
    res_no: str
    item: str
    wo_no: str
    mat_no: str
    qty: int
    req_date: date

    @property
    def key(self) -> str:
        return f"{self.res_no}-{self.item}"


@dataclass
class Activity:
    task_id: int
    act_id: str
    name: str
    wbs: str
    start: date
    finish: date
    wo_no: str | None
    equip_no: str | None
    predecessors: list[int] = field(default_factory=list)


@dataclass
class POLine:
    po_no: str
    item: str
    vendor_id: str
    mat_no: str
    qty: int
    order_date: date
    delivery_date: date
    component_key: str
    wo_no: str

    @property
    def key(self) -> str:
        return f"{self.po_no}-{self.item}"


@dataclass
class Sailing:
    sailing_id: str
    vessel: str
    depart: date


@dataclass
class ManifestLine:
    line_id: str
    sailing_id: str
    po_line_keys: list[str]  # usually one; partial deliveries split a PO line over several
    qty: int


@dataclass
class Email:
    message_id: str
    po_no: str
    vendor_id: str
    kind: str  # ack, slip, asn
    sent: date
    po_line_keys: list[str]
    new_date: date | None = None


@dataclass
class World:
    seed: int
    window_start: date
    window_end: date
    turnaround_start: date
    turnaround_end: date
    vendors: list[Vendor]
    equipment: list[Equipment]
    materials: list[Material]
    work_orders: list[WorkOrder]
    components: list[Component]
    activities: list[Activity]
    po_lines: list[POLine]
    sailings: list[Sailing]
    manifest_lines: list[ManifestLine]
    emails: list[Email]
    stock: dict[str, int]

    def links(self) -> list[tuple[str, str, str, str, str]]:
        """Cross-system links: (link_type, from_source, from_key, to_source, to_key)."""
        out = []
        for a in self.activities:
            if a.equip_no:
                out.append(("equipment_activity", "erp-pm", a.equip_no, "p6", a.act_id))
            if a.wo_no:
                out.append(("work_order_activity", "erp-pm", a.wo_no, "p6", a.act_id))
        for pl in self.po_lines:
            out.append(("component_po_line", "erp-pm", pl.component_key, "erp-mm", pl.key))
        for ml in self.manifest_lines:
            for k in ml.po_line_keys:
                out.append(("po_line_manifest_line", "erp-mm", k, "vessel", ml.line_id))
        for e in self.emails:
            out.append(("po_email", "erp-mm", e.po_no, "supplier-mail", e.message_id))
        return out


# Full-size targets from the plan (WS2); ``scale`` shrinks them for tests.
FULL_SIZE = {"equipment": 300, "work_orders": 1500}


def build_world(seed: int = 7, scale: float = 1.0, window_start: date | None = None) -> World:
    rng = random.Random(seed)
    start = window_start or date(2027, 1, 4)
    end = start + timedelta(days=182)
    ta_start = start + timedelta(days=95)
    ta_end = ta_start + timedelta(days=21)

    vendors = [Vendor(f"20{100 + i:04d}", n, d) for i, (n, d) in enumerate(VENDORS)]

    # Equipment: groups of 1-3 trains (A, B, C) sharing a model.
    n_equipment = max(12, int(FULL_SIZE["equipment"] * scale))
    equipment: list[Equipment] = []
    used_groups: set[str] = set()
    eq_seq = 10004000
    class_weights = [("P", 35), ("XV", 30), ("E", 12), ("K", 8), ("G", 5), ("CR", 3)]
    while len(equipment) < n_equipment:
        cls = rng.choices([c for c, _ in class_weights], [w for _, w in class_weights])[0]
        system, area = rng.choice(SYSTEMS)
        group = f"{cls}-{area}{rng.randint(1, 99):02d}"
        if group in used_groups:
            continue
        used_groups.add(group)
        model = f"{rng.choice(CLASSES[cls]['makers'])}/{rng.choice(['S', 'M', 'L', 'XL'])}"
        trains = rng.choice([1, 2, 2, 3]) if cls in ("P", "K", "E", "G") else 1
        for t in range(trains):
            tag = f"{group}{'ABC'[t]}" if trains > 1 else group
            eq_seq += rng.randint(1, 40)
            equipment.append(
                Equipment(
                    equip_no=str(eq_seq),
                    tag=tag,
                    group=group,
                    eq_class=cls,
                    system=system,
                    maker_model=model,
                    criticality=rng.choice(["A", "B", "B", "C"]),
                )
            )

    # Materials: one per (maker model, spare).
    materials: dict[tuple[str, str], Material] = {}
    mat_seq = 40012000
    for eq in equipment:
        for spare, (lt_lo, lt_hi), weight in CLASSES[eq.eq_class]["spares"]:
            key = (eq.maker_model, spare)
            if key in materials:
                continue
            mat_seq += rng.randint(1, 25)
            maker = eq.maker_model.split()[0]
            vidx = DISTRIBUTOR_SPARES.get(spare, MAKER_VENDOR[maker])
            if spare in DISTRIBUTOR_SPARES and rng.random() < 0.4:
                vidx = MAKER_VENDOR[maker]
            prefix = "".join(w[0] for w in spare.split()[:3] if w[0].isalpha())
            materials[key] = Material(
                mat_no=str(mat_seq),
                description=f"{spare}; {CLASSES[eq.eq_class]['name']}; {eq.maker_model}",
                spare=spare,
                maker_model=eq.maker_model,
                eq_class=eq.eq_class,
                vendor_id=vendors[vidx].vendor_id,
                vendor_part=f"{maker[:3]}-{prefix}-{rng.randint(1000, 9999)}",
                lead_time_days=rng.randint(lt_lo, lt_hi),
                unit_weight_kg=weight,
            )
    mats_by_model: dict[str, list[Material]] = {}
    for m in materials.values():
        mats_by_model.setdefault(m.maker_model, []).append(m)

    # Work orders, about a third of them in the turnaround.
    n_wo = max(40, int(FULL_SIZE["work_orders"] * scale))
    work_orders: list[WorkOrder] = []
    wo_seq = 41000100
    for _ in range(n_wo):
        eq = rng.choice(equipment)
        wo_seq += rng.randint(1, 7)
        r = rng.random()
        if r < 0.35:
            wo_type, in_ta = "PM03", True
            s = ta_start + timedelta(days=rng.randint(0, 18))
            dur = rng.randint(1, 3)
            verb = rng.choice(["OVERHAUL", "INSPECT AND REPAIR", "REPLACE INTERNALS", "RECERTIFY"])
        elif r < 0.75:
            wo_type, in_ta = "PM02", False
            s = start + timedelta(days=rng.randint(10, 175))
            dur = rng.randint(0, 2)
            verb = rng.choice(["PM SERVICE", "6M INSPECTION", "LUBE AND CHECK", "CALIBRATE"])
        else:
            wo_type, in_ta = "PM01", False
            s = start + timedelta(days=rng.randint(5, 175))
            dur = rng.randint(1, 4)
            verb = rng.choice(["REPAIR LEAK", "REPLACE FAILED", "INVESTIGATE TRIP", "HIGH VIB"])
        work_orders.append(
            WorkOrder(
                wo_no=str(wo_seq),
                equip_no=eq.equip_no,
                wo_type=wo_type,
                description=f"{eq.tag} {verb}",
                planned_start=s,
                planned_finish=s + timedelta(days=dur),
                in_turnaround=in_ta,
            )
        )
    eq_by_no = {e.equip_no: e for e in equipment}

    # Components: 0-4 parts per WO, reserved against the WO.
    components: list[Component] = []
    res_seq = 12030000
    for wo in work_orders:
        eq = eq_by_no[wo.equip_no]
        n_parts = (
            rng.choice([0, 1, 1, 2, 2, 3, 3, 4])
            if wo.wo_type != "PM02"
            else rng.choice([0, 0, 1, 1, 2])
        )
        if wo.in_turnaround:
            n_parts = max(n_parts, 2)
        parts = rng.sample(
            mats_by_model[eq.maker_model], min(n_parts, len(mats_by_model[eq.maker_model]))
        )
        if not parts:
            continue
        res_seq += rng.randint(1, 9)
        for i, m in enumerate(parts):
            components.append(
                Component(
                    res_no=f"00{res_seq}",
                    item=f"{i + 1:04d}",
                    wo_no=wo.wo_no,
                    mat_no=m.mat_no,
                    qty=rng.choice([1, 1, 1, 2, 2, 4, 6]) if m.unit_weight_kg < 50 else 1,
                    req_date=wo.planned_start - timedelta(days=rng.randint(3, 10)),
                )
            )

    # Stock covers about a quarter of components; the rest are bought.
    mat_by_no = {m.mat_no: m for m in materials.values()}
    stock: dict[str, int] = {}
    to_buy: list[Component] = []
    for c in components:
        if rng.random() < 0.25:
            stock[c.mat_no] = stock.get(c.mat_no, 0) + c.qty + rng.randint(0, 3)
        else:
            to_buy.append(c)
    for m in mat_by_no.values():
        if m.mat_no not in stock and rng.random() < 0.3:
            stock[m.mat_no] = rng.randint(1, 6)

    # PO lines: grouped into POs by vendor and order week.
    po_lines: list[POLine] = []
    wo_by_no = {w.wo_no: w for w in work_orders}
    po_seq = 4500210000
    open_pos: dict[tuple[str, int], tuple[str, int]] = {}
    for c in sorted(to_buy, key=lambda c: c.req_date):
        m = mat_by_no[c.mat_no]
        delivery = c.req_date - timedelta(days=rng.randint(5, 25))
        if rng.random() < 0.15:  # some lines are planned late already
            delivery = c.req_date + timedelta(days=rng.randint(1, 20))
        order = delivery - timedelta(days=m.lead_time_days)
        bucket = (m.vendor_id, order.toordinal() // 7)
        if bucket not in open_pos or open_pos[bucket][1] >= rng.randint(3, 9):
            po_seq += rng.randint(1, 30)
            open_pos[bucket] = (str(po_seq), 0)
        po_no, count = open_pos[bucket]
        open_pos[bucket] = (po_no, count + 1)
        po_lines.append(
            POLine(
                po_no=po_no,
                item=f"{(count + 1) * 10:05d}",
                vendor_id=m.vendor_id,
                mat_no=m.mat_no,
                qty=c.qty,
                order_date=order,
                delivery_date=delivery,
                component_key=c.key,
                wo_no=wo_by_no[c.wo_no].wo_no,
            )
        )

    # P6: turnaround WOs and most corrective WOs are scheduled; routine PMs mostly are not.
    activities: list[Activity] = []
    task_seq = 5000
    act_seq = {"TA": 1000, "MNT": 2000}
    last_in_system: dict[str, int] = {}
    for wo in sorted(work_orders, key=lambda w: w.planned_start):
        eq = eq_by_no[wo.equip_no]
        scheduled = (
            wo.in_turnaround
            or (wo.wo_type == "PM01" and rng.random() < 0.8)
            or (rng.random() < 0.3)
        )
        if not scheduled:
            continue
        area = "TA" if wo.in_turnaround else "MNT"
        act_seq[area] += 10
        task_seq += 1
        verb = wo.description.split(" ", 1)[1]
        act = Activity(
            task_id=task_seq,
            act_id=f"{ASSET_CODE}-{area}-{act_seq[area]}",
            name=f"{eq.tag} {verb.title()}",
            wbs=f"{ASSET_CODE}.{area}.{eq.system.replace(' ', '_')}",
            start=wo.planned_start,
            finish=wo.planned_finish,
            wo_no=wo.wo_no,
            equip_no=eq.equip_no,
        )
        if wo.in_turnaround and eq.system in last_in_system:
            act.predecessors.append(last_in_system[eq.system])
        if wo.in_turnaround:
            last_in_system[eq.system] = task_seq
        activities.append(act)
    # Activities with no work order or equipment (scaffolding, isolation), as noise.
    for _ in range(max(3, len(activities) // 12)):
        task_seq += 1
        act_seq["TA"] += 10
        system, _ = rng.choice(SYSTEMS)
        d = ta_start + timedelta(days=rng.randint(0, 20))
        activities.append(
            Activity(
                task_id=task_seq,
                act_id=f"{ASSET_CODE}-TA-{act_seq['TA']}",
                name=rng.choice(
                    [
                        "Erect Scaffold",
                        "Isolate And Purge",
                        "Remove Insulation",
                        "Leak Test And Reinstate",
                    ]
                )
                + f" - {system.title()}",
                wbs=f"{ASSET_CODE}.TA.{system.replace(' ', '_')}",
                start=d,
                finish=d + timedelta(days=1),
                wo_no=None,
                equip_no=None,
            )
        )

    # Sailings twice a week from the shore base.
    sailings: list[Sailing] = []
    d = start - timedelta(days=60)
    s_seq = 100
    while d <= end:
        for offset in (0, 3):
            s_seq += 1
            sailings.append(
                Sailing(
                    f"{ASSET_CODE}-S-{s_seq:04d}", VESSELS[s_seq % 2], d + timedelta(days=offset)
                )
            )
        d += timedelta(days=7)

    def next_sailing(after: date) -> Sailing | None:
        for s in sailings:
            if s.depart >= after:
                return s
        return None

    manifest_lines: list[ManifestLine] = []
    ml_seq = 0
    for pl in po_lines:
        first = next_sailing(pl.delivery_date + timedelta(days=rng.randint(1, 4)))
        if first is None:
            continue
        if pl.qty > 1 and rng.random() < 0.15:  # partial delivery over two sailings
            part = rng.randint(1, pl.qty - 1)
            second = next_sailing(first.depart + timedelta(days=rng.randint(5, 20)))
            shipments = [(first, part)] + ([(second, pl.qty - part)] if second else [])
        else:
            shipments = [(first, pl.qty)]
        for sailing, qty in shipments:
            ml_seq += 1
            manifest_lines.append(
                ManifestLine(f"ML-{ml_seq:06d}", sailing.sailing_id, [pl.key], qty)
            )

    # Supplier emails: acknowledgement, slip notices, and dispatch notices.
    emails: list[Email] = []
    msg_seq = 0
    lines_by_po: dict[str, list[POLine]] = {}
    for pl in po_lines:
        lines_by_po.setdefault(pl.po_no, []).append(pl)
    for po_no, lines in lines_by_po.items():
        vendor_id = lines[0].vendor_id
        order = min(pl.order_date for pl in lines)
        if rng.random() < 0.8:
            msg_seq += 1
            emails.append(
                Email(
                    f"msg-{msg_seq:06d}",
                    po_no,
                    vendor_id,
                    "ack",
                    order + timedelta(days=rng.randint(1, 5)),
                    [pl.key for pl in lines],
                )
            )
        if rng.random() < 0.25:
            slipped = rng.sample(lines, rng.randint(1, len(lines)))
            for pl in slipped:
                new = pl.delivery_date + timedelta(days=rng.randint(7, 42))
                msg_seq += 1
                emails.append(
                    Email(
                        f"msg-{msg_seq:06d}",
                        po_no,
                        vendor_id,
                        "slip",
                        pl.delivery_date - timedelta(days=rng.randint(5, 30)),
                        [pl.key],
                        new_date=new,
                    )
                )
        if rng.random() < 0.5:
            msg_seq += 1
            last = max(pl.delivery_date for pl in lines)
            emails.append(
                Email(
                    f"msg-{msg_seq:06d}",
                    po_no,
                    vendor_id,
                    "asn",
                    last - timedelta(days=rng.randint(0, 3)),
                    [pl.key for pl in lines],
                )
            )

    return World(
        seed=seed,
        window_start=start,
        window_end=end,
        turnaround_start=ta_start,
        turnaround_end=ta_end,
        vendors=vendors,
        equipment=equipment,
        materials=list(materials.values()),
        work_orders=work_orders,
        components=components,
        activities=activities,
        po_lines=po_lines,
        sailings=sailings,
        manifest_lines=manifest_lines,
        emails=emails,
        stock=stock,
    )
