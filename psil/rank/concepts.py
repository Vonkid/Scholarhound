import re

# Tier weights for pre-filter signal scoring
TIER1_WEIGHT = 5
TIER2_WEIGHT = 3
TIER3_WEIGHT = 1

# ============================================================================
# TIER 1 — Core Research Signals (highest weight)
# ============================================================================
TIER1_CORE_SIGNALS = [
    # NIR photochemistry / photon utilization
    "NIR-triggered release",
    "near-infrared-triggered release",
    "light-triggered release",
    "photoresponsive release",
    "photocleavage",
    "photolysis",
    "photouncaging",
    "photoactivation",
    "photocaging",
    "oxygen-independent photochemistry",
    "ROS-independent photochemistry",
    "hypoxia-tolerant phototherapy",
    "hypoxia-tolerant photoactivation",
    "direct bond cleavage",
    "photo-SN1",
    "heterolysis",
    "bond scission",
    "excited-state dynamics",
    "excited-state routing",
    "excited-state energy redistribution",
    "excited-state control",
    "energy redistribution",
    "energy transfer pathway",
    "energy routing",
    "photoinduced electron transfer",
    "PET mechanism",
    "photoredox activation",
    "photon utilization",
    "optical field-mediated reaction",
    "field-enhanced photochemistry",
    "nanostructure-enhanced photochemistry",
    "photochemical bond cleavage",
    "local optical environment",

    # Nanophotonics / optical sensing
    "light-matter interaction",
    "light-matter coupling",
    "structure-field coupling",
    "optical confinement",
    "field localization",
    "electromagnetic field enhancement",
    "near-field enhancement",
    "nanophotonics",
    "dielectric nanophotonics",
    "dielectric nanoparticle",
    "dielectric metasurface",
    "silicon photonics",
    "lithium niobate",
    "thin-film lithium niobate",
    "nonlinear optics",
    "integrated nonlinear optics",
    "nonlinear photonics",
    "microresonator",
    "microring resonator",
    "structured light",
    "structured optical vortex",
    "optical vortex",
    "optical skyrmion",
    "vortex microcomb",
    "microcomb",
    "spin-orbit coupling",
    "topological charge",
    "nonlinear Cherenkov radiation",
    "porous silicon",
    "photonic nanostructure",
    "Mie resonance",
    "radiative Q-factor modulation",
    "Q-factor modulation",
    "Q-factor sensing",
    "bound state in the continuum",
    "BIC metasurface",
    "quasi-BIC",
    "photonic crystal biosensor",
    "metasurface biosensor",
    "optical resonance biosensing",
    "resonance linewidth modulation",
    "linewidth narrowing",
    "linewidth broadening",
    "refractometric biosensing",
    "local density of optical states",
    "LDOS",
    "Purcell effect",
    "strong coupling",
    "polariton",
    "polaritonic biosensing",
    "electromagnetic hot spot",
    "optical antenna",
    "nanophotonic biosensor",

    # Bioelectronics / electrochemical sensing
    "organic electrochemical transistor",
    "OECT",
    "molecular bioelectronics",
    "small-molecule sensing",
    "small-molecule recognition",
    "RNA sensing",
    "bioelectronic interface",
    "adaptive bioelectronics",
    "strain-resilient electrochemical interface",
    "stretchable electrochemical biointerface",
    "mechanically gated biointerface",
    "force-coupled electrochemical sensing",
    "mechano-electrochemical sensing",
    "mechanobiology-enabled sensing",
    "dynamic biointerface",
    "soft bioelectronic interface",
    "stretchable OECT",
    "organic electrochemical transistor array",
    "electrochemical organic light-emitting transistor",
    "organic light-emitting transistor",
    "electric double layer",
    "contact injection",
    "ion transport enhancer",
    "single-ion conductive elastomer",
    "polyelectrolyte elastomer",
    "high-dielectric plasticizer",
    "ionogel gate dielectric",
    "ionogel-gated transistor",
    "synaptic transistor",
    "artificial mechanoreceptor",
    "triboelectric-capacitive gating",
    "immune-compatible semiconducting polymer",
    "foreign-body response",
    "mixed electron-ion conductivity",
    "selenophene semiconducting polymer",
    "organic semiconductor phase behavior",
    "semiconducting polymer blend stability",
    "configurational entropy",
    "iontronics",
    "neuromorphic bioelectronics",
    "on-body edge computing",
    "mixed ionic-electronic conduction",

    # Mechanobiology
    "mechanotransduction",
    "mechanobiology-enabled sensing",
    "mechano-electrochemical sensing",
    "force-coupled electrochemical sensing",
    "mechanically gated bioelectronics",
    "dynamic biointerfaces",

    # Active nanomaterial mediators (NOT passive carriers)
    "nanozyme",
    "single-atom nanozyme",
    "nanozyme cascade",
    "plasmonic catalyst",
    "plasmon-enhanced catalysis",
    "plasmonic nanoreactor",
    "hot electron transfer",
    "hot electron injection",
    "photocatalytic nanomaterial",
    "nano-heterojunction",
    "nanoheterojunction",
    "photothermal mediator",
    "nanoscale energy conversion",
    "photonic nanomaterial",
    "quantum dot sensor",
    "upconversion nanoparticle",
    "sonodynamic",
    "chemodynamic",
    "piezoelectric nanomaterial",
    "pyroelectric nanomaterial",
    "field-enhanced nanomaterial",
]

# ============================================================================
# TIER 2 — Adjacent / Bridge Signals (medium weight)
# ============================================================================
TIER2_ADJACENT_SIGNALS = [
    # Controlled release / stimuli-responsive
    "controlled release",
    "stimuli-responsive systems",
    "spatiotemporal control",
    "photothermal activation",
    "photothermal gating",
    "photothermal conversion",

    # Biosensing platforms
    "electrochemical biosensor",
    "transistor biosensor",
    "label-free sensing",
    "signal transduction",

    # Bioelectronics materials
    "hydrogel electronics",
    "conductive hydrogel",
    "ionic-electronic coupling",
    "electrochemical transduction",
    "ion transport material",
    "ion mobility",
    "ion dissociation",

    # Extracellular vesicle / organoid / stem cell
    "extracellular vesicle",
    "EV sensing",
    "organoid-derived EV",
    "organoid-derived extracellular vesicles",
    "organoid secretome",
    "secretome analysis",
    "secretome profiling",
    "extracellular vesicle diagnostics",
    "EV photonic sensing",
    "EV bioelectronic sensing",
    "microfluidic EV sensing",
    "organoid-on-a-chip",
    "organoid-on-chip sensing",
    "organ-on-a-chip",
    "microphysiological systems",
    "microphysiological sensing",
    "non-destructive organoid monitoring",
    "microfluidic platform",
    "organoid",
    "brain organoid",
    "cerebral organoid",
    "neural organoid",
    "iPSC-derived",
    "stem cell niche",
    "intercellular communication",
    "cell-cell communication",
    "liquid biopsy",
    "single-cell",
    "spatial transcriptomics",

    # Neurological / BBB / neurodegeneration
    "BBB delivery",
    "brain delivery",
    "neurodegenerative disease",
    "neurodegeneration",
    "neuroinflammation",
    "synaptic",
    "astrocyte",
    "microglia",
    "neural circuit",
    "neuronal",
    "brain clearance",
    "glymphatic",
    "cerebrospinal fluid",
    "CSF",
    "calcium signaling",
    "optogenetics",
    "electrophysiology",

    # Smart materials
    "programmable materials",
    "adaptive materials",
    "functional materials",

    # Wearable / flexible bioelectronics
    "soft bioelectronics",
    "wearable biosensing",
    "flexible bioelectronics",
    "strain-responsive electronics",
    "iontronics",

    # Personal Legacy — gold nanoclusters
    "gold nanocluster",
    "gold nanoclusters",
    "AuNC",
    "AuNCs",

    # Personal Legacy — fluorophores
    "BODIPY",
    "cyanine",
    "xanthene",

    # Personal Legacy — neuro / disease
    "alpha-synuclein",
    "tau",
    "Alzheimer's disease",
    "blood-brain barrier",

    # Personal Legacy — ROS / compounds
    "ROS scavenging",
    "salidroside",
    "minocycline",

    # Personal Legacy — photochemistry (note: photocleavage/photoactivation also in TIER1)
    "two-photon",
    "photouncaging",

    # NIR photochemistry materials
    "NIR photothermal conversion",
    "NIR photochemistry",
    "two-photon materials",
]

# ============================================================================
# TIER 2_META — AI for Scientific Discovery (weight 3)
# ============================================================================
TIER2_META_SIGNALS = [
    # Multi-agent systems
    "multi-agent system",
    "multi-agent discovery",
    "agent collaboration",
    "agent orchestration",
    "blackboard architecture",
    "blackboard system",
    # Automated scientific discovery
    "automated scientific discovery",
    "autonomous experiment",
    "self-driving lab",
    "automated hypothesis",
    "hypothesis generation",
    "autonomous science",
    # AI reasoning
    "neurosymbolic",
    "symbolic reasoning",
    "truth maintenance",
    "belief revision",
    "knowledge representation",
    "semantic network",
    "argumentation framework",
    "defeasible reasoning",
    # Literature-based discovery
    "literature-based discovery",
    "literature mining",
    "text mining biomedical",
    "scientific knowledge graph",
    "knowledge extraction",
    "relation extraction biomedical",
    # LLM for science
    "large language model scientific",
    "LLM scientific discovery",
    "LLM agent science",
    "foundation model science",
    "AI scientist",
    "robot scientist",
    # Benchmarking
    "scientific discovery benchmark",
    "laboratory automation",
    "high-throughput experiment",
    "automated data analysis",
    "computational discovery",
]

# ============================================================================
# TIER 3 — Weak / Generic Signals (lowest weight, but not negative)
# ============================================================================
TIER3_WEAK_SIGNALS = [
    "drug delivery",
    "nanomedicine",
    "theranostics",
    "biosensing",
    "diagnostics",
    "point-of-care",
    "microfluidics",
    "lab-on-a-chip",
    "hydrogel",
    "nanoparticle",
    "nanocluster",
    "phototherapy",
    "photodynamic therapy",
    "PDT",
    "photothermal therapy",
    "PTT",
    "oxidative stress",
    "ROS scavenging",
    "inflammation",
    "Alzheimer",
    "Parkinson",
    "brain organoid",
    "tumor microenvironment",
    "hypoxia",
    "cell communication",
    "biological interface",
    "tissue engineering",
    # Biology / biomedical (for Cell Press coverage)
    "transcriptomics",
    "proteomics",
    "CRISPR",
    "immunotherapy",
    "cytokine",
    "immune",
    "neuron",
    "glia",
    "stem cell",
    "regeneration",
    "differentiation",
    "organ development",
    "disease model",
    "patient-derived",
    "xenograft",
    "neurovascular",
    "neuroimmune",
    "synapse",
    "circuit",
    "behavior",
]

# ============================================================================
# PERSONAL LEGACY TOPICS — always at least WATCHLIST
# ============================================================================
PERSONAL_LEGACY_TOPICS = [
    "gold nanocluster",
    "gold nanoclusters",
    "AuNC",
    "AuNCs",
    "BODIPY",
    "cyanine",
    "xanthene",
    "porous silicon",
    "pSiNP",
    "OECT",
    "organic electrochemical transistor",
    "extracellular vesicle",
    "EV sensing",
    "organoid",
    "organoid-on-chip",
    "BBB",
    "blood-brain barrier",
    "alpha-synuclein",
    "tau",
    "Alzheimer's disease",
    "ROS scavenging",
    "salidroside",
    "minocycline",
    "conductive hydrogel",
    "hydrogel electronics",
    "mechanobiology",
    "mechanotransduction",
    "two-photon",
    "photouncaging",
    "photocleavage",
    "photoactivation",
    "organoid",
    "brain organoid",
    "cerebral organoid",
    "neurodegeneration",
    "neuroinflammation",
    "synaptic",
    "astrocyte",
    "microglia",
    "brain clearance",
    "glymphatic",
    "cerebrospinal fluid",
    "CSF",
    "single-cell",
    "spatial transcriptomics",
    "liquid biopsy",
    "intercellular communication",
    # Meta-track
    "multi-agent system",
    "automated scientific discovery",
    "neurosymbolic",
    "literature-based discovery",
    "AI scientist",
]

# ============================================================================
# FUTURE TRAJECTORY TOPICS — expanded high-value concepts
# ============================================================================
FUTURE_TRAJECTORY_TOPICS = [
    # Nanophotonics / optical sensing
    "radiative Q-factor modulation",
    "Q-factor modulation",
    "Q-factor sensing",
    "bound state in the continuum",
    "BIC metasurface",
    "quasi-BIC",
    "photonic crystal biosensor",
    "metasurface biosensor",
    "optical resonance biosensing",
    "resonance linewidth modulation",
    "linewidth narrowing",
    "linewidth broadening",
    "refractometric biosensing",
    "optical confinement",
    "field localization",
    "local density of optical states",
    "LDOS",
    "Purcell effect",
    "strong coupling",
    "polariton",
    "polaritonic biosensing",
    "light-matter coupling",
    "dielectric metasurface",
    "lithium niobate",
    "thin-film lithium niobate",
    "nonlinear optics",
    "integrated nonlinear optics",
    "nonlinear photonics",
    "microresonator",
    "microring resonator",
    "structured light",
    "structured optical vortex",
    "optical vortex",
    "optical skyrmion",
    "vortex microcomb",
    "microcomb",
    "spin-orbit coupling",
    "topological charge",
    "nonlinear Cherenkov radiation",
    "Mie resonance",
    "electromagnetic hot spot",
    "near-field enhancement",
    "optical antenna",
    "nanophotonic biosensor",

    # Photochemistry / photon utilization
    "photon utilization",
    "energy routing",
    "excited-state routing",
    "excited-state energy redistribution",
    "nanostructure-enhanced photochemistry",
    "field-enhanced photochemistry",
    "optical field-mediated reaction",
    "local optical environment",
    "photochemical bond cleavage",
    "direct bond cleavage",
    "photo-SN1",
    "heterolysis",
    "oxygen-independent photochemistry",
    "ROS-independent photochemistry",
    "hypoxia-tolerant photoactivation",
    "photoredox activation",
    "photoinduced electron transfer",

    # Bioelectronics / mechanobiology
    "strain-resilient electrochemical interface",
    "stretchable electrochemical biointerface",
    "mechanically gated biointerface",
    "force-coupled electrochemical sensing",
    "mechano-electrochemical sensing",
    "mechanobiology-enabled sensing",
    "dynamic biointerface",
    "adaptive bioelectronics",
    "soft bioelectronic interface",
    "stretchable OECT",
    "organic electrochemical transistor array",
    "electrochemical organic light-emitting transistor",
    "electric double layer",
    "contact injection",
    "ion transport enhancer",
    "single-ion conductive elastomer",
    "polyelectrolyte elastomer",
    "high-dielectric plasticizer",
    "iontronics",
    "neuromorphic bioelectronics",
    "on-body edge computing",
    "mixed ionic-electronic conduction",

    # Organoid / EV / microphysiological sensing
    "organoid-derived extracellular vesicles",
    "organoid secretome",
    "EV photonic sensing",
    "EV bioelectronic sensing",
    "microfluidic EV sensing",
    "organoid-on-chip sensing",
    "microphysiological sensing",
    "non-destructive organoid monitoring",
    "secretome profiling",
    "extracellular vesicle diagnostics",
    "brain organoid",
    "cerebral organoid",
    "neural organoid",
    "iPSC-derived",
    "brain clearance",
    "glymphatic",
    "neuroinflammation",
    "single-cell",
    "spatial transcriptomics",
    "liquid biopsy",
    "intercellular communication",
    "cell-cell communication",
    # Active nanomaterial mediators
    "nanozyme",
    "single-atom nanozyme",
    "plasmonic catalyst",
    "plasmon-enhanced catalysis",
    "hot electron transfer",
    "photocatalytic nanomaterial",
    "nano-heterojunction",
    "photothermal mediator",
    "upconversion nanoparticle",
    "sonodynamic",
    "chemodynamic",
]

# ============================================================================
# Build combined concept dictionary with normalized keys
# ============================================================================
def _normalize_key(key: str) -> str:
    """Normalize a concept key: strip HTML, replace hyphens/dashes with spaces."""
    key = re.sub(r'<[^>]+>', '', key)
    key = re.sub(r'[-‐‑‒–—―]', ' ', key)
    return key.lower().strip()


CONCEPTS = {}
for c in TIER1_CORE_SIGNALS:
    CONCEPTS[_normalize_key(c)] = TIER1_WEIGHT
for c in TIER2_ADJACENT_SIGNALS:
    CONCEPTS[_normalize_key(c)] = TIER2_WEIGHT
for c in TIER2_META_SIGNALS:
    CONCEPTS[_normalize_key(c)] = TIER2_WEIGHT
for c in TIER3_WEAK_SIGNALS:
    CONCEPTS[_normalize_key(c)] = TIER3_WEIGHT

# Build lookup sets for classification logic (normalized)
PERSONAL_LEGACY_SET = set(_normalize_key(c) for c in PERSONAL_LEGACY_TOPICS)
FUTURE_TRAJECTORY_SET = set(_normalize_key(c) for c in FUTURE_TRAJECTORY_TOPICS)


def _prepare_text(text: str) -> str:
    # Strip HTML tags (Nature RSS feeds include <i>, <sub>, etc.)
    text = re.sub(r'<[^>]+>', '', text)
    # Normalize all dash/hyphen variants to spaces
    # U+002D hyphen-minus, U+2010 hyphen, U+2011 non-breaking hyphen,
    # U+2012 figure dash, U+2013 en dash, U+2014 em dash, U+2015 horizontal bar
    text = re.sub(r'[-‐‑‒–—―]', ' ', text)
    return text.lower().strip()


def get_concept_score(text: str) -> int:
    text_lower = _prepare_text(text)
    score = 0
    for phrase, weight in CONCEPTS.items():
        pattern = re.compile(re.escape(phrase), re.IGNORECASE)
        if pattern.search(text_lower):
            score += weight
    return score


def get_matched_concepts(text: str) -> list[tuple[str, int]]:
    text_lower = _prepare_text(text)
    matched = []
    for phrase, weight in CONCEPTS.items():
        pattern = re.compile(re.escape(phrase), re.IGNORECASE)
        if pattern.search(text_lower):
            matched.append((phrase, weight))
    return matched


def check_personal_legacy(text: str) -> list[str]:
    """Return list of Personal Legacy topics matched in the text."""
    text_lower = _prepare_text(text)
    matched = []
    for topic in PERSONAL_LEGACY_SET:
        pattern = re.compile(re.escape(topic), re.IGNORECASE)
        if pattern.search(text_lower):
            matched.append(topic)
    return matched


def check_future_trajectory(text: str) -> list[str]:
    """Return list of Future Trajectory topics matched in the text."""
    text_lower = _prepare_text(text)
    matched = []
    for topic in FUTURE_TRAJECTORY_SET:
        pattern = re.compile(re.escape(topic), re.IGNORECASE)
        if pattern.search(text_lower):
            matched.append(topic)
    return matched


def is_non_research(title: str, abstract: str = "") -> bool:
    """Check if a paper is non-research content (correction, editorial, etc.)."""
    combined = (title + " " + abstract).lower()
    non_research_patterns = [
        "correction to",
        "author correction",
        "editorial",
        "news item",
        "errata",
        "erratum",
        "issue publication information",
        "issue editorial masthead",
        "in other journals",
        "beyond moonshots",
        "team updates at",
        "when 'don't eat me'",
        "a robot that",
        "old scientific art",
        "aaas unveils",
        "an uncommon introduction",
        "take it slow",
        "getting home in the dark",
        "silent interference",
        "researching while chinese",
        "in science journals",
        "has jwst spotted",
        "closing the loop",
        "stop",
    ]
    for pattern in non_research_patterns:
        if pattern in combined:
            return True
    return False
