"""
Research Identity Engine — loads and updates the researcher's evolving profile.

Philosophy: identity is dynamic, not static. Each scan updates the identity
based on emerging concepts and trajectory shifts.
"""

from dataclasses import dataclass, field
from datetime import date
import json
from pathlib import Path

IDENTITY_PATH = Path.home() / ".psil" / "research_identity.json"


@dataclass
class ResearchIdentity:
    current_core: list[str] = field(default_factory=lambda: [
        "OECT biosensing",
        "small-molecule recognition",
        "RNA sensing",
        "molecular bioelectronics",
        "NIR photocleavage",
        "BODIPY photochemistry",
        "gold nanoclusters",
        "BBB delivery",
    ])

    emerging_directions: list[str] = field(default_factory=lambda: [
        "nanophotonics",
        "photon utilization",
        "optical confinement",
        "structure-field coupling",
        "local optical environment",
        "dielectric nanophotonics",
        "mechanobiology-enabled sensing",
        "stretchable biointerfaces",
        "EV sensing",
        "organoid sensing",
        "microphysiological systems",
        "adaptive bioelectronics",
    ])

    long_term_vision: list[str] = field(default_factory=lambda: [
        "molecular bioelectronics",
        "organoid + EV + sensing platforms",
        "Alzheimer's disease diagnostic-therapeutic systems",
        "adaptive biointerfaces",
        "nanophotonic field control",
        "intelligent sensing platforms",
        "mechanobiology-enabled diagnostics",
    ])

    trajectory_influence_topics: list[str] = field(default_factory=lambda: [
        # Nanophotonics
        "radiative Q-factor modulation",
        "Q-factor sensing",
        "Q-factor biosensing",
        "bound state in the continuum",
        "BIC metasurface",
        "quasi-BIC",
        "dielectric metasurface",
        "photonic crystal biosensor",
        "optical resonance biosensing",
        "resonance linewidth modulation",
        "optical confinement",
        "field localization",
        "Purcell effect",
        "local density of optical states",
        "LDOS",
        "strong coupling",
        "polariton",
        "polaritonic biosensing",
        "light-matter interaction",
        "Mie resonance",
        "optical antenna",
        "nanophotonic biosensor",
        # Photon utilization
        "photon utilization",
        "energy routing",
        "excited-state routing",
        "excited-state energy redistribution",
        "local optical environment",
        "structure-field coupling",
        "nanostructure-enhanced photochemistry",
        "field-enhanced photochemistry",
        "optical field-mediated reaction",
        "direct bond cleavage",
        "photo-SN1",
        "heterolysis",
        "oxygen-independent photochemistry",
        "ROS-independent photochemistry",
        "photoredox activation",
        "photoinduced electron transfer",
        # Mechanobiology
        "mechanobiology-enabled sensing",
        "mechanotransduction",
        "force-coupled sensing",
        "mechano-electrochemical sensing",
        "mechanically gated biointerface",
        "dynamic biointerface",
        "adaptive bioelectronics",
        "stretchable electrochemical interface",
        "strain-resilient electrochemical interface",
        "soft bioelectronics",
        # Organoid + EV
        "organoid-derived EV",
        "organoid secretome",
        "EV photonic sensing",
        "EV bioelectronic sensing",
        "organoid-on-chip sensing",
        "non-destructive organoid monitoring",
        "microphysiological sensing",
        "secretome profiling",
    ])

    last_updated: str = ""
    concept_momentum: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "current_core": self.current_core,
            "emerging_directions": self.emerging_directions,
            "long_term_vision": self.long_term_vision,
            "trajectory_influence_topics": self.trajectory_influence_topics,
            "last_updated": self.last_updated,
            "concept_momentum": self.concept_momentum,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ResearchIdentity":
        return cls(
            current_core=data.get("current_core", []),
            emerging_directions=data.get("emerging_directions", []),
            long_term_vision=data.get("long_term_vision", []),
            trajectory_influence_topics=data.get("trajectory_influence_topics", []),
            last_updated=data.get("last_updated", ""),
            concept_momentum=data.get("concept_momentum", {}),
        )

    def to_prompt_context(self) -> str:
        """Render identity as LLM prompt context."""
        lines = ["## RESEARCHER IDENTITY", ""]
        lines.append("### Current Core")
        for item in self.current_core:
            lines.append(f"- {item}")
        lines.append("")
        lines.append("### Emerging Directions")
        for item in self.emerging_directions:
            lines.append(f"- {item}")
        lines.append("")
        lines.append("### Long-Term Vision")
        for item in self.long_term_vision:
            lines.append(f"- {item}")
        lines.append("")
        lines.append("### High Trajectory Influence Topics")
        for item in self.trajectory_influence_topics[:20]:
            lines.append(f"- {item}")
        if len(self.trajectory_influence_topics) > 20:
            lines.append(f"- ... and {len(self.trajectory_influence_topics) - 20} more")
        return "\n".join(lines)

    def update_concept_momentum(self, concept: str, detected: bool):
        """Track concept appearance for momentum analysis."""
        if concept not in self.concept_momentum:
            self.concept_momentum[concept] = {
                "first_seen": str(date.today()),
                "appearances": 0,
                "status": "emerging",
            }
        if detected:
            self.concept_momentum[concept]["appearances"] += 1
            appearances = self.concept_momentum[concept]["appearances"]
            if appearances >= 3:
                self.concept_momentum[concept]["status"] = "gaining momentum"
            if appearances >= 5:
                self.concept_momentum[concept]["status"] = "established"

    def get_momentum_report(self) -> dict:
        """Generate concept momentum summary."""
        increasing = []
        stable = []
        decreasing = []
        for concept, data in self.concept_momentum.items():
            if data["status"] == "gaining momentum" or data["status"] == "established":
                increasing.append((concept, data))
            elif data["appearances"] >= 1:
                stable.append((concept, data))
            else:
                decreasing.append((concept, data))
        return {
            "increasing": sorted(increasing, key=lambda x: x[1]["appearances"], reverse=True),
            "stable": stable,
            "decreasing": decreasing,
        }


def load_identity() -> ResearchIdentity:
    """Load research identity from disk, or create default."""
    if IDENTITY_PATH.exists():
        with open(IDENTITY_PATH) as f:
            data = json.load(f)
        return ResearchIdentity.from_dict(data)
    return ResearchIdentity()


def save_identity(identity: ResearchIdentity):
    """Persist research identity to disk."""
    identity.last_updated = str(date.today())
    IDENTITY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(IDENTITY_PATH, "w") as f:
        json.dump(identity.to_dict(), f, indent=2)
