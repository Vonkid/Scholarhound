from psil.compress import _parse_frameworks


class FakeDB:
    def __init__(self):
        self.frameworks = []

    def insert_framework(self, **kwargs):
        self.frameworks.append(kwargs)


def test_parse_frameworks_skips_empty_heading_artifacts():
    text = """
Framework: Threshold-Triggered Catalytic Amplification

Framework name: Threshold-Triggered Catalytic Amplification (TTCA)
Description: A threshold condition releases catalytic amplification.
Core logic: Latent catalyst -> threshold crossing -> amplified output.
Worldview shift: Treats sensing as controlled threshold release.
Compression_score: 7/10
Novelty_score: 6/10
Predictive_power: 8/10
Falsifiability: 8/10
Actionability: 9/10
Transferability: 7/10
Taste_fit: 9/10
Suggested_experiment: Titrate catalyst concentration across the activation threshold.
"""
    db = FakeDB()

    frameworks = _parse_frameworks(text, db)

    assert [fw["framework_name"] for fw in frameworks] == [
        "Threshold-Triggered Catalytic Amplification (TTCA)"
    ]
    assert len(db.frameworks) == 1
    assert db.frameworks[0]["framework_name"] == "Threshold-Triggered Catalytic Amplification (TTCA)"
    assert db.frameworks[0]["compression_score"] == 7.0


def test_parse_frameworks_keeps_best_duplicate_by_substance():
    text = """
Framework name: Structural Energy Landscape Engineering
Compression_score: 1/10

Framework name: Structural Energy Landscape Engineering
Description: Redirects energy flow by modifying the structural energy landscape.
Compression_score: 8/10
Novelty_score: 7/10
"""
    db = FakeDB()

    frameworks = _parse_frameworks(text, db)

    assert len(frameworks) == 1
    assert db.frameworks[0]["description"].startswith("Redirects energy flow")
    assert db.frameworks[0]["compression_score"] == 8.0
