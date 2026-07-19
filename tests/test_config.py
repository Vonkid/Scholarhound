import os
import tempfile
from psil.config import load_config


def test_load_config_resolves_env_vars():
    yaml_content = """
llm:
  api_key: ${TEST_API_KEY}
  model: deepseek-chat
vault_path: /tmp/vault
journals:
  - name: Test Journal
    issn: 1234-5678
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        tmp_path = f.name

    os.environ["TEST_API_KEY"] = "sk-test-123"
    config = load_config(tmp_path)
    assert config["llm"]["api_key"] == "sk-test-123"
    assert config["journals"][0]["issn"] == "1234-5678"
    os.unlink(tmp_path)


def test_load_config_defaults():
    config = load_config()
    assert "journals" in config
    assert "llm" in config
    assert "vault_path" in config
    assert len(config["journals"]) >= 18
