"""
Final test coverage for DataHandler focusing on API key loading edge cases.
"""

import logging
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from core.data_handler import DataHandler


class TestDataHandlerFinalCoverage:
    """Tests to achieve 100% coverage for DataHandler."""

    @pytest.fixture
    def temp_cache_dir(self):
        """Create a temporary cache directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    def test_alpha_vantage_api_key_from_secrets_file_error(self, temp_cache_dir, caplog):
        """Test Alpha Vantage API key loading when secrets file exists but fails to load."""
        config = {
            'cache_dir': temp_cache_dir,
            'providers': ['alpha_vantage'],
            'alpha_vantage': {}  # Empty config to trigger secrets loading
        }

        # Create a config directory with invalid secrets file
        config_dir = Path(temp_cache_dir) / "config"
        config_dir.mkdir()
        secrets_path = config_dir / "secrets.yaml"
        secrets_path.write_text("invalid: yaml: content:")  # Invalid YAML

        # Patch Path to use our temp directory
        with patch.object(Path, '__file__', property(lambda self: str(temp_cache_dir / "data_handler.py"))):
            with patch('os.getenv', return_value=None):  # No env var
                with caplog.at_level(logging.DEBUG):
                    handler = DataHandler(config)

                    # Should log the error
                    assert "Could not load API key from secrets.yaml" in caplog.text
                    # Should not have alpha_vantage adapter
                    assert 'alpha_vantage' not in handler.adapters

    def test_alpha_vantage_no_api_key_warning(self, temp_cache_dir, caplog):
        """Test warning when no Alpha Vantage API key is available."""
        config = {
            'cache_dir': temp_cache_dir,
            'providers': ['alpha_vantage']
        }

        # No API key in env or config
        with patch('os.getenv', return_value=None):
            with patch('pathlib.Path.exists', return_value=False):  # No secrets file
                with caplog.at_level(logging.WARNING):
                    handler = DataHandler(config)

                    # Should log warning
                    assert "No Alpha Vantage API key provided" in caplog.text
                    # Should not have alpha_vantage adapter
                    assert 'alpha_vantage' not in handler.adapters

    def test_alpha_vantage_api_key_yaml_parsing_error(self, temp_cache_dir, caplog):
        """Test Alpha Vantage API key loading with YAML parsing error."""
        config = {
            'cache_dir': temp_cache_dir,
            'providers': ['alpha_vantage']
        }

        # Create a secrets file that will cause yaml.safe_load to fail
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            # Write content that causes YAML parsing error
            f.write("alpha_vantage:\n  api_key: !!python/object/apply:os.system ['echo hacked']")
            secrets_path = f.name

        try:
            # Mock the path operations
            with patch('pathlib.Path.exists', return_value=True):
                with patch('builtins.open', return_value=open(secrets_path)):
                    with patch('os.getenv', return_value=None):
                        with caplog.at_level(logging.DEBUG):
                            handler = DataHandler(config)

                            # Should handle the error gracefully
                            assert 'alpha_vantage' not in handler.adapters
                            # Check that some error was logged
                            assert any("Could not load API key" in msg for msg in caplog.messages)
        finally:
            Path(secrets_path).unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
