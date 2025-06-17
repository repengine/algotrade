"""
Test coverage for DataHandler Alpha Vantage API key loading paths.
"""

import logging
import shutil
import tempfile
from pathlib import Path
from unittest.mock import mock_open, patch

import pytest
from core.data_handler import DataHandler


class TestDataHandlerAlphaVantage:
    """Tests for Alpha Vantage API key loading edge cases."""

    @pytest.fixture
    def temp_cache_dir(self):
        """Create a temporary cache directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    def test_alpha_vantage_yaml_load_exception(self, temp_cache_dir, caplog):
        """Test exception during YAML loading of API key."""
        config = {
            'cache_dir': temp_cache_dir,
            'providers': ['alpha_vantage'],
            'alpha_vantage': {}  # Empty config to force secrets loading
        }

        # Mock the file path to exist
        Path("/fake/config/secrets.yaml")

        # Mock yaml to raise exception
        with patch('os.getenv', return_value=None):
            with patch('pathlib.Path.exists', return_value=True):
                with patch('builtins.open', mock_open(read_data="data_providers:\n  alphavantage:\n    api_key: test")):
                    # Mock yaml.safe_load to raise exception
                    import sys
                    yaml_mock = type(sys)('yaml')
                    yaml_mock.safe_load = lambda x: (_ for _ in ()).throw(Exception("YAML parse error"))

                    with patch.dict('sys.modules', {'yaml': yaml_mock}):
                        with caplog.at_level(logging.DEBUG):
                            handler = DataHandler(config)

                            # Should log the debug message
                            assert any("Could not load API key from secrets.yaml" in record.message
                                     for record in caplog.records)
                            # Should not have alpha_vantage adapter
                            assert 'alpha_vantage' not in handler.adapters

    def test_alpha_vantage_no_api_key_anywhere(self, temp_cache_dir, caplog):
        """Test warning when no API key is found anywhere."""
        config = {
            'cache_dir': temp_cache_dir,
            'providers': ['alpha_vantage'],
            'alpha_vantage': {}  # Empty config
        }

        # No API key in env, config, or secrets
        with patch('os.getenv', return_value=None):
            with patch('pathlib.Path.exists', return_value=False):
                with caplog.at_level(logging.WARNING):
                    handler = DataHandler(config)

                    # Should log warning
                    assert any("No Alpha Vantage API key provided" in record.message
                             for record in caplog.records)
                    # Should not have alpha_vantage adapter
                    assert 'alpha_vantage' not in handler.adapters

    def test_alpha_vantage_file_read_exception(self, temp_cache_dir, caplog):
        """Test exception when reading secrets file."""
        config = {
            'cache_dir': temp_cache_dir,
            'providers': ['alpha_vantage'],
            'alpha_vantage': {}
        }

        with patch('os.getenv', return_value=None):
            with patch('pathlib.Path.exists', return_value=True):
                # Mock open to raise exception
                with patch('builtins.open', side_effect=OSError("Cannot read file")):
                    with caplog.at_level(logging.DEBUG):
                        handler = DataHandler(config)

                        # Should handle error gracefully
                        assert 'alpha_vantage' not in handler.adapters
                        # Should have logged something about the error
                        assert len(caplog.records) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
