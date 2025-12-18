"""Tests for the utils module."""

import contextlib
import pytest
from pygnd import utils


class TestTqdmJoblib:
    """Tests for tqdm_joblib context manager."""

    def test_context_manager_protocol(self):
        """Test that tqdm_joblib follows context manager protocol."""
        # Create a mock tqdm object with required methods
        class MockTqdm:
            def __init__(self):
                self.closed = False
                self.updates = 0

            def update(self, n=1):
                self.updates += n

            def close(self):
                self.closed = True

        mock_tqdm = MockTqdm()

        # Test that it can be used as a context manager
        with utils.tqdm_joblib(mock_tqdm) as tqdm_obj:
            assert tqdm_obj is mock_tqdm
            assert not mock_tqdm.closed

        # After exiting, tqdm should be closed
        assert mock_tqdm.closed

    def test_is_context_manager(self):
        """Test that tqdm_joblib returns a context manager."""
        class DummyTqdm:
            def close(self):
                pass

        result = utils.tqdm_joblib(DummyTqdm())
        assert hasattr(result, "__enter__")
        assert hasattr(result, "__exit__")
