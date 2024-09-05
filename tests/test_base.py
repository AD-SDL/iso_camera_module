"""Base module tests."""

import unittest


class TestModule_Base(unittest.TestCase):
    """Base test class for this module."""

    pass


class TestImports(TestModule_Base):
    """Test the imports of the module are working correctly"""

    def test_driver_import(self):
        """Test the driver and rest node imports"""
        import rpl_tag_engine_driver
        import rpl_tag_engine_rest_node

        assert rpl_tag_engine_driver
        assert rpl_tag_engine_rest_node


if __name__ == "__main__":
    unittest.main()
