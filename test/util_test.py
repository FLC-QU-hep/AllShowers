import os
import shutil
import unittest

from allshowers import util


class TestDataUtil(unittest.TestCase):
    def test_setup_result_path(self):
        path1 = util.setup_result_path(
            "test", "conf/transformer.yaml", fast_dev_run=True
        )
        path2 = util.setup_result_path(
            "test", "conf/transformer.yaml", fast_dev_run=False
        )
        path3 = util.setup_result_path(
            "test", "conf/transformer.yaml", fast_dev_run=False
        )
        self.assertTrue(path1 != path2)
        self.assertTrue(path2 != path3)
        self.assertTrue(path1 != path3)
        self.assertTrue(path1.endswith("results/test"))
        self.assertTrue("/results/" in path2)
        self.assertTrue("/results/" in path3)
        self.assertTrue("test" in path2)
        self.assertTrue("test" in path3)
        self.assertTrue(os.path.isdir(path1))
        self.assertTrue(os.path.isdir(path2))
        self.assertTrue(os.path.isdir(path3))
        shutil.rmtree(path1)
        shutil.rmtree(path2)
        shutil.rmtree(path3)


if __name__ == "__main__":
    unittest.main()
