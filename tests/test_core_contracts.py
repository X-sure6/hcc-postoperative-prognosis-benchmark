import unittest
import pandas as pd
import numpy as np
import hcc_postoperative_prognosis_benchmark as core

class CoreContractTests(unittest.TestCase):
    def test_strict_binary_accepts_01_and_missing(self):
        s=core.sanitize_binary_y(pd.Series([0,1,np.nan,"0","1"]))
        self.assertEqual(set(s.dropna().astype(int).unique()),{0,1})

    def test_strict_binary_rejects_12(self):
        with self.assertRaises(ValueError):
            core.sanitize_binary_y(pd.Series([1,2,1,2]))

    def test_unsupported_endpoint_name_rejected(self):
        with self.assertRaises(ValueError):
            core.validate_endpoint_names(["OS12m","REC12m"],"test",allow_subset=True)

    def test_temporal_definition(self):
        core.validate_temporal_definition()
        self.assertEqual(str(core.TEMPORAL_DEV_START.date()),"2015-10-05")
        self.assertEqual(str(core.TEMPORAL_DEV_END.date()),"2019-06-30")
        self.assertEqual(str(core.TEMPORAL_GAP_START.date()),"2019-07-01")
        self.assertEqual(str(core.TEMPORAL_GAP_END.date()),"2019-09-30")
        self.assertEqual(str(core.TEMPORAL_VAL_START.date()),"2019-10-01")
        self.assertEqual(str(core.TEMPORAL_VAL_END.date()),"2020-12-25")

    def test_all_targets_use_os_and_ttr(self):
        self.assertEqual(len(core.ALL_TARGETS),10)
        self.assertEqual(
            core.ALL_TARGETS,
            ["OS12m","OS24m","OS36m","OS48m","OS60m",
             "TTR12m","TTR24m","TTR36m","TTR48m","TTR60m"],
        )
        self.assertEqual(core.TEMPORAL_TARGETS,["OS12m","OS24m","TTR12m","TTR24m"])

if __name__ == "__main__":
    unittest.main()
