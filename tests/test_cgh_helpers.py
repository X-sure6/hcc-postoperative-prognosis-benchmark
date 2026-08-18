import unittest
import numpy as np
import cgh_supplementary_analyses as cgh

class CGHHelperTests(unittest.TestCase):
    def test_paired_delta_positive_when_second_scores_are_better(self):
        y=np.array([0,0,0,1,1,1])
        a=np.array([.4,.45,.5,.5,.55,.6])
        b=np.array([.1,.2,.3,.7,.8,.9])
        point,ci=cgh.paired_delta(y,a,b,20,42)
        self.assertGreater(point[0],0)
        self.assertLess(point[2],0)
        self.assertEqual(len(ci),3)

    def test_safe_metrics(self):
        y=np.array([0,0,1,1]); p=np.array([.1,.2,.8,.9])
        self.assertAlmostEqual(cgh.safe_auc(y,p),1.0)
        self.assertAlmostEqual(cgh.safe_ap(y,p),1.0)

    def test_km_and_cif_helpers(self):
        self.assertGreaterEqual(cgh.km_survival_at([1,2,3],[1,0,1],2),0.0)
        cif=cgh.cif_at([1,2,3],[1,2,0],3)
        self.assertGreaterEqual(cif,0.0)
        self.assertLessEqual(cif,1.0)

if __name__ == "__main__":
    unittest.main()
