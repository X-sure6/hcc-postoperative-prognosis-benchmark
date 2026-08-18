import ast
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
MAIN=ROOT/"hcc_postoperative_prognosis_benchmark.py"

class NoFallbackStaticTests(unittest.TestCase):
    def test_tabpfn_constructor_is_explicit(self):
        tree=ast.parse(MAIN.read_text(encoding="utf-8"))
        calls=[]
        for node in ast.walk(tree):
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=="TabPFNClassifier":
                calls.append(node)
        self.assertGreaterEqual(len(calls),2)  # CV + temporal
        for call in calls:
            # No constructor may be called without keywords/**kwargs. The CV path
            # intentionally uses a predeclared strict kwargs dictionary.
            self.assertTrue(call.keywords)
        text=MAIN.read_text(encoding="utf-8")
        self.assertIn('"model_path": self.checkpoint',text)
        self.assertIn('"device": "cuda"',text)
        self.assertIn('"random_state": RANDOM_STATE',text)
        self.assertIn('model_path=self.checkpoint',text)

    def test_no_default_tabpfn_constructor(self):
        text=MAIN.read_text(encoding="utf-8")
        self.assertNotIn("TabPFNClassifier()",text)
        self.assertIn("cpu_fallback_allowed",text)
        self.assertIn("default_constructor_allowed",text)

if __name__ == "__main__":
    unittest.main()
