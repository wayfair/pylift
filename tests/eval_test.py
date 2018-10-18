import numpy as np
import unittest

from pylift.eval import _get_counts, _get_tc_counts, _get_no_sure_thing_counts, _get_no_sleeping_dog_counts, _get_overfit_counts, get_scores

class TestScores(unittest.TestCase):
    """Test `eval.get_scores`.

    """
    def test_cgains(self):
        self.assertAlmostEqual(get_scores([1,0,1,0], [1,0,0,1], [1, 0.5, 0.25, 0], [0.5, 0.5, 0.5, 0.5])['Q_cgains'], 0.28125, msg="Incorrect cumulative gains score.")

    def test_aqini(self):
        self.assertAlmostEqual(get_scores([1,0,1,0], [1,0,0,1], [1, 0.5, 0.25, 0], [0.5, 0.5, 0.5, 0.5])['Q_aqini'], 0.375, msg="Incorrect cumulative gains score.")

    def test_qini(self):
        self.assertAlmostEqual(get_scores([1,0,1,0], [1,0,0,1], [1, 0.5, 0.25, 0], [0.5, 0.5, 0.5, 0.5])['Q_qini'], 0.375, msg="Incorrect cumulative gains score.")

treatment = np.array([1,1,0,1,0])
outcome = np.array([1,1,1,0,0])
p = np.array([0.75,0.75,0.75,0.75,0.5])
Nt1o1, Nt0o1, Nt1o0, Nt0o0 = _get_counts(treatment, outcome, p)

class Counts(unittest.TestCase):
    """Test all counting functions.

    """

    def test_counts(self):
        self.assertTrue(np.allclose(_get_counts(treatment, outcome, p), (4/3, 2, 2/3, 1)))

    def test_nostcounts(self):
        self.assertTrue(np.allclose(_get_no_sure_thing_counts(Nt1o1, Nt0o1, Nt1o0, Nt0o0), (10/3, 10/3, 0, -5/3)))

    def test_nosdcounts(self):
        self.assertTrue(np.allclose(_get_no_sleeping_dog_counts(Nt1o1, Nt0o1, Nt1o0, Nt0o0), (0, 0, 10/3, 5/3)))

    def test_ofcounts(self):
        self.assertTrue(np.allclose(_get_overfit_counts(Nt1o1, Nt0o1, Nt1o0, Nt0o0), (4/3, 2, 0, 5/3)))

if __name__ == '__main__':
    unittest.main()
