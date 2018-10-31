import numpy as np
import unittest

from pylift.eval import _get_counts, _get_tc_counts, _get_no_sure_thing_counts, _get_no_sleeping_dog_counts, _get_overfit_counts, get_scores

treatment = np.array([1,1,0,1,0])
outcome = np.array([1,1,1,0,0])
p = np.array([0.75,0.5,0.75,0.75,0.5])
prediction = np.array([1,0.9,0.8,0.5,0.6])
Nt1o1, Nt0o1, Nt1o0, Nt0o0 = _get_counts(treatment, outcome, p)
scores = get_scores(treatment, outcome, prediction, p)

simple_counts_example = (4/3, 2, 2/3, 1)

treatment_simple = [1,0,1,0]
outcome_simple = [1,0,0,1]
prediction_simple = [1, 0.5, 0.25, 0]
p_simple = [0.5, 0.5, 0.5, 0.5]
scores_simple = get_scores(treatment_simple, outcome_simple, prediction_simple, p_simple)

class CountsAndScores(unittest.TestCase):
    """Test all counting and scoring functions.

    """

    # The following tests were calculated with the package and each step was verified.
    def test_Nt1o1(self):
        self.assertAlmostEqual(Nt1o1, 5/3)

    def test_Nt0o1(self):
        self.assertAlmostEqual(Nt0o1, 2)

    def test_Nt1o0(self):
        self.assertAlmostEqual(Nt1o0, 2/3)

    def test_Nt0o0(self):
        self.assertAlmostEqual(Nt0o0, 1)

    def test_aqini(self):
        self.assertAlmostEqual(scores['Q_aqini'], 0.21279762)

    def test_qini(self):
        self.assertAlmostEqual(scores['Q_qini'], 0.18601190)

    def test_cgains(self):
        self.assertAlmostEqual(scores['Q_cgains'], 0.11737351)

    def test_practical_max(self):
        self.assertAlmostEqual(scores['Q_practical_max'], 0.24943311)

    def test_max(self):
        self.assertAlmostEqual(scores['Q_max'], 0.39527530)

    # The following tests were calculated by hand.
    def test_counts(self):
        self.assertTrue(np.allclose(_get_counts(treatment, outcome, p), (5/3, 2, 2/3, 1)))

    # The following were also calculated by hand, but are redundant to earlier
    # tests. These examples are a nice in that they are a bit simpler, though.
    def test_cgains(self):
        self.assertAlmostEqual(scores_simple['Q_cgains'], 0.28125, msg="Incorrect cumulative gains score.")

    def test_aqini(self):
        self.assertAlmostEqual(scores_simple['Q_aqini'], 0.375, msg="Incorrect cumulative gains score.")

    def test_qini(self):
        self.assertAlmostEqual(scores_simple['Q_qini'], 0.375, msg="Incorrect cumulative gains score.")

if __name__ == '__main__':
    unittest.main()
