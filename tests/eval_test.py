import unittest
from pylift.eval import get_scores

class TestScores(unittest.TestCase):

    def test_cgains(self):
        self.assertAlmostEqual(get_scores([1,0,1,0], [1,0,0,1], [1, 0.5, 0.25, 0], [0.5, 0.5, 0.5, 0.5])['Q_cgains'], 0.28125, msg="Incorrect cumulative gains score.")

    def test_aqini(self):
        self.assertAlmostEqual(get_scores([1,0,1,0], [1,0,0,1], [1, 0.5, 0.25, 0], [0.5, 0.5, 0.5, 0.5])['Q_aqini'], 0.375, msg="Incorrect cumulative gains score.")

    def test_qini(self):
        self.assertAlmostEqual(get_scores([1,0,1,0], [1,0,0,1], [1, 0.5, 0.25, 0], [0.5, 0.5, 0.5, 0.5])['Q_qini'], 0.375, msg="Incorrect cumulative gains score.")

if __name__ == '__main__':
    unittest.main()
