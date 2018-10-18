import numpy as np
import unittest

from pylift.methods.derivatives import TransformedOutcome

treatment = np.array([1,1,0,1,0])
outcome = np.array([1,1,1,0,0])
p = np.array([0.75,0.75,0.75,0.75,0.5])
transformed_outcome = TransformedOutcome._transform_func(treatment, outcome, p)
treatment_new, outcome_new, p_new = TransformedOutcome._untransform_func(transformed_outcome)

class TransformedOutcomeTransformation(unittest.TestCase):
    """Test transformations and inverse transformations for data recovery.

    """

    def test_untransform_treatment(self):
        self.assertTrue(np.isclose(treatment, treatment_new).all())

    def test_untransform_outcome(self):
        self.assertTrue(np.isclose(outcome, outcome_new).all())

    def test_untransform_p(self):
        self.assertTrue(np.isclose(p, p_new).all())

if __name__ == '__main__':
    unittest.main()
