#!/usr/bin/env python3
"""
Run optimization tests directly without pytest to check implementation.
"""

import sys

sys.path.insert(0, '.')

from tests.unit.core.test_optimization_comprehensive import (
    TestBayesianOptimizerComprehensive,
    TestCoarseToFineOptimizer,
    TestEdgeCasesAndErrorHandling,
    TestEnsembleOptimizerComprehensive,
    TestOptimizationDataPipeline,
    TestOptimizationResultBackwardCompatibility,
    TestOptuniaObjectiveBuilder,
    TestParameterSpaceDefinition,
    TestPlateauDetectorComprehensive,
)


def run_test_class(test_class):
    """Run all test methods in a test class."""
    print(f"\n{'='*60}")
    print(f"Running tests for {test_class.__name__}")
    print('='*60)

    instance = test_class()
    passed = 0
    failed = 0

    for attr_name in dir(instance):
        if attr_name.startswith('test_'):
            try:
                method = getattr(instance, attr_name)
                print(f"\n{attr_name}...", end=' ')
                method()
                print("PASSED")
                passed += 1
            except Exception as e:
                print(f"FAILED: {e}")
                failed += 1

    print(f"\nSummary: {passed} passed, {failed} failed")
    return passed, failed

if __name__ == "__main__":
    total_passed = 0
    total_failed = 0

    test_classes = [
        TestPlateauDetectorComprehensive,
        TestCoarseToFineOptimizer,
        TestBayesianOptimizerComprehensive,
        TestEnsembleOptimizerComprehensive,
        TestOptimizationDataPipeline,
        TestOptuniaObjectiveBuilder,
        TestParameterSpaceDefinition,
        TestOptimizationResultBackwardCompatibility,
        TestEdgeCasesAndErrorHandling,
    ]

    for test_class in test_classes:
        passed, failed = run_test_class(test_class)
        total_passed += passed
        total_failed += failed

    print(f"\n{'='*60}")
    print(f"TOTAL: {total_passed} passed, {total_failed} failed")
    print('='*60)
