import copy

import pytest

from scripts.ood_po_protocol import submission_eligibility
from scripts.submit_v21 import validate_report, score_value


def valid_report():
    integrity = dict.fromkeys(['all_5016_species', 'new_assessment_excludes_old_v20_heldouts',
        'twenty_km_training_buffer', 'original_v20_bundle_verified', 'unchanged_v20_csv_exact',
        'production_policy_has_nonzero_po', 'policies_unchanged_since_freeze',
        'production_csv_unchanged_since_freeze', 'competition_only', 'test_labels_unused', 'registered_runtime'], True)
    policy = {'alpha': .1, 'gate': 'pa_distance', 'k': 20, 'species_gate': 'all', 'row_scaling': 'none'}
    base = {'mean_difference': .01, 'ci95': [.001, .02], 'spatial_blocks': 28, 'iterations': 500, 'seed': 20250921}
    zero = {'mean_difference': .005, 'ci95': [-.001, .02], 'spatial_blocks': 28, 'iterations': 500, 'seed': 20250921}
    report = {'experiment': 'v21_competition_only_ood_po_expert', 'status': 'complete', 'source_commit': 'tested',
              'external_data_or_weights': False, 'test_labels_used': False, 'post_calibration_finetuning': False,
              'original_v20_retrained_to_recover_outputs': False, 'notebook_tests_before_and_after_passed': True,
              'total_pipeline_hours': 2., 'registered_max_total_hours': 10.5, 'frozen_v20_source_version': 20,
              'split': {'assessment_original_v20_training_only': True, 'minimum_evaluation_distance_km': 20.1,
                        'partition_counts': {'assessment': 17163}},
              'submission': {'rows': 14784, 'species_vocabulary': 5016}, 'integrity': integrity,
              'selected_policies': {'evaluation': {'po': policy}, 'deployment': {'po': policy}},
              'assessment': {'used_for_selection': False, 'versus_frozen_v20': base, 'versus_zero_po': zero,
                             'metrics': {name: {'sample_f1': score, 'species': 5016, 'surveys': 17163}
                                         for name, score in [('po', .31), ('zero_po', .305), ('frozen_v20_recipe', .30)]}}}
    report['submission_gate'] = submission_eligibility(policy, base, zero, integrity)
    return report


def test_only_a_consistent_complete_success_can_be_submitted():
    report = valid_report()
    validate_report(report, 'tested')
    with pytest.raises(ValueError, match='Source commit'):
        validate_report(report, 'different')
    for field, value in [('notebook_tests_before_and_after_passed', False),
                         ('total_pipeline_hours', 10.6), ('test_labels_used', True),
                         ('external_data_or_weights', True)]:
        changed = copy.deepcopy(report)
        changed[field] = value
        with pytest.raises(ValueError):
            validate_report(changed, 'tested')


def test_gate_cannot_be_forged_by_eligibility_flag_or_inconsistent_scores():
    report = valid_report()
    report['assessment']['versus_frozen_v20']['ci95'][0] = -.001
    with pytest.raises(ValueError, match='recomputed submission gate'):
        validate_report(report, 'tested')
    report = valid_report()
    report['assessment']['metrics']['po']['sample_f1'] = .5
    with pytest.raises(ValueError, match='gains disagree'):
        validate_report(report, 'tested')
    report = valid_report()
    del report['integrity']['twenty_km_training_buffer']
    with pytest.raises(ValueError, match='integrity checks'):
        validate_report(report, 'tested')


def test_missing_scores_are_not_reported_as_zero():
    assert score_value(None) is None
    assert score_value('') is None
    assert score_value('NaN') is None
    assert score_value('0.19360') == .19360
    assert score_value(0) == 0
