import pytest

from greto.fast_features.feature_level import build_feature_specs, feature_values


def test_feature_specs_are_available_and_cover_expected_features():
    specs = build_feature_specs()

    assert len(specs) == 240
    assert specs[0].name == "rsg_sum_1"
    assert specs[-1].name == "-log_escape_probability"
    assert "rsg_wmean_2v" in {spec.name for spec in specs}


def test_feature_values_preserves_public_interface():
    names = feature_values(None, None, name_mode=True)
    dependencies = feature_values(None, None, dependency_mode=True)

    assert len(names) == 240
    assert len(dependencies) == 240
    assert names[0] == "rsg_sum_1"
    assert names[-1] == "-log_escape_probability"
    assert "rsg_sum_1" in dependencies
