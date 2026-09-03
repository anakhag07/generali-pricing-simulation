import numpy as np
import pytest

from objective.policy import (
    CallableFeatureMap,
    CubicFeatureMap,
    IdentityFeatureMap,
    QuadraticFeatureMap,
    QuarticFeatureMap,
    TotalDegreePolynomialFeatureMap,
)


def test_identity_feature_map_returns_input_features() -> None:
    x_array = np.array([[1.0, -2.0], [3.0, 4.0]], dtype=float)
    feature_map = IdentityFeatureMap()

    result = feature_map.transform(x_array)

    assert feature_map.output_dim(2) == 2
    np.testing.assert_allclose(result, x_array)


def test_quadratic_feature_map_order_and_dim() -> None:
    x_array = np.array([[2.0, 3.0], [-1.0, 4.0]], dtype=float)
    feature_map = QuadraticFeatureMap()

    result = feature_map.transform(x_array)
    expected = np.array(
        [
            [2.0, 3.0, 4.0, 6.0, 9.0],
            [-1.0, 4.0, 1.0, -4.0, 16.0],
        ],
        dtype=float,
    )

    assert feature_map.output_dim(2) == 5
    np.testing.assert_allclose(result, expected)


def test_total_degree_polynomial_map_is_nested_and_includes_all_interactions() -> None:
    x_array = np.array([[2.0, 3.0], [-1.0, 4.0]], dtype=float)
    degree_two = TotalDegreePolynomialFeatureMap(max_degree=2)
    degree_three = TotalDegreePolynomialFeatureMap(max_degree=3)

    quadratic = degree_two.transform(x_array)
    cubic = degree_three.transform(x_array)
    expected = np.array(
        [
            [2.0, 3.0, 4.0, 6.0, 9.0, 8.0, 12.0, 18.0, 27.0],
            [-1.0, 4.0, 1.0, -4.0, 16.0, -1.0, 4.0, -16.0, 64.0],
        ],
        dtype=float,
    )

    assert degree_two.output_dim(2) == 5
    assert degree_three.output_dim(2) == 9
    np.testing.assert_allclose(cubic[:, : quadratic.shape[1]], quadratic)
    np.testing.assert_allclose(cubic, expected)


def test_total_degree_polynomial_map_has_expected_capacity_for_19_inputs() -> None:
    assert [
        1 + TotalDegreePolynomialFeatureMap(max_degree=degree).output_dim(19)
        for degree in range(4)
    ] == [1, 20, 210, 1540]


def test_quadratic_feature_map_without_interactions() -> None:
    x_array = np.array([[2.0, 3.0]], dtype=float)
    feature_map = QuadraticFeatureMap(include_interactions=False)

    result = feature_map.transform(x_array)

    assert feature_map.output_dim(2) == 4
    np.testing.assert_allclose(result, np.array([[2.0, 3.0, 4.0, 9.0]], dtype=float))


def test_cubic_feature_map_order_and_dim() -> None:
    x_array = np.array([[2.0, 3.0], [-1.0, 4.0]], dtype=float)
    feature_map = CubicFeatureMap()

    result = feature_map.transform(x_array)
    expected = np.array(
        [
            [2.0, 3.0, 8.0, 12.0, 18.0, 27.0],
            [-1.0, 4.0, -1.0, 4.0, -16.0, 64.0],
        ],
        dtype=float,
    )

    assert feature_map.output_dim(2) == 6
    np.testing.assert_allclose(result, expected)


def test_cubic_feature_map_without_interactions() -> None:
    x_array = np.array([[2.0, 3.0]], dtype=float)
    feature_map = CubicFeatureMap(include_interactions=False)

    result = feature_map.transform(x_array)

    assert feature_map.output_dim(2) == 4
    np.testing.assert_allclose(result, np.array([[2.0, 3.0, 8.0, 27.0]], dtype=float))


def test_quartic_feature_map_order_and_dim() -> None:
    x_array = np.array([[2.0, 3.0], [-1.0, 4.0]], dtype=float)
    feature_map = QuarticFeatureMap()

    result = feature_map.transform(x_array)
    expected = np.array(
        [
            [2.0, 3.0, 16.0, 24.0, 36.0, 54.0, 81.0],
            [-1.0, 4.0, 1.0, -4.0, 16.0, -64.0, 256.0],
        ],
        dtype=float,
    )

    assert feature_map.output_dim(2) == 7
    np.testing.assert_allclose(result, expected)


def test_quartic_feature_map_without_interactions() -> None:
    x_array = np.array([[2.0, 3.0]], dtype=float)
    feature_map = QuarticFeatureMap(include_interactions=False)

    result = feature_map.transform(x_array)

    assert feature_map.output_dim(2) == 4
    np.testing.assert_allclose(result, np.array([[2.0, 3.0, 16.0, 81.0]], dtype=float))


def test_callable_feature_map_accepts_valid_lambda() -> None:
    feature_map = CallableFeatureMap(
        lambda x: np.column_stack([x[:, 0], x[:, 1] ** 2]),
        feature_dim=2,
        name="x0_x1_squared",
    )
    x_array = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float)

    result = feature_map.transform(x_array)

    assert feature_map.output_dim(2) == 2
    np.testing.assert_allclose(result, np.array([[1.0, 4.0], [3.0, 16.0]], dtype=float))


@pytest.mark.parametrize(
    "feature_map",
    [
        IdentityFeatureMap(),
        QuadraticFeatureMap(),
        CubicFeatureMap(),
        QuarticFeatureMap(),
        CallableFeatureMap(lambda x: x, feature_dim=2),
    ],
)
def test_feature_maps_reject_non_2d_input(feature_map) -> None:
    with pytest.raises(ValueError, match="2D"):
        feature_map.transform(np.zeros(2))


def test_callable_feature_map_rejects_1d_output() -> None:
    feature_map = CallableFeatureMap(lambda x: x[:, 0], feature_dim=1)
    with pytest.raises(ValueError, match="2D"):
        feature_map.transform(np.zeros((2, 2)))


def test_callable_feature_map_rejects_wrong_row_count() -> None:
    feature_map = CallableFeatureMap(lambda x: np.zeros((x.shape[0] + 1, 1)), feature_dim=1)
    with pytest.raises(ValueError, match="number of samples"):
        feature_map.transform(np.zeros((2, 2)))


def test_callable_feature_map_rejects_wrong_feature_dim() -> None:
    feature_map = CallableFeatureMap(lambda x: x, feature_dim=3)
    with pytest.raises(ValueError, match="expected 3"):
        feature_map.transform(np.zeros((2, 2)))


def test_callable_feature_map_rejects_non_finite_output() -> None:
    feature_map = CallableFeatureMap(lambda x: np.full((x.shape[0], 1), np.nan), feature_dim=1)
    with pytest.raises(ValueError, match="finite"):
        feature_map.transform(np.zeros((2, 2)))
