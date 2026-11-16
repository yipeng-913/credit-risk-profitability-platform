import pytest
from src.finance.engine import Assumptions, calculate, scenarios


def test_known_cashflows():
    result = calculate(Assumptions())
    assert result == dict(
        interest_income=1200, funding_cost=400, operating_cost=150, expected_loss_proxy=450, profit=200
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"amount": 0},
        {"amount": -1},
        {"lgd": 1.1},
        {"risk_proxy": -0.1},
        {"funding_rate": float("nan")},
        {"operating_cost": float("inf")},
    ],
)
def test_invalid_inputs(kwargs):
    with pytest.raises(ValueError):
        Assumptions(**kwargs)


def test_scenario_direction_and_boundary():
    values = scenarios(Assumptions())
    assert values["Higher risk"]["profit"] < values["Base"]["profit"]
    assert values["Funding stress"]["profit"] < values["Base"]["profit"]
    assert calculate(Assumptions(risk_proxy=0))["expected_loss_proxy"] == 0
    assert calculate(Assumptions(risk_proxy=1, lgd=1))["expected_loss_proxy"] == 10000
