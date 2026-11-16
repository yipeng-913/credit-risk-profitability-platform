"""Synthetic currency EUR, one-year, constant exposure; no regulatory PD claim."""

from dataclasses import dataclass, asdict, replace
import math


@dataclass(frozen=True)
class Assumptions:
    amount: float = 10000
    interest_rate: float = 0.12
    risk_proxy: float = 0.10
    lgd: float = 0.45
    funding_rate: float = 0.04
    operating_cost: float = 150

    def __post_init__(self):
        for k, v in asdict(self).items():
            if not math.isfinite(v) or v < 0:
                raise ValueError(f"{k} must be finite and nonnegative")
        if self.amount <= 0:
            raise ValueError("amount must be positive")
        for k in ["interest_rate", "risk_proxy", "lgd", "funding_rate"]:
            if getattr(self, k) > 1:
                raise ValueError(f"{k} must be between zero and one")


def calculate(a):
    income = a.amount * a.interest_rate
    funding = a.amount * a.funding_rate
    loss = a.amount * a.risk_proxy * a.lgd
    return dict(
        interest_income=income,
        funding_cost=funding,
        operating_cost=a.operating_cost,
        expected_loss_proxy=loss,
        profit=income - funding - a.operating_cost - loss,
    )


def scenarios(a):
    return {
        name: calculate(v)
        for name, v in {
            "Base": a,
            "Higher risk": replace(a, risk_proxy=min(1, a.risk_proxy * 1.5)),
            "Funding stress": replace(a, funding_rate=min(1, a.funding_rate + 0.02)),
        }.items()
    }
