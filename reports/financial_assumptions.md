# Illustrative financial assumptions

All EUR examples and all financial parameters are synthetic, not observed Home Credit pricing or profit. No currency conversion of competition amounts is performed.

PD: probability of a defined default over a specified horizon. LGD: fraction of exposure lost conditional on default. EAD: outstanding exposure at default. The familiar ECL relationship is PD × LGD × EAD; this project substitutes an explicitly labeled **scenario risk proxy** because TARGET is repayment difficulty without a validated regulatory default horizon.

Assume one year, constant exposure equal to loan principal, simple interest on full principal and no taxes. Profit = principal × annual rate − principal × annual funding rate − annual operating cost − principal × scenario risk proxy × LGD. Foregone interest upon default is not modeled; default losses and gross interest are simplified independently.

Arithmetic reference: principal EUR 10,000; annual interest 12%; funding 4%; operations EUR 150; proxy 10%; LGD 45%. Income 1,200; funding 400; loss proxy 450; profit 200. At proxy 15%, profit is −25. At funding 6%, base-risk profit is 0. These deterministic cases are covered by tests. Exported scenarios use the actual mean model score instead of the 10% reference, and can therefore differ.

Dashboard inputs: positive finite amount; finite nonnegative operating cost; rates, loss severity and risk proxy in [0,1]. Higher-risk scenario multiplies the editable proxy by 1.5 (capped at 1); funding stress adds 0.02 (capped at 1).

Limitations: unknown default timing and horizon; no amortization or exposure trajectory; no recovery timing/cost; no detailed servicing, origination, capital or tax costs; no validated LGD; no realized-profit labels; no interest cessation on default. Results support sensitivity discussion only, not product pricing or regulatory reporting.
