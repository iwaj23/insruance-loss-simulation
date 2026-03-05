# Insurance Loss Simulation (Actuarial Modeling)

This project simulates annual aggregate insurance losses for a generic portfolio using a frequency–severity model and Monte Carlo simulation.

## Model
- Claim frequency: Poisson(lambda)
- Claim severity:
  - Lognormal calibrated to a target mean claim size
  - Gamma calibrated to match the same mean and approximate variability
- Aggregate loss: sum of severities for all claims in a simulated year

## Outputs
- Risk metrics: mean, standard deviation, and 90th/95th/99th percentiles
- Aggregate loss distribution comparison (Lognormal vs Gamma)
- Loss exceedance curve (probability aggregate loss exceeds a threshold)

## Run
pip install -r requirements.txt
python loss_simulation.py