import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import poisson, lognorm, gamma


#____________________________
#1) Assumptions
#____________________________
# Remove seed for random unreproducible results
SEED = 42
NUM_SIMS = 50_000
LAMBDA_CLAIMS = 20

POLICY_LIMIT = 20000

TARGET_MEAN_SEVERITY = 5000   #average claim size in dollars
SIGMA = 0.8                   #lognormal shape parameter (controls tail heaviness)

#If X ~ Lognormal(mu,sigma), then E[X] = exp(mu + 0.5*sigma^2)
MU = np.log(TARGET_MEAN_SEVERITY) - 0.5 * SIGMA**2
#Compare to Gamma severity with the same mean and approx same CV as Lognormal(SIGMA)
cv2 = np.exp(SIGMA**2) - 1
gamma_shape = 1/ cv2                             #K
gamma_scale = TARGET_MEAN_SEVERITY / gamma_shape #theta

print("- Model Assumptions -")
print(f"Claim Frequency: Poisson (lambda = {LAMBDA_CLAIMS})")
print(f"Claim Severity: Lognormal (mean = {TARGET_MEAN_SEVERITY:,.0f}, sigma = {SIGMA})")
print(f"Policy Limit: {POLICY_LIMIT:,.0f}")
print(f"Simulations:    {NUM_SIMS:,}\n")

#____________________________
# 2) Simulation
#____________________________

rng = np.random.default_rng(SEED)
# Replace the above line with the one below for no seed. (Every run of the program will produce different results)
# rng = np.random.default_rng()

# SIMULATION FUNCTION

# Simulates all claims at once
# Aggregates them efficiently
def simulate_aggregate_losses_fast(severity_dist:str):
    claim_counts = poisson(mu=LAMBDA_CLAIMS).rvs(size=NUM_SIMS, random_state=rng)
    total_claims = claim_counts.sum()
    if severity_dist == "lognormal":
        severities = lognorm(s=SIGMA, scale =np.exp(MU)).rvs(size=total_claims,random_state=rng)
    elif severity_dist =="gamma":
        severities = gamma(a=gamma_shape, scale=gamma_scale).rvs(size=total_claims, random_state=rng)
    else:
        raise ValueError("severity_dist must be 'lognormal' or 'gamma'")
    severities = np.minimum(severities, POLICY_LIMIT)
    splits = np.cumsum(claim_counts)[:-1]
    losses_by_sim = np.split(severities, splits)
    totals = np.array([x.sum() for x in losses_by_sim])
    return pd.Series(totals)


# 50000 loops
# simulates claims each time
def simulate_aggregate_losses(severity_dist:str) -> pd.Series:
    claim_counts = poisson(mu=LAMBDA_CLAIMS).rvs(size=NUM_SIMS,random_state=rng)
    total_losses = np.empty(NUM_SIMS,dtype=float)
    for j,n in enumerate(claim_counts):
        if n == 0:
            total_losses[j] = 0.0
            continue
        if severity_dist =="lognormal":
            severities = lognorm(s=SIGMA, scale =np.exp(MU)).rvs(size=n, random_state=rng)
        elif severity_dist == "gamma":
            severities = gamma(a=gamma_shape, scale=gamma_scale).rvs(size=n, random_state=rng)
        else:
            raise ValueError("Severity_dist must be 'lognormal' or 'gamma'")
        # Apply Insurance policy limit (claims are capped at this amount)
        severities = np.minimum(severities, POLICY_LIMIT)
        total_losses[j] = severities.sum()
    return pd.Series(total_losses, name = f"total_loss_{severity_dist}")

# Run the original version
# 50000 loops
# simulates claims each time

# loss_logn = simulate_aggregate_losses("lognormal")
# loss_gamm = simulate_aggregate_losses("gamma")


# Run the fast version
# Simulates all claims at once
# Aggregates them efficiently
loss_logn = simulate_aggregate_losses_fast("lognormal")
loss_gamm = simulate_aggregate_losses_fast("gamma")

# Thresholds
thresholds = [100000,150000,200000,250000]

print("\n- Probability Loss Exceeds Threshold -")
for t in thresholds:
    prob = (loss_logn >t).mean()
    print(f"P(Loss > {t:,.0f}) = {prob:.3%}")

var_95 = loss_logn.quantile(0.95)
var_99 = loss_logn.quantile(0.99)

print("\n- Value at Risk -")
print(f"Var 95%: {var_95:,.0f}")
print(f"VaR 99%: {var_99:,.0f}")

# Reinsurance Scenario
retention = 150000

insurer_losses = np.minimum(loss_logn, retention)
reinsurer_losses = np.maximum(loss_logn - retention, 0)

print("\n- Reinsurance Layer Analysis -")
print(f"Mean Insurere loss: {insurer_losses.mean():,.0f}")
print(f"Mean reinsurer loss: {reinsurer_losses.mean():,.0f}")



def summarize(losses: pd.Series) -> dict:
    return {
        "mean": losses.mean(),
        "std":losses.std(),
        "p90": losses.quantile(0.90),
        "p95": losses.quantile(0.95),
        "p99": losses.quantile(0.99),
    }

summary_df = pd.DataFrame(
    {"Lognormal": summarize(loss_logn), "Gamma": summarize(loss_gamm)}
).T
print("\n- Aggregate Loss Summary (Compare Severity Distributiuons) -")
# print(summary_df.applymap(lambda x: f"{x:,.0f}"))
# print(summary_df.applymap(lambda x: f"{x:,.0f}"))
print(summary_df.round(0).astype(int))


#____________________________
#3) Plot
#____________________________

plt.figure()
plt.hist(loss_logn, bins=60, alpha = 0.6, label = "Lognormal severity")
plt.hist(loss_gamm, bins=60,alpha=0.6,label="Gamma severity")

plt.title("Aggregate Loss Distribution Comparison")
plt.xlabel("Total Loss")
plt.ylabel("Frequency")
plt.legend()


plt.figure()
sorted_losses = np.sort(loss_logn)
exceedance_prob = 1 -np.arange(1, len(sorted_losses) +1) / len(sorted_losses)

plt.plot(sorted_losses, exceedance_prob)
plt.title("Loss Exceedance Curve (Lognormal Severity)")
plt.xlabel("Loss Level")
plt.ylabel("Probability Loss Exceeds Level")
plt.yscale("log")

sorted_losses = np.sort(loss_logn)
exceedance_prob = 1 -np.arange(1, len(sorted_losses) +1) / (len(sorted_losses) +1)
return_period =1 / exceedance_prob

plt.figure()
plt.plot(return_period, sorted_losses)
plt.xscale("log")

plt.xlabel("Return Period (Years)")
plt.ylabel("Loss Level")

plt.title("Return Period Curve")

plt.show()
