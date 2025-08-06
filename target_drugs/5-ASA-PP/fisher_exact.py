# active users: 13/31, 5/20, 1/2
# healthy controls: 0/26, 0/0, 0/15

# conduct Fisher's exact test for 5-ASA use



import numpy as np
from scipy.stats import fisher_exact

# Create the contingency table
# Rows: Active Users, Controls
# Columns: Drug Detected, Drug Not Detected
contingency_table = np.array([
    [19, 34],  # Active users: 19 detected, 34 not detected
    [0, 41]    # Controls: 0 detected, 41 not detected
])

print("Contingency Table:")
print("                Drug Detected    Drug Not Detected    Total")
print(f"Active Users         {contingency_table[0,0]}              {contingency_table[0,1]}              {contingency_table[0].sum()}")
print(f"Controls              {contingency_table[1,0]}              {contingency_table[1,1]}              {contingency_table[1].sum()}")
print(f"Total                {contingency_table[:,0].sum()}              {contingency_table[:,1].sum()}              {contingency_table.sum()}")
print()

# Calculate proportions
prop_active = contingency_table[0,0] / contingency_table[0].sum()
prop_control = contingency_table[1,0] / contingency_table[1].sum()

print(f"Proportion with drug detected:")
print(f"Active users: {prop_active:.1%} ({contingency_table[0,0]}/{contingency_table[0].sum()})")
print(f"Controls: {prop_control:.1%} ({contingency_table[1,0]}/{contingency_table[1].sum()})")
print()

# Fisher's Exact Test
odds_ratio, p_fisher = fisher_exact(contingency_table)

print("FISHER'S EXACT TEST")
print("=" * 40)
print(f"Odds Ratio: {odds_ratio:.2f}")
print(f"P-value: {p_fisher:.2e}")

if p_fisher < 0.001:
    print("Result: Highly significant (p < 0.001)")
elif p_fisher < 0.01:
    print("Result: Very significant (p < 0.01)")
elif p_fisher < 0.05:
    print("Result: Significant (p < 0.05)")
else:
    print("Result: Not significant (p ≥ 0.05)")
