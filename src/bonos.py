import numpy as np

DAYS_IN_A_YEAR = 364

DAY = 1 / DAYS_IN_A_YEAR
COUPON = 1
AMORT = 3
TIME = 4
DISC_FACTOR = 0

rows = 6
years = 2 
pays_per_year = 3
interest = .04

bon = np.zeros(rows * pays_per_year*years).reshape(rows,pays_per_year * years)
bon[COUPON] = interest / pays_per_year 
bon[AMORT, pays_per_year * years - 1] = 1
bon[TIME] = np.linspace(1/pays_per_year, years, pays_per_year * years)
# bon[0] = np.exp(-bon[4] * .04)
bon[DISC_FACTOR] = np.power(1 + interest / pays_per_year, -pays_per_year*bon[TIME])


bon[5] = (bon[1] + bon[3]) * bon[0]


print(bon)

print(bon[5].sum())
p = np.zeros(years * DAYS_IN_A_YEAR - 1)
for i in range(years * DAYS_IN_A_YEAR - 1):
    bon[4] -= DAY
    mask = np.where(bon[4] < 1e-6 , 0, 1)
    bon[0] = np.power(1 + .04 / 2, -2*bon[4]) * mask
    bon[5] = (bon[1] + bon[3]) * bon[0]
    p[i] =  bon[5].sum()

print(bon)
print(bon[5].sum())

import matplotlib.pyplot as plt
plt.plot(p)
plt.show()
print(p.shape)
