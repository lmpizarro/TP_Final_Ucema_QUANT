import numpy as np

DAYS_IN_A_YEAR = 364

DAY = 1 / DAYS_IN_A_YEAR
COUPON = 1
AMORT = 3
TIME = 4
DISC_FACTOR = 0
PV = 5

rows = 6
years = 2 
pays_per_year = 2
interest = .04

bon = np.zeros(rows * pays_per_year*years).reshape(rows,pays_per_year * years)
bon[COUPON] = interest / pays_per_year 
bon[AMORT, pays_per_year * years - 1] = 1
bon[TIME] = np.linspace(1/pays_per_year, years, pays_per_year * years)
# bon[0] = np.exp(-bon[4] * .04)
bon[DISC_FACTOR] = np.power(1 + interest / pays_per_year, -pays_per_year*bon[TIME])

bon[PV] = (bon[COUPON] + bon[AMORT]) * bon[DISC_FACTOR]


print(bon)

print(bon[5].sum())

price = np.zeros(years * DAYS_IN_A_YEAR - 1)
positions = [p for p in range(180, price.shape[0], 180)]
cupon_acc = np.zeros(years * DAYS_IN_A_YEAR - 1)
np.put(cupon_acc, positions, interest / pays_per_year)
for t in range(int( years * DAYS_IN_A_YEAR) - 1):
    bon[TIME] -= DAY
    mask = np.where(bon[TIME] < 1e-6 , 0, 1)
    bon[DISC_FACTOR] = np.power(1 + .04 / pays_per_year, -pays_per_year * bon[TIME]) * mask
    bon[PV] = (bon[COUPON] + bon[AMORT]) * bon[DISC_FACTOR]
    price[t] =  bon[PV].sum()

print(cupon_acc)
print(bon[PV].sum())

import matplotlib.pyplot as plt
plt.plot(price)
plt.show()
print(price.shape)
