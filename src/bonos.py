import numpy as np
import matplotlib.pyplot as plt

DAYS_IN_A_YEAR = 364
DAY = 1 / DAYS_IN_A_YEAR


N_ROWS = 7
ROW_DISC_FACTOR = 0
ROW_COUPON = 1
ROW_AMORT = 3
ROW_TIME = 4
ROW_PV = 5


years = 2 
pays_per_year = 2
interest = .04

bon = np.zeros(N_ROWS * pays_per_year*years).reshape(N_ROWS, pays_per_year * years)
bon[ROW_COUPON] = interest / pays_per_year 
bon[ROW_AMORT, pays_per_year * years - 1] = 1
bon[ROW_TIME] = np.linspace(1/pays_per_year, years, pays_per_year * years)
# bon[0] = np.exp(-bon[4] * .04)
bon[ROW_DISC_FACTOR] = np.power(1 + interest / pays_per_year, -pays_per_year*bon[ROW_TIME])
bon[ROW_PV] = (bon[ROW_COUPON] + bon[ROW_AMORT]) * bon[ROW_DISC_FACTOR]
print(bon)

def simulation(rate: float = .04):
    price = np.zeros(years * DAYS_IN_A_YEAR - 1)
    durt = np.zeros(years * DAYS_IN_A_YEAR - 1)
    positions = [p for p in range(int(DAYS_IN_A_YEAR/pays_per_year), price.shape[0], int(DAYS_IN_A_YEAR/pays_per_year))]
    cupon_acc = np.zeros(years * DAYS_IN_A_YEAR - 1)
    np.put(cupon_acc, positions, interest / pays_per_year)
    bon_c = np.copy(bon)
    for t in range(int( years * DAYS_IN_A_YEAR) - 1):
        bon_c[ROW_COUPON] *= (1 + cupon_acc[t])
        bon_c[ROW_AMORT] *= (1 + cupon_acc[t])
        bon_c[ROW_TIME] -= DAY
        mask = np.where(bon_c[ROW_TIME] < 1e-6 , 0, 1)
        bon_c[ROW_DISC_FACTOR] = np.power(1 + rate / pays_per_year, -pays_per_year * bon_c[ROW_TIME]) * mask
        bon_c[ROW_PV] = (bon_c[ROW_COUPON] + bon_c[ROW_AMORT]) * bon_c[ROW_DISC_FACTOR]
        durt[t] = (bon_c[ROW_PV] * bon_c[ROW_TIME] * mask).sum()
        price[t] =  bon_c[ROW_PV].sum()
    plt.plot(durt)
    plt.show()
    return price

prices = simulation(.5)



