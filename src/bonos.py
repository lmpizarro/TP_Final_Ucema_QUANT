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
interest = .08

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
    rates = rate * np.ones(years * DAYS_IN_A_YEAR - 1)
    # rates = rate * np.exp(-0.001 * np.arange(0, years * DAYS_IN_A_YEAR - 1))
    # rates = rate * (1-np.exp(-0.001 * np.arange(0, years * DAYS_IN_A_YEAR - 1)))

    durt = np.zeros(years * DAYS_IN_A_YEAR - 1)
    days_cupons = [p for p in range(int(DAYS_IN_A_YEAR/pays_per_year), price.shape[0], int(DAYS_IN_A_YEAR/pays_per_year))]
    cupon_acc = np.zeros(years * DAYS_IN_A_YEAR - 1)
    np.put(cupon_acc, days_cupons, interest / pays_per_year)
    bon_c = np.copy(bon)
    for t in range(int( years * DAYS_IN_A_YEAR) - 1):
        # bon_c[ROW_COUPON] *= (1 + cupon_acc[t])
        # bon_c[ROW_AMORT] *= (1 + cupon_acc[t])
        bon_c[ROW_TIME] -= DAY
        mask_past = np.where(bon_c[ROW_TIME] < 1e-6 , 0, 1)
        bon_c[ROW_DISC_FACTOR] = np.power(1 + rates[t] / pays_per_year, -pays_per_year * bon_c[ROW_TIME]) * mask_past
        bon_c[ROW_PV] = (bon_c[ROW_COUPON] + bon_c[ROW_AMORT]) * bon_c[ROW_DISC_FACTOR]
        durt[t] = (bon_c[ROW_PV] * bon_c[ROW_TIME] * mask_past).sum()
        price[t] =  bon_c[ROW_PV].sum()
    return price



from datetime import datetime

datetime_str = '07/07/22 00:00:00'
from_time = datetime.strptime(datetime_str, '%m/%d/%y %H:%M:%S')
now = datetime.now()
delta_time = (now - from_time).days

ONE_PBS = .1 / 100

prices0 = simulation(.08)
prices1 = simulation(.08 - 1*ONE_PBS)
plt.plot(prices0)
plt.show()

delta_p =  prices1[delta_time] - prices0[delta_time] 

print(100*delta_p/ prices0[delta_time])

prices = np.zeros(100 * (years * DAYS_IN_A_YEAR - 1)).reshape(100,  years * DAYS_IN_A_YEAR - 1)
for r in range(0, 100):
    prices[r] = simulation(r/100.0)

print(prices.shape)
for i in range(0, prices.shape[1]-30, 30):
    plt.plot(prices[:, [i+1]].reshape(100))
    print(i)
plt.show()
