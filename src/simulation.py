import pickle
import numpy as np
from nelson_siegel_svensson.calibrate import calibrate_ns_ols
from datetime import datetime, timedelta, date
import matplotlib.pyplot as plt

t = np.asarray([0.0,   0.5, 1.0,   2.0,   3.0,   4.0,   5.0, 10.0, 15.0, 20.0, 25.0, 30.0])
y = np.asarray([0.01, 0.02, 0.03, 0.035, 0.03, 0.035, 0.040, 0.05, 0.035, 0.037, 0.038, 0.04])
nsv_curve, status = calibrate_ns_ols(t, y, tau0=1.0)  # starting value of 1.0 for the optimization of tau

def create_amortizable_bond(years=5, pays_per_year=2, rate: float=.05, first_pay: str='09/01/23', n_amort: int=3, amortizacion=20):
    periods = pays_per_year * 2

    p_rate = rate / pays_per_year

    inicia_amort = years * pays_per_year - n_amort + 1
    coef_desc = np.power((1+rate/pays_per_year), -(np.arange(pays_per_year * years) + 1))

    pago_tasa = 100 * p_rate * np.ones(pays_per_year * years) 
    pago_amortizacion_zeros = np.zeros(inicia_amort - 1)

    # amortizacion = 100 / (pays_per_year * years - inicia_amort + 2)
    pago_amortizacion = amortizacion * np.ones(pays_per_year * years - inicia_amort + 1)

    print(pago_tasa.shape, pago_amortizacion_zeros.shape, pago_amortizacion.shape)
    pago_amortizacion = np.concatenate([pago_amortizacion_zeros, pago_amortizacion])
    pago_total = pago_tasa + pago_amortizacion
    print(pago_tasa, pago_total, pago_amortizacion, pago_total.sum())
  
    sum_menos_uno_amortizacion = (coef_desc * pago_amortizacion)[:-1].sum()
    sum_tasa_total = (coef_desc * pago_tasa).sum()
   
    resto_100 = 100 - (sum_menos_uno_amortizacion + sum_tasa_total)
    pago_amortizacion[-1] = (resto_100 / coef_desc[-1]).round(2)
    print((coef_desc * (pago_tasa + pago_amortizacion)).sum())

    pagos = []
    pay_date = datetime.strptime(first_pay, "%d/%m/%y").date()
    k = 0
    for i in range(years):
        for j in range(pays_per_year):
            pagos.append((pay_date.strftime("%d/%m/%y"), pago_tasa[k], pago_amortizacion[k]))
            pay_date = pay_date + timedelta(days=180)
            k += 1
 
    bono = {}
    bono['pagos'] = pagos

    return bono


def create_bullet_bond(face: float=100, years: float=10, pays_per_year: int=2,
                       rate: float=.05, first_pay: str='09/01/23'):

    pagos = []
    pay_date = datetime.strptime(first_pay, "%d/%m/%y").date()

    p_rate = rate / pays_per_year
    for i in range(years-1):
        pago = p_rate * face
        for j in range(pays_per_year):
            pagos.append((pay_date.strftime("%d/%m/%y"), pago, 0))
            pay_date = pay_date + timedelta(days=180)
    pagos.append((pay_date.strftime("%d/%m/%y"), pago, 0))

    pay_date = pay_date + timedelta(days=180)
    pagos.append((pay_date.strftime("%d/%m/%y"), pago, 100))
    name = f'B{rate}-{str(pay_date.year)[2:]}'
    bono = {}
    bono['pagos'] = pagos

    return bono


def volatility_model(t, tao, sigma=0.0005, alfa=0.1, beta=.5):
    return np.random.normal(0, sigma*(1 - alfa*t/tao + beta))

def define_term_curve(r, total_dates, s=1, term='flat'):
    dates_index = np.asarray(list(range(total_dates))) / DAYS_IN_A_YEAR
    if term == 'flat':
        print('FLAT')
        curve = r * np.ones(dates_index.shape[0])
    elif term == 'exp_inv':
        print('INV')
        curve = r * (np.exp(-s*dates_index))
    elif term == 'exp':
        curve = r * (1 - np.exp(-s*dates_index))
    else:
        curve = nsv_curve(dates_index)
    return curve

DAYS_IN_A_YEAR = 364
def calc_hist_price(bono, r=0.05, init_date=datetime(2022, 9, 20).date(), term='exp_inv'):
    mem_pagos = []

    for pago in bono['pagos']:
        amortizacion = pago[2]
        renta = pago[1]
        mem_pagos.append([datetime.strptime(pago[0], "%d/%m/%y").date(), renta, amortizacion])
    total_dates = (mem_pagos[-1][0]-init_date).days

    curve = define_term_curve(r, total_dates, term=term)

    plt.plot(range(total_dates), curve)
    plt.show()
    valores = []
    durations = []
    for i in range(total_dates + 1):
        new_date = init_date + timedelta(days=i)
        valor_dia = 0
        duration_dia = []
        cupon = False
        for j, mp in enumerate(mem_pagos):

            if new_date < mp[0]:
                ttm_dates = (mp[0] - new_date).days
                r_curve = curve[ttm_dates-1] + volatility_model(ttm_dates, total_dates)
                ttm_years = ttm_dates/DAYS_IN_A_YEAR
                valor = mp[1]+ mp[2]
                valor_ajustado = valor * np.exp(-r_curve*ttm_years)
                duration = valor_ajustado * ttm_years

            else:
                valor_ajustado = 0
                duration = 0

            duration_dia.append(duration)
            valor_dia += valor_ajustado
            if new_date == mp[0]:
                cobro = mp[1] + mp[2]
                cupon = False

        if valor_dia != 0:
            valores.append(valor_dia)
            durations.append(np.asarray([duration_dia]).sum()/valor_dia)
            if cupon:
                cupon = False
                incremento = 1 + cobro / valor_dia 
                print('valor_dia ', cobro, valor_dia, incremento)
                mem_pagos = [[e[0], incremento * e[1], incremento * e[2]] for e in mem_pagos]


    return valores, durations



def test_calc_prices():
    bono = create_bullet_bond(years=4)
    bono = create_amortizable_bond()

    with open('bonos.pkl', 'rb') as fp:
        bonds = pickle.load(fp)

    bono = bonds['AL30']
    # bono = bonos['GD29']


    val_per_dia, durations_dia = calc_hist_price(bono)

    plt.grid()
    plt.plot(val_per_dia)
    plt.axhline(94)
    plt.axhline(106)
    plt.show()

    plt.plot(durations_dia)
    plt.show()



def main():
    # test_others()
    test_calc_prices()

if __name__ == '__main__':
    main()

