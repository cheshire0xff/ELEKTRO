import numpy as np
from scipy import optimize
from matplotlib import pyplot as plt

# :,0 coz these CSV have endline chars

CH1ARR = [np.genfromtxt(f'A000{i}CH1.CSV', delimiter = ',', skip_header=25)[:,0] for i in range(0, 9)]
CH2ARR = [np.genfromtxt(f'A000{i}CH2.CSV', delimiter = ',', skip_header=25)[:,0] for i in range(0, 9)]
#potrzebne do ustalenia wartosci stanu ustalonego, zaleznie od wykresu
#mniej wiecej wszystkie wykresy w tym samym miejscu maja stan ustalony
X_STANU_USTALONEGO = 900
#aby nie probowac aproksymowac za duzo, co spowoduje absuradlny wynik
#idealnie powinna to byc ilosc probek miedzy poczatkiem stanu nieustalonego a koncem ustalonego
DLUGOSC_APROKSYMACJI = 1000
#dla uproszczenia sobie zycia, dzieki temu wiemy ze x = 0 jest poczatkiem stanu niestalonego
POCZATEK_STANU_NIEUSTALONEGO = 430
#jak dlugi wykres rysowac(nie)
ILOSC_PROBEK_WYKRESU = 1500

def validate_parameters():
    for arr in CH1ARR:
        assert len(arr) >= ILOSC_PROBEK_WYKRESU
    for arr in CH2ARR:
        assert len(arr) >= ILOSC_PROBEK_WYKRESU
    assert X_STANU_USTALONEGO < DLUGOSC_APROKSYMACJI
    assert DLUGOSC_APROKSYMACJI < ILOSC_PROBEK_WYKRESU
    assert POCZATEK_STANU_NIEUSTALONEGO < X_STANU_USTALONEGO

def g_factory(b):
    def g(A, a, x):
        return  A * np.exp( -a * x) + b
    return g

def dg_factory(x, y):
    def dg(Aa):
        g = g_factory(y[X_STANU_USTALONEGO])
        return sum((g(Aa[0],Aa[1], x) - y) ** 2)
    return dg

def basic_plot(y1, y2):
    validate_parameters()
    n = np.arange(0,ILOSC_PROBEK_WYKRESU)
    n = np.transpose(n)
    x = n * 4e-6
    y1 = y1[POCZATEK_STANU_NIEUSTALONEGO:POCZATEK_STANU_NIEUSTALONEGO + ILOSC_PROBEK_WYKRESU]
    y2 = y2[POCZATEK_STANU_NIEUSTALONEGO:POCZATEK_STANU_NIEUSTALONEGO + ILOSC_PROBEK_WYKRESU]
    plt.plot(x, y1)
    #plt.plot(x, y2)
    plt.ylabel('Voltage')
    plt.xlabel('Time')
    plt.show()

def approx_plot(y):
    validate_parameters()
    n = np.arange(0,ILOSC_PROBEK_WYKRESU)
    n = np.transpose(n)
    x = n * 4e-6
    y = y[POCZATEK_STANU_NIEUSTALONEGO:POCZATEK_STANU_NIEUSTALONEGO + ILOSC_PROBEK_WYKRESU]
    plt.plot(x, y)
    print(x[0], y[0])
    dg = dg_factory(x[:DLUGOSC_APROKSYMACJI], y[:DLUGOSC_APROKSYMACJI])
    Aa = optimize.fmin(dg, (1, 1e-3))
    print(Aa)
    gapprox = g_factory(y[X_STANU_USTALONEGO])
    approx_y = np.array([gapprox(Aa[0], Aa[1], i) for i in x])
    plt.plot(x, approx_y)
    plt.ylabel('Voltage')
    plt.xlabel('Time')
    plt.show()




def show_all_basic():
    for i in range(0,3):
            basic_plot(CH1ARR[i], CH2ARR[i])

def show_basic(index):
    basic_plot(CH1ARR[index], CH2ARR[index])
def show_approx(index):
    approx_plot(CH1ARR[index])
