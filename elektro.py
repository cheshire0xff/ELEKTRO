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
DLUGOSC_APROKSYMACJI_OSCI = 1000
#dla uproszczenia sobie zycia, dzieki temu wiemy ze x = 0 jest poczatkiem stanu niestalonego
POCZATEK_STANU_NIEUSTALONEGO = 430
POCZATEK_STANU_NIEUSTALONEGO_OSCI = 34
#jak dlugi wykres rysowac(nie)
ILOSC_PROBEK_WYKRESU = 1500
POZYCJA_STYCZNEJ = 200

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
def g_factory_osci(b):
    def g(A, a, w, x):
        return A * np.exp(-a * x) * np.cos(w * x) + b 
    return g

def dg_factory_osci(x, y, b):
    def dg(Aaw):
        g = g_factory_osci(b)
        return sum((g(Aaw[0],Aaw[1], Aaw[2], x) - y) ** 2)
    return dg

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

def derivative(Aa, x):
    return Aa[0] * np.exp(-Aa[1]*x) * -Aa[1]

def tangent_factory(der, x0, y0):
    def tangent(x):
        return der * (x - x0) + y0
    return tangent

def approx_plot_0_5(y):
    validate_parameters()
    n = np.arange(0,ILOSC_PROBEK_WYKRESU)
    n = np.transpose(n)
    x = n * 4e-6
    y = y[POCZATEK_STANU_NIEUSTALONEGO:POCZATEK_STANU_NIEUSTALONEGO + ILOSC_PROBEK_WYKRESU]
    plt.plot(x, y)
    print(x[0], y[0])
    dg = dg_factory(x[:DLUGOSC_APROKSYMACJI], y[:DLUGOSC_APROKSYMACJI])
    Aa = optimize.fmin(dg, (1, 1e-3))
    print(Aa[0], Aa[1])
    gapprox = g_factory(y[X_STANU_USTALONEGO])
    approx_y = np.array([gapprox(Aa[0], Aa[1], i) for i in x])
    der = derivative(Aa, x[POZYCJA_STYCZNEJ])
    tangent = tangent_factory(der, x[POZYCJA_STYCZNEJ], approx_y[POZYCJA_STYCZNEJ])
    tangent_y = np.array([tangent(i) for i in x])
    print(f'Approx Func: y = {Aa[0]} * e^({-Aa[1]} * x) + {y[X_STANU_USTALONEGO]}')
    print(f'Tangent Func: y = {der} * (x - {x[POZYCJA_STYCZNEJ]} + {y[POZYCJA_STYCZNEJ]}')
    plt.plot(x, approx_y, label =f'Approx Func: y = {Aa[0]:.1f} * e^({-Aa[1]:.1f} * x) + {y[X_STANU_USTALONEGO]}')
    plt.plot(x, tangent_y, label =f'Tangent Func: y = {der:.1f} * (x - {x[POZYCJA_STYCZNEJ]:.5f}) + {y[POZYCJA_STYCZNEJ]}')
    tau_last_x = np.where(tangent_y >= y[X_STANU_USTALONEGO])[0][0]
    if not tau_last_x:
        tau_last_x=np.where(tangent_y <= y[X_STANU_USTALONEGO])[0][0]
    tau_x = np.array([i for i in x[POZYCJA_STYCZNEJ:tau_last_x - 1]])
    plt.plot(tau_x, [y[X_STANU_USTALONEGO] for _ in range(0, tau_last_x - POZYCJA_STYCZNEJ - 1)],
            label=f'tau(graphically) = {x[tau_last_x - 1] - x[POZYCJA_STYCZNEJ]:.6f}\ntau(1/a) = {1 / Aa[1]:.6f}',
            dashes=[6, 2])
    plt.ylabel('Voltage')
    plt.xlabel('Time')
    plt.legend()
    plt.show()


def approx_plot_6_7(y):
    validate_parameters()
    n = np.arange(0,ILOSC_PROBEK_WYKRESU)
    n = np.transpose(n)
    x = n * 4e-6
    y = y[POCZATEK_STANU_NIEUSTALONEGO_OSCI:POCZATEK_STANU_NIEUSTALONEGO_OSCI + ILOSC_PROBEK_WYKRESU]
    plt.plot(x, y)
    dg = dg_factory_osci(x[:DLUGOSC_APROKSYMACJI_OSCI], y[:DLUGOSC_APROKSYMACJI_OSCI], y[X_STANU_USTALONEGO])
    Aaw = optimize.fmin(dg, (2, 100, 1), maxiter=100000000)
    print(Aaw[0], Aaw[1], Aaw[2])
    gapprox = g_factory_osci(y[X_STANU_USTALONEGO])
    approx_y = np.array([gapprox(Aaw[0],Aaw[1],Aaw[2], i) for i in x])
    print(f'Approx Func(Aaw): y = {Aaw[0]} * e^({-Aaw[1]} * x) * cos({Aaw[2]}*x) + {y[X_STANU_USTALONEGO]}')
    plt.plot(x, approx_y, label =f'Approx Func: y = {Aaw[0]:.1f} * e^({-Aaw[1]:.1f} * x) * cos({Aaw[2]:.1f}*x) + {y[X_STANU_USTALONEGO]}')
    
    prev = approx_y[0]
    j = 0
    for i in range(0, len(approx_y)):
        if prev > approx_y[i]:
            prev = approx_y[i]
            j = i
            break
        else:
            prev = approx_y[i]
    max_loc1 = prev
    max_loc1_x = j
    for i in range(j, len(approx_y)):
        if prev < approx_y[i]:
            prev = approx_y[i]
            j = i
            break
        else:
            prev = approx_y[i]
    for i in range(j, len(approx_y)):
        if prev > approx_y[i]:
            prev = approx_y[i]
            j = i
            break
        else:
            prev = approx_y[i]
    max_loc2 = prev
    max_loc2_x = j
    frequency = 1/(x[max_loc2_x] - x[max_loc1_x])
    dekrement = 2 * np.pi * Aaw[1] /np.sqrt(Aaw[2]**2 - Aaw[1]**2)
    plt.plot([x[max_loc1_x], x[max_loc1_x]], [y[X_STANU_USTALONEGO], max_loc1],'C3', dashes=[4, 2])
    plt.plot([x[max_loc2_x], x[max_loc2_x]], [y[X_STANU_USTALONEGO], max_loc2],'C3', dashes=[4, 2],
            label=f'log dekrement(graficznie) = a * T(okres) = {Aaw[1] * 1/frequency:.3f}\n\
            dekrement = 2*pi*a/sqrt(w^2 - a^2) {dekrement:.3f}')
    plt.plot([x[max_loc1_x], x[max_loc2_x]], [max_loc1, max_loc1], dashes = [4,2],
            label=f'czestotliwosc = {frequency:.1f} Hz')
    plt.ylabel('Napiecie')
    plt.xlabel('Czas')
    plt.title(f'a = {Aaw[1]:.1f}, w = {Aaw[2]:.1f}')
    plt.legend()
    plt.show()


def show_all_basic():
    for i in range(0,3):
            basic_plot(CH1ARR[i], CH2ARR[i])

def show_basic(index):
    basic_plot(CH1ARR[index], CH2ARR[index])
def show_approx(index):
    if index >=0:
        if index < 6:
            approx_plot_0_5(CH1ARR[index])
        elif index < 8:
            approx_plot_6_7(CH1ARR[index])
