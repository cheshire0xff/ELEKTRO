import numpy as np
from scipy import optimize
from matplotlib import pyplot as plt
import matplotlib

# :,0 coz these CSV have endline chars
CH1ARR = [np.genfromtxt(f'A000{i}CH1.CSV', delimiter = ',', skip_header=25)[:,0] for i in range(0, 9)]
CH2ARR = [np.genfromtxt(f'A000{i}CH2.CSV', delimiter = ',', skip_header=25)[:,0] for i in range(0, 9)]

#ponizsze wartosci dramatycznie wplywaja na jakosc aproksymacji
#zaleznie od wykresu czasem warto zmienic w funkcji g sin na cos

#potrzebne do ustalenia wartosci stanu ustalonego, zaleznie od wykresu
#mniej wiecej wszystkie wykresy w tym samym miejscu maja stan ustalony
X_STANU_USTALONEGO = 900

#aby nie probowac aproksymowac za duzo, co spowoduje absuradlny wynik
#idealnie powinna to byc ilosc probek miedzy poczatkiem stanu nieustalonego a koncem ustalonego
DLUGOSC_APROKSYMACJI = 901
DLUGOSC_APROKSYMACJI_OSCI = 1000

#po to aby aproksymacja byla skuteczna koniecznym jest aby początek wykresu zaczynal sie
#od poczatku stanu nieustalonego ponizsze zmienne to ustawiaja
POCZATEK_STANU_NIEUSTALONEGO = 1387
POCZATEK_STANU_NIEUSTALONEGO_OSCI = 1165

#jak dlugi wykres rysowac
ILOSC_PROBEK_WYKRESU = 1500

#miejsce w ktorym narysuje styczna
#tau wyznaczone ze stycznej powinno byc te same w kazdym miejscu
#graficznie jednak nie zawsze dobrze sie to prezentuje
POZYCJA_STYCZNEJ = 200

def validate_parameters():
    for arr in CH1ARR:
        assert len(arr) >= ILOSC_PROBEK_WYKRESU
        assert POCZATEK_STANU_NIEUSTALONEGO < len(arr) + DLUGOSC_APROKSYMACJI
    for arr in CH2ARR:
        assert len(arr) >= ILOSC_PROBEK_WYKRESU
    assert X_STANU_USTALONEGO < DLUGOSC_APROKSYMACJI
    assert DLUGOSC_APROKSYMACJI < ILOSC_PROBEK_WYKRESU

def g_factory(b):
    def g(A, a, x):
        return  A * np.exp( -a * x) + b
    return g
def g_factory_osci(b):
    def g(A, a, w, x):
        return A * np.exp(-a * x) * np.sin(w * x) + b 
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
    plt.plot(x * 1000, y, label='Dane z oscyloskopu')
    print(x[0], y[0])
    dg = dg_factory(x[:DLUGOSC_APROKSYMACJI], y[:DLUGOSC_APROKSYMACJI])
    Aa = optimize.fmin(dg, (1, 1e-3), maxfun=10000)
    print(Aa[0], Aa[1])
    gapprox = g_factory(y[X_STANU_USTALONEGO])
    approx_y = np.array([gapprox(Aa[0], Aa[1], i) for i in x])
    der = derivative(Aa, x[POZYCJA_STYCZNEJ])
    tangent = tangent_factory(der, x[POZYCJA_STYCZNEJ], approx_y[POZYCJA_STYCZNEJ])
    tangent_y = np.array([tangent(i) for i in x])
    print(f'Aproksymacja:  A*exp^(-a*x) + b\ny = {Aa[0]} * e^({-Aa[1]} * x) + {y[X_STANU_USTALONEGO]}')
    print(f"Styczna:  y - y0 = f'(x)(x - x0)\ny = {der} * (x - {x[POZYCJA_STYCZNEJ]} + {y[POZYCJA_STYCZNEJ]})") 
    print(f'x0 = {x[POZYCJA_STYCZNEJ]}')
    print(f'y0 = {approx_y[POZYCJA_STYCZNEJ]}')
    plt.plot(x * 1000, approx_y, label =f'Aproksymacja:  ' + r'$ A*e^{-a*x} + b$')# + '\n' + r'${Aa[0]:.1f} * e^({-Aa[1]:.1f} * x) + {y[X_STANU_USTALONEGO]}$')
    plt.plot(x * 1000, tangent_y, label =f"Styczna:  " + r"$y = f'(x)(x - x_0) + y_0$") #\ny = {der:.1f} * (x - {x[POZYCJA_STYCZNEJ]:.5f}) + {y[POZYCJA_STYCZNEJ]}")
    tau_last_x=np.where(tangent_y >= y[X_STANU_USTALONEGO])[0][0]
    if not tau_last_x:
        tau_last_x=np.where(tangent_y <= y[X_STANU_USTALONEGO])[0][0]
    tau_x = np.array([i for i in x[POZYCJA_STYCZNEJ:tau_last_x - 1]])
    taugraph = (y[X_STANU_USTALONEGO] - (-der * x[POZYCJA_STYCZNEJ] + approx_y[POZYCJA_STYCZNEJ])) / der - x[POZYCJA_STYCZNEJ]
    plt.plot(tau_x * 1000, [y[X_STANU_USTALONEGO] for _ in range(0, tau_last_x - POZYCJA_STYCZNEJ - 1)],
            label=f'tau(graficznie)= {taugraph:.6f}\ntau(1/a) = {1 / Aa[1]:.6f}',
            dashes=[6, 2])
    plt.ylabel('Napiecie [V]')
    plt.xlabel('Czas [ms]')
    plt.legend()
    plt.grid(True)
    print(
'\
[$]A = %.2f\n[/$]<br>\
[$]a = %.2f\n[/$]<br>\
[$]b = %.1f\n[/$]<br>\
[$]f^\\prime\\left(x\\right)=A\\cdot e^{-ax}\\cdot-b\n[/$]<br>\
[$]x_0 = %.4f \n[/$]<br>\
[$]y_0=%.2f\\cdot e^{-\\left(%.2f\\right)\\cdot%.2f}+%.1f=%.2f\n[/$]<br>\
[$]f^\\prime\\left(%.4f\\right)=%.2f\\cdot e^{-\\left(%.2f\\right)\\cdot%.4f}\\cdot-\\left(%.2f\\right)=%.2f\n[/$]<br>\
funkcja stycznej: [$]y\\ =\\ %.2fx\\ +\\ %.2f\n[/$]<br>\
x dla stycznej dla wartosci [$]y = %.1f\n[/$]<br>\
[$]%.1f\\ =\\ %.2fx\\ +\\ %.2f\n[/$]<br>\
[$]x\\ =\\ %.4f\n[/$]<br>\
[$]tau = x - x_0 = %.6f\n[/$]<br>\
[$]tau2 = 1 / a = %.6f\n[/$]<br>\
'
        %
        (
        Aa[0],
        Aa[1], 
        y[X_STANU_USTALONEGO],
        x[POZYCJA_STYCZNEJ],
        Aa[0], Aa[1], x[POZYCJA_STYCZNEJ], y[X_STANU_USTALONEGO] ,approx_y[POZYCJA_STYCZNEJ],
        x[POZYCJA_STYCZNEJ], Aa[0], Aa[1], x[POZYCJA_STYCZNEJ], Aa[1], der,
        der, -der * x[POZYCJA_STYCZNEJ] + approx_y[POZYCJA_STYCZNEJ],
        y[X_STANU_USTALONEGO],
        y[X_STANU_USTALONEGO], der, -der * x[POZYCJA_STYCZNEJ] + approx_y[POZYCJA_STYCZNEJ],
        (y[X_STANU_USTALONEGO] - (-der * x[POZYCJA_STYCZNEJ] + approx_y[POZYCJA_STYCZNEJ])) / der,
        taugraph,
        1 / Aa[1]
        )
        )

    plt.show()



def approx_plot_6_7(y):
    validate_parameters()
    n = np.arange(0,ILOSC_PROBEK_WYKRESU)
    n = np.transpose(n)
    x = n * 4e-6
    y = y[POCZATEK_STANU_NIEUSTALONEGO_OSCI:POCZATEK_STANU_NIEUSTALONEGO_OSCI + ILOSC_PROBEK_WYKRESU]
    plt.plot(x, y, label='Dane z oscyloskopu')
    dg = dg_factory_osci(x[:DLUGOSC_APROKSYMACJI_OSCI], y[:DLUGOSC_APROKSYMACJI_OSCI], y[X_STANU_USTALONEGO])
    Aaw = optimize.fmin(dg, (2, 100, 1), maxiter=100000000)
    print(Aaw[0], Aaw[1], Aaw[2])
    gapprox = g_factory_osci(y[X_STANU_USTALONEGO])
    approx_y = np.array([gapprox(Aaw[0],Aaw[1],Aaw[2], i) for i in x])
    print(f'Aproksymacja: A*e^(-a*x) * sin(w*x) + b\ny = {Aaw[0]} * e^({-Aaw[1]} * x) * cos({Aaw[2]}*x) + {y[X_STANU_USTALONEGO]}')
    plt.plot(x, approx_y, label =f'Aproksymacja: ' + r'$A*e^{-a*x} * sin(w*x) + b$')# + '\n' + r'$y = {Aaw[0]:.1f} * e^({-Aaw[1]:.1f} * x) * cos({Aaw[2]:.1f}*x) + {y[X_STANU_USTALONEGO]}$')
    
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
    frequency_graph = 1/(x[max_loc2_x] - x[max_loc1_x])
    frequency = Aaw[2] / (2 * np.pi)
    T = 1/ frequency
    dekrement_graph = frequency * np.log(max_loc1/max_loc2)
    print('dekrement', dekrement_graph)
    plt.plot([x[max_loc1_x], x[max_loc1_x]], [y[X_STANU_USTALONEGO], max_loc1],'C3', dashes=[4, 2])
    plt.plot([x[max_loc2_x], x[max_loc2_x]], [y[X_STANU_USTALONEGO], max_loc2],'C3', dashes=[4, 2],
            label=f'Logaritmiczny dekrement tlumienia = {Aaw[1]:.2f}')# = {:.1f}\n' +
#            f'Logaritmiczny dekrement tlumienia = 2*pi*a/sqrt(w^2 - a^2) = {dekrement:.3f}')
    plt.plot([x[max_loc1_x], x[max_loc2_x]], [max_loc1, max_loc1], dashes = [4,2],
            label=f'Czestotliwosc = {frequency:.1f} Hz')
    plt.ylabel('Napiecie [V]')
    plt.xlabel('Czas[s]')
    plt.title(f'A = {Aaw[0]:.1f}, a = {Aaw[1]:.1f}, w = {Aaw[2]:.1f}')
    plt.legend()
    print(
'\
[$]A = %.2f\n[/$]<br>\
[$]a = %.2f\n[/$]<br>\
[$]b = %.1f\n[/$]<br>\
[$]\\phi =\\fraq{1}{T}\\ln{\\frac{A_n}{A_{n + 1}}}= %.2f\n[/$]<br>\
[$]Czestotliwosc x_1 - x_2 = %.1f \n[/$]<br>\
'
        %
        (
        Aaw[0],
        Aaw[1], 
        y[X_STANU_USTALONEGO],
        dekrement_graph,
        frequency_graph,
        )
        )

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
