import pandas as pd
import matplotlib.pyplot as plt
import datetime
import math


def get_data(filename):
    stock_data = pd.read_csv(filename)
    closing_prices = stock_data['Close']
    closing_prices = closing_prices[:1000]
    dates = stock_data['Date']
    dates = dates[:1000]
    dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in dates]
    highs = stock_data['High']
    highs = highs[:1000]
    lows = stock_data['Low']
    lows = lows[:1000]
    return closing_prices, dates, highs, lows


def ema(data, n, current):
    alpha = 2 / (n + 1)
    numerator = 0.0
    denominator = 0.0

    for i in range(n + 1):
        if current-i >= 0:
            numerator += pow((1 - alpha), i) * data[current - i]
            denominator += pow((1 - alpha), i)
    return numerator/denominator


def macd(data, no_of_samples):
    MACD = list(range(0, no_of_samples))

    for i in range(no_of_samples):
        EMA12 = ema(data, 12, i)
        EMA26 = ema(data, 26, i)
        MACD[i] = EMA12 - EMA26
    return MACD


def signal(MACD, no_of_samples):
    SIGNAL = list(range(0, no_of_samples))

    for i in range(no_of_samples):
        SIGNAL[i] = ema(MACD, 9, i)
    return SIGNAL


def buy_and_sell(closing_prices, macd, signal, no_of_samples):
    owned_stock = 0
    funds = 1000.0

    for i in range(1, no_of_samples):
        if macd[i - 1] <= signal[i - 1] and macd[i] > signal[i]:
            owned_stock += math.floor(funds / closing_prices[i])
            funds -= float(owned_stock * closing_prices[i])
        elif macd[i - 1] >= signal[i - 1] and macd[i] < signal[i]:
            funds += float(owned_stock * closing_prices[i])
            owned_stock = 0
    funds += float(owned_stock * closing_prices[no_of_samples - 1])
    return funds


def sma(data, current, n):
    sum = 0.0
    for i in range(n):
        if current - i >= 0:
            sum += data[current - i]
    return sum / n


def stochastic_osc(closing_prices, highs, lows, no_of_samples, period):
    k_line = [0] * (no_of_samples)
    for i in range(period-1, no_of_samples):
        lowest_low = min(lows[i+1-period:i+1])
        highest_high = max(highs[i+1-period:i+1])
        k = (closing_prices[i] - lowest_low) / (highest_high - lowest_low) * 100
        k_line[i] = k
    return k_line


def buy_and_sell_stochastic(closing_prices, k_line, d_line, no_of_samples):
    funds = 1000
    owned_stock = 0
    position = -1
    for i in range(1000):
        if k_line[i] < d_line[i] < 15 and position != 1:
            owned_stock += math.floor(funds / closing_prices[i])
            funds -= float(owned_stock * closing_prices[i])
            position = 1
        elif k_line[i] > d_line[i] > 90 and position != 0:
            funds += float(owned_stock * closing_prices[i])
            owned_stock = 0
            position = 0
    funds += float(owned_stock * closing_prices[no_of_samples - 1])
    return funds


def buy_and_sell_both(closing_prices, MACD, SIGNAL, no_of_samples, k_line, d_line):
    owned_stock = 0
    funds = 1000.0
    position = 0

    signal_macd = 0
    signal_so = 0

    for i in range(1, no_of_samples):
        if 20 > d_line[i] > k_line[i]:
            if position != 1:
                owned_stock += math.floor(funds / closing_prices[i])
                funds -= float(owned_stock * closing_prices[i])
                position = 1
                signal_so = 2
        elif 90 < d_line[i] < k_line[i]:
            if position != -1:
                funds += float(owned_stock * closing_prices[i])
                owned_stock = 0
                position = -1
                signal_so = -2
        if MACD[i - 1] <= SIGNAL[i - 1] and MACD[i] > SIGNAL[i]:
            signal_macd = 2
        elif MACD[i - 1] >= SIGNAL[i - 1] and MACD[i] < SIGNAL[i]:
            signal_macd = -2

        if signal_macd < 0:
            signal_macd += 1
        elif signal_macd > 0:
            signal_macd -= 1
        if signal_so < 0:
            signal_so += 1
        elif signal_so > 0:
            signal_so -= 1

        if signal_macd > 0 and signal_so > 0:
            owned_stock += math.floor(funds / closing_prices[i])
            funds -= float(owned_stock * closing_prices[i])
        elif signal_macd < 0 and signal_so < 0:
            funds += float(owned_stock * closing_prices[i])
            owned_stock = 0

    funds += float(owned_stock * closing_prices[no_of_samples - 1])
    return funds


closing_prices, dates, highs, lows = get_data('SPOT.csv')

# transactions analyzed in the report
markers1 = [47, 62]
markers2 = [112, 122]

plt.figure(figsize=(10, 5))
plt.plot(dates, closing_prices)
plt.scatter([dates[markers1[0]], dates[markers1[1]]], [closing_prices[markers1[0]], closing_prices[markers1[1]]], marker=".", color="red", zorder=2)
plt.scatter([dates[markers2[0]], dates[markers2[1]]], [closing_prices[markers2[0]], closing_prices[markers2[1]]], marker=".", color="black", zorder=2)
plt.title("Ceny zamknięcia dla spółki Spotify Technology S.A.")
plt.xlabel("Data")
plt.ylabel("Cena zamknięcia w USD")
plt.show()


# MACD and SIGNAL
MACD = macd(closing_prices, 1000)
SIGNAL = signal(MACD, 1000)

plt.figure(figsize=(10, 5))
plt.plot(dates, MACD)
plt.plot(dates, SIGNAL)
plt.scatter([dates[markers1[0]], dates[markers1[1]]], [MACD[markers1[0]], MACD[markers1[1]]], marker=".", color="red", zorder=2)
plt.scatter([dates[markers2[0]], dates[markers2[1]]], [MACD[markers2[0]], MACD[markers2[1]]], marker=".", color="black", zorder=2)
plt.title("Wskaźnik MACD i SIGNAL")
plt.xlabel("Data")
plt.ylabel("Wartość wskaźnika")
plt.legend(['MACD', 'SIGNAL'])
plt.show()


# buying and selling using MACD
capital = buy_and_sell(closing_prices, MACD, SIGNAL, 1000)
print(f"Kapitał końcowy z wykorzystaniem MACD: {capital:.2f}\n")


# stochastic oscillator
k_line = stochastic_osc(closing_prices, highs, lows, 1000, 14)
d_line = [0] * 1000

for i in range(1000):
    d_line[i] = sma(k_line, i, 3)


plt.figure(figsize=(10, 5))
plt.plot(dates, k_line)
plt.plot(dates, d_line)
plt.title("Oscylator stochastyczny")
plt.xlabel("Data")
plt.ylabel("Wartość % wskaźnika")
plt.legend(['%K', '%D'])
plt.show()

# buying and selling using stochastic oscillator
capital = buy_and_sell_stochastic(closing_prices, k_line, d_line, 1000)
print(f"Kapitał końcowy z wykorzystaniem oscylatora: {capital:.2f}\n")

# buying and selling using MACD and stochastic oscillator
capital = buy_and_sell_both(closing_prices, MACD, SIGNAL, 1000, k_line, d_line)
print(f"Kapitał końcowy z wykorzystaniem MACD i oscylatora: {capital:.2f}\n")
