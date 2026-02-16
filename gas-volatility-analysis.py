# Project 1: Market Volatility Analysis (Reusable Version)
# Author: Aaron Isaacs
# Description: Pulls historical market data, calculates log returns, rolling volatility,
#              extreme events, plots graphs, and saves a CSV report. Works with any ticker.


#| Commodity       | Ticker |
#| --------------- | ------ |
#| Dutch TTF Gas   | TTF=F  |
#| Jet Fuel        | HO=F   |
#| Brent Crude Oil | BZ=F   |
#| WTI Crude Oil   | CL=F   |
#| Natural Gas US  | NG=F   |
#| Heating Oil     | HO=F   |

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


def analyze_market(ticker, start_date="2018-01-01"):
    # Download historical data
    data = yf.download(ticker, start=start_date)

    # Calculate log returns
    data["Returns"] = np.log(data["Close"] / data["Close"].shift(1))

    # Calculate rolling 30-day volatility
    data["Volatility_30d"] = data["Returns"].rolling(window=30).std()

    # Compute skewness and kurtosis
    skew = data["Returns"].skew()
    kurt = data["Returns"].kurt()
    print(f"{ticker} Skewness: {skew:.4f}, Kurtosis: {kurt:.4f}")

    # Identify extreme moves (3-sigma events)
    threshold = 3 * data["Returns"].std()
    extreme_up = data[data["Returns"] > threshold]
    extreme_down = data[data["Returns"] < -threshold]
    print(f"Extreme positive returns (>3σ): {len(extreme_up)}")
    print(f"Extreme negative returns (<-3σ): {len(extreme_down)}")

    # ---- Plot 1: Price Series with Extreme Events ----
    plt.figure()
    plt.plot(data["Close"], label=f"{ticker} Close Price")
    plt.scatter(extreme_up.index, data.loc[extreme_up.index, "Close"], color='r', label="Extreme Up", marker='^', s=100)
    plt.scatter(extreme_down.index, data.loc[extreme_down.index, "Close"], color='b', label="Extreme Down", marker='v',
                s=100)
    plt.title(f"{ticker} Price with Extreme Events")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.show()

    # ---- Plot 2: Returns Histogram with Normal Overlay ----
    returns = data["Returns"].dropna()
    mu, sigma = returns.mean(), returns.std()

    plt.figure()
    plt.hist(returns, bins=50, density=True, alpha=0.6, color='g')
    x = np.linspace(returns.min(), returns.max(), 100)
    plt.plot(x, norm.pdf(x, mu, sigma), 'r', lw=2)
    plt.title(f"{ticker} Log Returns Distribution")
    plt.xlabel("Log Return")
    plt.ylabel("Density")
    plt.show()

    # ---- Plot 3: Rolling Volatility (comparisons) ----
    plt.figure()
    data["Volatility_30d"].plot(title=f"{ticker} 30-Day Rolling Volatility")
    plt.xlabel("Date")
    plt.ylabel("Volatility")
    plt.show()

    # ---- Export CSV report ----
    csv_filename = f"{ticker}_analysis_report.csv"
    data[["Close", "Returns", "Volatility_30d"]].to_csv(csv_filename)
    print(f"CSV report saved: {csv_filename}")

    return data, extreme_up, extreme_down

#Comparisons

#Dutch TTF Gas
gas_data, gas_up, gas_down = analyze_market("TTF=F")
#Jet Fuel
jet_data, jet_up, jet_down = analyze_market("HO=F")
#Brent Crude Oil
brent_data, brent_up, brent_down = analyze_market("BZ=F")
#WTI Crude Oil
wti_data, wti_up, wti_down = analyze_market("CL=F")
#Natural Gas US
ng_data, ng_up, ng_down = analyze_market("NG=F")
#Heating Oil
ho_data, ho_up, ho_down = analyze_market("HO=F")

plt.figure()
gas_data['Volatility_30d'].plot(label='TTF Gas')
jet_data['Volatility_30d'].plot(label='Jet Fuel')
brent_data['Volatility_30d'].plot(label='Brent Crude')
wti_data['Volatility_30d'].plot(label='WTI Crude')
ng_data['Volatility_30d'].plot(label='US Natural Gas')
ho_data['Volatility_30d'].plot(label='Heating Oil')
plt.legend()
plt.show()
