import pandas as pd
from datetime import datetime
import numpy as np
from QuantConnect.Data.Custom.TradingEconomics import *
# from risk import inverse_correlation, portfolio_volatility

class OpenGap(QCAlgorithm):

    def Initialize(self):

        #1. Required: Five years of backtest history
        self.SetStartDate(2014, 10, 1)
        # self.SetEndDate(2016, 1, 1)
    
        #2. Required: Alpha Streams Models:
        self.SetBrokerageModel(BrokerageName.AlphaStreams)
    
        #3. Required: Significant AUM Capacity
        self.SetCash(1000000)

        #4. Required: Benchmark to SPY
        self.SetBenchmark("SPY")
        
        cash = ['BIL','SHV']
        self.short_term = ['SPTS','SHY','VGSH','SCHO']
        self.med_term = ['IEF','IEI','VGIT','SCHR']
        self.long_term = ['SPTL','VGLT','TLH','EDV','TLT']
        
        leveraged_short = ['TBF','TBT','TMV']
        mis = ['GOVT']
        # self.symbols = ['IEF','SHY','TLT','SHV','IEI','TLH','EDV','BIL','SPTL','TBT','TMF','TMV','TBF','VGSH','VGIT','VGLT','SCHO','SCHR','SPTS','GOVT']
        # self.symbols = ['TBF','TBT','TMV']
        # self.symbols = cash + short_term + mis
        # self.symbols = long_term
        self.symbols = self.short_term + self.med_term + self.long_term
    
        self.AddEquity("SPY", Resolution.Minute)
        # Add Equity ------------------------------------------------
        self.universe = []
        for symbol in self.symbols:
            asset = self.AddEquity(str(symbol), Resolution.Minute)
            self.universe.append(asset.Symbol)
            # if symbol in short_term:
            #     self.short_term_symbols.append(asset.Symbol)
            # if symbol in med_term:
            #     self.med_term_symbols.append(asset.Symbol)
            # if symbol in long_term:
            #     self.long_term_symbols.append(asset.Symbol)
            
        us = TradingEconomics.Calendar.UnitedStates
        self.short_term_yield = self.AddData(TradingEconomicsCalendar, us.TwoYearNoteYield).Symbol
        self.med_term_yield = self.AddData(TradingEconomicsCalendar, us.FiveYearNoteYield).Symbol
        self.long_term_yield = self.AddData(TradingEconomicsCalendar, us.ThirtyYearBondYield).Symbol
        # self.rates = self.AddData(TradingEconomicsCalendar, TradingEconomics.Calendar.UnitedStates.InterestRate).Symbol
        
        self.Schedule.On(
            self.DateRules.EveryDay("SPY"),
            # self.DateRules.Every(DayOfWeek.Monday, DayOfWeek.Monday),
            # self.DateRules.Every(DayOfWeek.Tuesday, DayOfWeek.Tuesday),
            # self.DateRules.Every(DayOfWeek.Wednesday, DayOfWeek.Wednesday),
            # self.DateRules.Every(DayOfWeek.Thursday, DayOfWeek.Thursday),
            # self.DateRules.Every(DayOfWeek.Friday, DayOfWeek.Friday),
            # self.DateRules.MonthStart("SPY"),
            # self.DateRules.WeekStart("SPY"),
            # self.DateRules.WeekEnd("SPY"),
            self.TimeRules.AfterMarketOpen("SPY", 30),
            Action(self.Rebalance)
        )
        
    def OnData(self, data):
        pass

    def Exit(self):
        for tkr in self.Portfolio.Values:
            if (tkr.Invested): 
                self.EmitInsights(Insight.Price(tkr.Symbol, timedelta(1), InsightDirection.Flat))
                self.Liquidate(tkr.Symbol)
        
    def Execution(self, weights):
        string_symbols = [sym.split(" ")[0] for sym in weights.keys()]
        # string_symbols = weights.keys()
        for tkr in self.Portfolio.Values:
            if (tkr.Invested) and str(tkr.Symbol) not in string_symbols: 
                self.EmitInsights(Insight.Price(tkr.Symbol, timedelta(1), InsightDirection.Flat))
                self.Liquidate(tkr.Symbol)
        
        for sym, weight in weights.items():
            if abs(weight) < 0.001:
                if self.Portfolio[sym].Invested:
                    self.EmitInsights(Insight.Price(sym, timedelta(1), InsightDirection.Flat))
                    self.Liquidate(sym)
            else:
                if weight < 0:
                    if self.Portfolio[sym].Invested:
                        self.Log("Error: NEGATIVE Weight for Symbol {}".format(sym))
                        self.SetHoldings(sym, 0)
                else:
                    self.EmitInsights(Insight.Price(sym, timedelta(1), InsightDirection.Up, weight, weight))
                    self.SetHoldings(sym, weight)
                
    def yield_to_signal_df(self, yield_data, new_index, time_length):
        yield_data = yield_data.resample('1D').mean()
        yield_data = yield_data.set_index(yield_data.index.strftime("%Y-%m-%d")).dropna(how='all')
        
        # # forward fill missing dates in between
        yield_data.index = pd.DatetimeIndex(yield_data.index)
        yield_data = yield_data.reindex(new_index).ffill()
    
        # rename columns to be the same
        yield_data = yield_data.rename(columns={yield_data.actual.columns[0]: "rate"})
        yield_data = (1+yield_data.actual)**(1/time_length) - 1
        return yield_data
    
    def Rebalance(self):
        daily_historical = self.History(self.symbols, 300, Resolution.Daily)
        short_term_symbols = self.History(self.short_term, 300, Resolution.Daily).unstack(level=0).close.columns.values
        med_term_symbols = self.History(self.med_term, 300, Resolution.Daily).unstack(level=0).close.columns.values
        long_term_symbols = self.History(self.long_term, 300, Resolution.Daily).unstack(level=0).close.columns.values
        daily_historical_month = self.History(self.universe, 20, Resolution.Daily)
        
        short_term_yield = self.History(TradingEconomicsCalendar, self.short_term_yield, 200, Resolution.Daily).unstack(level=0)
        med_term_yield = self.History(TradingEconomicsCalendar, self.med_term_yield, 200, Resolution.Daily).unstack(level=0)
        long_term_yield = self.History(TradingEconomicsCalendar, self.long_term_yield, 200, Resolution.Daily).unstack(level=0)
        
        if len(short_term_yield.values) == 0 or len(med_term_yield.values) == 0 or len(long_term_yield.values) == 0:
            return
        
        daily_returns = daily_historical.unstack(level=0).pct_change().dropna(how='all').close
        prices = daily_historical.unstack(level=0)
        # signal = prices.pct_change(3*20).dropna(how='all').close
        
        ##################### PRICE SIGNAL HERE
        ## 
        ## 
        #####################
        #monthly_returns = prices.close.pct_change(20)
        weekly_returns = prices.close.pct_change(200)
        #weekly_returns = monthly_returns - weekly_returns
        
        price_signal = weekly_returns.stack(level=0).unstack(level=0).rank() # 1-10
        ranked_price_signal = price_signal.stack(level=0).unstack(level=0)
        ranked_price_signal = ranked_price_signal.dropna(how='all')
        # self.Log(price_signal)
        # self.Log(ranked_price_signal)
        
        mid_rank = int(len(ranked_price_signal.columns.values) * 0.5) + 1
        # Bot Quantile since higher rank is worse
        # ranked_price_signal = ranked_price_signal[ranked_price_signal <= mid_rank]
        
        new_index = pd.date_range(start=prices.index.min(), end=prices.index.max())
        
        ##################### YIELD RATE LENGTHS
        ## 
        ## 
        #####################
        short_term_yield = self.yield_to_signal_df(short_term_yield, new_index, 2)
        med_term_yield = self.yield_to_signal_df(med_term_yield, new_index, 5)
        long_term_yield = self.yield_to_signal_df(long_term_yield, new_index, 30)

        signal = short_term_yield
        for symbol in daily_returns.columns.values:
            if symbol in short_term_symbols:
                signal[symbol] = short_term_yield[signal.columns[0]]
            if symbol in med_term_symbols:
                signal[symbol] = med_term_yield[signal.columns[0]]
            if symbol in long_term_symbols:
                signal[symbol] = long_term_yield[signal.columns[0]]

        signal = signal.drop(columns=[signal.columns[0]])

        signal = signal.stack(level=0).unstack(level=0).rank(method='min') # 1-10
        signal = signal.stack(level=0).unstack(level=0)
        signal = signal.dropna(how='all')

        # signal = signal[signal <= mid_rank]
        
        # signal += ranked_price_signal
        
        weights = {}
        threshold = 0 # top 5
        total_weight = 0.
        symbol_list = signal.columns.values
        vol_target = 0.20
        total_vol = 0.0
        portfolio_target = 0.3
        ann_vol = {}
        
        for i, symbol in enumerate(symbol_list):
            weights[symbol] = 0.0
            ann_vol[symbol] = 1.0
            
            if symbol not in signal.columns or symbol not in ranked_price_signal.columns:
                continue
            if len(signal[symbol].values) == 0 or len(ranked_price_signal[symbol].values) == 0:
                continue
            asset_signal = signal[symbol].values[-1]
            ranked_signal = ranked_price_signal[symbol].values[-1]
            if True:
                ##################### CHANGE QUANTILE MID RANK FILTER
                ## 
                ## 
                #####################
                if asset_signal <= mid_rank and ranked_signal <= mid_rank:
                # if np.abs(asset_signal) < threshold:
                # if np.abs(asset_signal) > threshold:
                    annualized_vol = daily_returns.std().loc[symbol] * (252 ** 0.5)
                    adjusted_vol = (vol_target / annualized_vol)
                    ann_vol[symbol] = annualized_vol
                    weights[symbol] = 1 #* annualized_vol
            total_weight += weights[symbol]
        # weights = inverse_correlation(weights, daily_historical.unstack(level=0).close)
        # weights = portfolio_volatility(daily_historical_month, weights, target_volatility=0.30)
        
        # self.Log(weights)
        self.Execution(weights)
