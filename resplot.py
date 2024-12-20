import backtrader as bt
import numpy as np
import yfinance as yf
from collections import defaultdict
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, FileResponse
import uvicorn
import matplotlib
import matplotlib.pyplot as plt
import os
import io

app = FastAPI()
matplotlib.use('Agg')


class QLearningStrategy(bt.Strategy):
    params = dict(
        alpha=0.1,
        gamma=0.99,
        rsi_period=14,
        macd1=12,
        macd2=26,
        macdsig=9,
        vol_period=20,
        vol_bins=[-1.0, -0.5, 0.0, 0.5, 1.0],
        rsi_bins=[0, 30, 50, 70, 100],
        macd_bins=[-2.0, -1.0, 0.0, 1.0, 2.0],
        epsilon=1e-6,
        base_volatility=0.5,  # Base volatility for scaling
        shift=1
    )

    def __init__(self, q_table=None):
        # Initialize indicators
        self.rsi = bt.indicators.RSI(period=self.params.rsi_period)
        self.macd = bt.indicators.MACD(
            period_me1=self.params.macd1, period_me2=self.params.macd2, period_signal=self.params.macdsig)
        self.vol_ma = bt.indicators.SMA(self.data.volume, period=self.params.vol_period)
        self.normalized_vol = (self.data.volume - self.vol_ma) / self.vol_ma

        # Initialize Q-table
        num_actions = 3  # [Hold, Buy, Sell]
        initial_q_value = 1 / num_actions
        self.q_table = q_table if q_table is not None else defaultdict(lambda: np.full(num_actions, initial_q_value))
        self.prev_state = None
        self.prev_action = None
        self.prev_portfolio_value = None
        self.starting_cash = None
        self.realized_profit = 0
        self.unrealized_profit = 0
        self.buy_price = None
        self.buy_size = None
        self.volatility_sum = 0
        self.trade_count = 0

    def get_state(self):
        vol_bin = np.digitize(self.normalized_vol[0], self.params.vol_bins) - 1
        rsi_bin = np.digitize(self.rsi[0], self.params.rsi_bins) - 1
        macd_bin = np.digitize(self.macd.macd[0], self.params.macd_bins) - 1
        return (vol_bin, rsi_bin, macd_bin)

    def calculate_volatility(self):
        # Price changes over the last 10 periods
        recent_prices = [self.data.close[i] for i in range(-10, 0)]
        price_changes = np.diff(recent_prices)
        price_volatility = np.std(price_changes)
    
        # Volume changes over the last 10 periods
        recent_volumes = [self.data.volume[i] for i in range(-10, 0)]
        volume_changes = np.diff(recent_volumes)
        volume_volatility = np.std(volume_changes)
    
        # Combine price and volume volatilities
        combined_volatility = (price_volatility + volume_volatility) / 2
    
        # Normalize volatility using sigmoid function
        scaled_volatility = 1 / (1 + np.exp(-combined_volatility))
    
        return scaled_volatility


    def next(self):
        if self.starting_cash is None:
            self.starting_cash = self.data.close[0] * 100  # Initial cash is 100 times the first closing price
            self.broker.set_cash(self.starting_cash)
            self.prev_portfolio_value = self.broker.getvalue()

        state = self.get_state()
        self.volatility = self.calculate_volatility()
        self.volatility_sum += self.volatility  # Accumulate volatility for averaging

        # Determine number of actions to consider based on volatility
        num_actions = len(self.q_table[state])
        num_available_actions = max(1, int(self.volatility * num_actions))

        # Sort actions by Q-values
        sorted_actions = np.argsort(self.q_table[state])[::-1]
        chosen_actions = sorted_actions[:num_available_actions]

        # Get Q-values for the chosen actions
        q_values = self.q_table[state][chosen_actions]
        q_values = np.maximum(q_values, 0)  # Ensure Q-values are non-negative
        total_q = np.sum(q_values)
        if total_q == 0:
            probabilities = np.ones(len(q_values)) / len(q_values)  # Uniform distribution
        else:
            probabilities = q_values / total_q

        # Roulette wheel selection
        action = np.random.choice(chosen_actions, p=probabilities)

        if self.prev_state is not None:
            reward = self.calculate_reward()
            self.q_table[self.prev_state][self.prev_action] += self.params.alpha * (
                reward + self.params.gamma * np.max(self.q_table[state]) - self.q_table[self.prev_state][self.prev_action])

        self.prev_state = state
        self.prev_action = action

        # Execute action based on selection
        if action == 1 and self.broker.get_cash() > self.data.close[0]:
            self.buy(size=20)
            self.buy_price = self.data.close[0]
            self.buy_size = 20
            self.trade_count += 1
        elif action == 2 and self.position:
            sell_price = self.data.close[0]
            self.sell(size=20)
            if self.buy_price is not None:
                profit = (sell_price - self.buy_price) * self.buy_size
                self.realized_profit += profit
                self.trade_count += 1
        # No action for Hold (action == 0)

        # Update unrealized profit
        if self.position:
            self.unrealized_profit = (self.data.close[0] - self.buy_price) * self.buy_size

        self.prev_portfolio_value = self.broker.getvalue()

    def calculate_reward(self):
        current_portfolio_value = self.broker.getvalue()
        reward = current_portfolio_value - self.prev_portfolio_value
        return reward

    def stop(self):
        final_portfolio_value = self.broker.getvalue()
        initial_portfolio_value = self.starting_cash

        realized_profit_pct = (self.realized_profit / initial_portfolio_value) * 100
        unrealized_profit_pct = (self.unrealized_profit / initial_portfolio_value) * 100
        # portfolio_pnl = (final_portfolio_value - initial_portfolio_value) / initial_portfolio_value * 100

        sum_realized_unrealized_pct = realized_profit_pct + unrealized_profit_pct
        avg_volatility = self.volatility_sum / len(self)

        print(f"Initial Portfolio Value: {initial_portfolio_value:.2f}")
        print(f"Final Portfolio Value: {final_portfolio_value:.2f}")
        print(f"Total Realized Profit: {self.realized_profit:.2f} ({realized_profit_pct:.2f}%)")
        print(f"Total Unrealized Profit: {self.unrealized_profit:.2f} ({unrealized_profit_pct:.2f}%)")
        print(f"Sum of Realized and Unrealized Profit Percentages: {sum_realized_unrealized_pct:.2f}%")
        print(f"Average Volatility: {avg_volatility:.2f}")

        self.total_pnl = sum_realized_unrealized_pct

# Define Buy and Hold strategy
class BuyAndHoldStrategy(bt.Strategy):
    def next(self):
        if not self.position:
            self.buy(size=100)

    def stop(self):
        final_portfolio_value = self.broker.getvalue()
        initial_portfolio_value = self.broker.startingcash
        pnl = (final_portfolio_value - initial_portfolio_value) / initial_portfolio_value * 100
        #print(f"Buy and Hold Strategy Initial Portfolio Value: {initial_portfolio_value:.2f}")
        #print(f"Buy and Hold Strategy Final Portfolio Value: {final_portfolio_value:.2f}")
        print(f"Buy and Hold Strategy PnL: {pnl:.2f}%")

@app.post("/run-strategy")
async def run_strategy(ticker: str = "HCC.NS", date: str = "2024-09-23", iterations: int = 50):
    # Get data from yfinance
    end_date = "2024-09-26"
    data = yf.download(ticker, start=date, end=end_date, interval="1m")

    if data.empty:
        raise HTTPException(status_code=400, detail="No data found for the given ticker and date")

    data_feed = bt.feeds.PandasData(dataname=data)

    best_qlearning_pnl = -np.inf
    best_strategy = None
    best_plot_filename = None

    # Directory for storing plots
    plot_dir = os.path.join(os.getcwd(), 'plots')
    os.makedirs(plot_dir, exist_ok=True)

    # Run the Q-learning strategy multiple times
    for i in range(iterations):
        q_table = defaultdict(lambda: np.full(3, 1/3))  # 3 possible actions

        cerebro = bt.Cerebro()
        cerebro.addstrategy(QLearningStrategy, q_table=q_table)
        cerebro.adddata(data_feed)
        cerebro.run()

        strategy = cerebro.run()[0]
        qlearning_pnl = strategy.total_pnl

        if qlearning_pnl > best_qlearning_pnl:
            # Update the best PnL
            best_qlearning_pnl = qlearning_pnl
            best_strategy = strategy

            # Generate plot for the best strategy so far
            fig = cerebro.plot(style='candlestick',iplot=False)[0][0]
            img = io.BytesIO()
            fig.savefig(img, format='png')
            img.seek(0)

            # Save plot image in 'plots' directory
            best_plot_filename = f"{ticker}_{date}.png"
            plot_path = os.path.join(plot_dir, best_plot_filename)
            with open(plot_path, 'wb') as f:
                f.write(img.read())

            img.seek(0)  # Reset buffer for FastAPI to send

    # Run Buy and Hold Strategy for comparison
    cerebro_bh = bt.Cerebro()
    cerebro_bh.addstrategy(BuyAndHoldStrategy)
    cerebro_bh.adddata(data_feed)
    cerebro_bh.broker.set_cash(data['Close'].iloc[0] * 100)
    cerebro_bh.run()

    bh_initial_value = data['Close'].iloc[0] * 100
    bh_final_value = cerebro_bh.broker.getvalue()
    bh_pnl = (bh_final_value - bh_initial_value) / bh_initial_value * 100

    # If no plot was generated, return a warning
    if best_plot_filename is None:
        raise HTTPException(status_code=500, detail="Failed to generate any plots")

    response = {
        "best_qlearning_pnl": best_qlearning_pnl,
        "buy_and_hold_pnl": bh_pnl,
        "plot_image": f"/plot/{best_plot_filename}"
    }

    return JSONResponse(content=response)


@app.get("/plot/{filename}")
async def get_plot(filename: str):
    plot_dir = os.path.join(os.getcwd(), 'plots')
    plot_path = os.path.join(plot_dir, filename)

    if os.path.exists(plot_path):
        return FileResponse(plot_path, media_type='image/png')
    else:
        raise HTTPException(status_code=404, detail="Plot not found")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)


