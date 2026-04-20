import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pandas as pd
from typing import Dict, List


def plot_equity_curve(equity: List[float], dates: List[str] = None):
    """绘制净值曲线"""
    plt.figure(figsize=(12, 6))
    if dates:
        plt.plot(dates, equity)
    else:
        plt.plot(equity)
    plt.title('Equity Curve')
    plt.xlabel('Date')
    plt.ylabel('Equity')
    plt.grid(True)
    plt.show()


def plot_kline(data: pd.DataFrame):
    """绘制K线图"""
    fig = go.Figure(data=[go.Candlestick(
        x=data.index,
        open=data['open'],
        high=data['high'],
        low=data['low'],
        close=data['close']
    )])
    fig.update_layout(title='K Line Chart', xaxis_title='Date', yaxis_title='Price')
    fig.show()


def plot_metrics(metrics: Dict[str, float]):
    """绘制指标雷达图"""
    categories = list(metrics.keys())
    values = list(metrics.values())

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself'
    ))
    fig.update_layout(title='Performance Metrics', polar=dict(radialaxis=dict(visible=True)))
    fig.show()
