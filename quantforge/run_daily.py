import os
import pandas as pd
from quantforge.dataset.loader import MarketDataLoader
from quantforge.dataset.features import FeatureEngineer
from quantforge.dataset.store import FeatureStore
from quantforge.training.engine import WalkForwardEngine
from quantforge.modeling.lightgbm import LightGBMModel
from quantforge.training.splitter import WalkForwardSplitter
from quantforge.portfolio_engine.allocator import build_portfolio
from quantforge.execution.executor import OrderExecutor
from quantforge.execution.live_gateway import LiveBrokerGateway
from quantforge.execution.kite_gateway import KiteBrokerGateway

def main():
    print('=== QUANTFORGE INSTITUTIONAL DAILY PIPELINE ===')
    
    # [1/6] Download latest market data
    loader = MarketDataLoader()
    df_raw = loader.download_all()
    loader.save(df_raw)
    
    # [2/6] Build feature store
    fe = FeatureEngineer()
    df_feat = fe.generate(df_raw)
    store = FeatureStore()
    store.save(df_feat)
    df_feat = store.load()
    
    # [3/6] Run model training and inference
    splitter = WalkForwardSplitter(n_splits=5)
    engine = WalkForwardEngine(LightGBMModel, {'verbose': -1, 'random_state': 42}, splitter)
    oos = engine.run(df_feat, fe.feature_columns, 'TARGET_5D')
    
    if oos.empty:
        print('Warning: Out-of-sample predictions dataframe is empty.')
        return
        
    # [4/6] Compute target portfolio weights across time
    portfolio = build_portfolio(
        oos, 
        method='inverse_vol', 
        score_column='Prediction', 
        top_k=10, 
        buffer_k=15, 
        rebalance_freq=5
    )
    
    # Filter strictly for the latest available rebalance date
    if 'Date' in portfolio.columns:
        latest_date = portfolio['Date'].max()
        portfolio = portfolio[portfolio['Date'] == latest_date]
        print(f'Isolating execution portfolio for latest date: {latest_date}')

    # Merge latest actual Close prices from df_feat to ensure realistic order sizing
    if 'Close' in df_feat.columns and 'Date' in df_feat.columns:
        latest_prices = df_feat[df_feat['Date'] == latest_date][['Ticker', 'Close']].drop_duplicates()
        portfolio = portfolio.merge(latest_prices, on='Ticker', how='left')

    # [5/6] Generate execution orders with true prices
    executor = OrderExecutor(portfolio_value=1000000.0)
    orders = executor.generate_orders(portfolio)
    
    # [6/6] Route orders through broker gateway
    use_live_kite = os.getenv('USE_LIVE_KITE', 'false').lower() == 'true'
    gateway = KiteBrokerGateway() if use_live_kite else LiveBrokerGateway(broker_name='paper')
    
    gateway.place_orders(orders)
    print('=== PIPELINE EXECUTION COMPLETE ===')

if __name__ == '__main__':
    main()
