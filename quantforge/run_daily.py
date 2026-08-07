import pandas as pd
from quantforge.live_data.downloader import DataDownloader
from quantforge.dataset.loader import DataLoader
from quantforge.dataset.features import FeatureEngineer
from quantforge.feature_store.store import FeatureStore
from quantforge.dataset.splitter import WalkForwardSplitter
from quantforge.modeling.lightgbm import LightGBMModel
from quantforge.training.engine import WalkForwardEngine
from quantforge.portfolio_engine.allocator import build_portfolio
from quantforge.execution.order_generator import OrderGenerator
from quantforge.execution.live_gateway import LiveBrokerGateway

def main():
    print('=== QUANTFORGE INSTITUTIONAL DAILY PIPELINE ===')
    
    # Step 1: Download latest market data
    print('[1/6] Downloading latest market data...')
    DataDownloader().download()
    
    # Step 2: Generate features
    print('[2/6] Building feature store...')
    df = DataLoader().load()
    fe = FeatureEngineer()
    df_feat = fe.generate(df)
    fs = FeatureStore()
    fs.save(df_feat)
    
    # Step 3: Run Walk-Forward Model Inference
    print('[3/6] Running model training and inference...')
    engine = WalkForwardEngine(
        LightGBMModel, 
        {'verbose': -1, 'max_depth': 3, 'num_leaves': 7, 'learning_rate': 0.05, 'n_estimators': 50, 'colsample_bytree': 0.5, 'subsample': 0.8}, 
        WalkForwardSplitter(train_days=100, test_days=20, step_days=20)
    )
    oos = engine.run(df_feat, fe.feature_columns, 'TARGET_5D')
    
    # Step 4: Build Portfolio Target Weights (Institutional Config: Inverse Vol + Regime Filter + Hysteresis + Weekly Rebalance)
    print('[4/6] Computing target portfolio weights...')
    portfolio = build_portfolio(oos, method='inverse_vol', score_column='Prediction', top_k=10, buffer_k=15, rebalance_freq=5)
    
    if portfolio.empty:
        print('[REGIME FILTER] Market sentiment bearish. Going 100% to cash. No orders placed.')
        return
        
    latest_date = portfolio['Date'].max()
    latest_weights = portfolio[portfolio['Date'] == latest_date]
    print(f'Target weights for date: {latest_date}')
    print(latest_weights[['Ticker', 'Prediction', 'VOL_20D', 'Weight']])
    
    # Step 5: Generate Broker Orders
    print('[5/6] Generating execution orders...')
    latest_prices = df_feat[df_feat['Date'] == latest_date].set_index('Ticker')['Close'].to_dict()
    generator = OrderGenerator(portfolio_value=1000000)
    orders = generator.generate_orders(latest_weights, latest_prices)
    print(orders)
    
    # Step 6: Route via Production Broker Gateway
    print('[6/6] Routing orders through broker gateway...')
    gateway = LiveBrokerGateway(broker_name='paper')
    gateway.place_orders(orders)

if __name__ == '__main__':
    main()
