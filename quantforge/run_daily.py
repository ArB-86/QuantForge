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

def main():
    print('=== QUANTFORGE DAILY EXECUTION PIPELINE ===')
    
    # Step 1: Download latest market data
    print('[1/5] Downloading latest market data...')
    DataDownloader().download()
    
    # Step 2: Generate features
    print('[2/5] Building feature store...')
    df = DataLoader().load()
    fe = FeatureEngineer()
    df_feat = fe.generate(df)
    fs = FeatureStore()
    fs.save(df_feat)
    
    # Step 3: Run Walk-Forward Model Inference
    print('[3/5] Running model training and inference...')
    engine = WalkForwardEngine(
        LightGBMModel, 
        {'verbose': -1, 'max_depth': 3, 'num_leaves': 7, 'learning_rate': 0.05, 'n_estimators': 50, 'colsample_bytree': 0.5, 'subsample': 0.8}, 
        WalkForwardSplitter(train_days=100, test_days=20, step_days=20)
    )
    oos = engine.run(df_feat, fe.feature_columns, 'TARGET_5D')
    
    # Step 4: Build Portfolio Target Weights
    print('[4/5] Computing target portfolio weights...')
    portfolio = build_portfolio(oos, method='score_weight', score_column='Prediction')
    latest_date = portfolio['Date'].max()
    latest_weights = portfolio[portfolio['Date'] == latest_date]
    print(f'Target weights for date: {latest_date}')
    print(latest_weights)
    
    # Step 5: Generate Broker Orders (Mock live prices for testing)
    print('[5/5] Generating execution orders...')
    latest_prices = df_feat[df_feat['Date'] == latest_date].set_index('Ticker')['Close'].to_dict()
    generator = OrderGenerator(portfolio_value=1000000)
    orders = generator.generate_orders(latest_weights, latest_prices)
    
    print('\n--- FINAL EXECUTION ORDERS ---')
    print(orders)

if __name__ == '__main__':
    main()
