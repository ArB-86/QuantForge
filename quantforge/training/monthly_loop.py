import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime

from quantforge.data.loader import DataLoader
from quantforge.features.processor import FeatureProcessor
from quantforge.models.factory import build
from quantforge.portfolio.allocator import build_portfolio
from quantforge.backtest.simulator import simulate
from quantforge.backtest.metrics import evaluate
from quantforge.portfolio.volatility_target import VolatilityTarget

logger = logging.getLogger(__name__)


class MonthlyLoop:
    """Monthly training and backtesting loop for walk-forward analysis."""

    def __init__(self, config, model_manager=None):
        self.config = config
        self.model_manager = model_manager
        self.loader = DataLoader(config)
        self.feature_processor = FeatureProcessor(config)

    def _load_data(self, train_start, train_end, test_start, test_end):
        """Load and prepare training and test data for a specific period."""
        # Load training data
        train_df = self.loader.load(
            start_date=train_start,
            end_date=train_end,
        )

        # Load test data
        test_df = self.loader.load(
            start_date=test_start,
            end_date=test_end,
        )

        # Process features
        X_train, y_train = self.feature_processor.process(train_df)
        X_test, y_test = self.feature_processor.process(test_df)

        return X_train, y_train, X_test, y_test, test_df

    def _train_model(self, X_train, y_train):
        """Train model for the current period."""
        # Directly use model_manager as the model instance
        model = self.model_manager

        # Fit the model
        model.fit(X_train, y_train)

        return model

    def _generate_predictions(self, model, X_test, test_df):
        """Generate predictions on test data."""
        test_df = test_df.copy()

        # Make predictions
        test_df['PRED_RETURN'] = model.predict(X_test)

        return test_df

    def _backtest(self, predictions, period_info):
        """Run backtest on predictions."""
        # Build portfolio
        portfolio = build_portfolio(
            predictions,
            method=self.config.get('portfolio', 'equal_weight'),
            score_column='PRED_RETURN',
            top_n=self.config.get('top_n', 15),
            max_stock_weight=self.config.get('max_stock_weight', 1.0),
        )

        # Simulate returns
        portfolio = simulate(
            portfolio,
            return_column=self.config['target'],
            holding_days=self.config.get('holding_days', 20),
            round_trip_cost=self.config.get('transaction_cost', 0.003),
        )

        # Apply volatility targeting
        portfolio['Return'] = VolatilityTarget(
            target_vol=self.config.get('target_volatility', 0.20),
        ).apply(portfolio['Return'])

        # Recalculate equity curve
        portfolio['Equity'] = (1 + portfolio['Return']).cumprod()

        # Evaluate metrics
        metrics = evaluate(
            portfolio,
            holding_days=self.config.get('holding_days', 20),
        )

        return portfolio, metrics

    def run(self, train_start, train_end, test_start, test_end):
        """Run the complete monthly loop for a specific period."""
        logger.info(f"Running monthly loop from {train_start} to {test_end}")

        # Load data
        X_train, y_train, X_test, y_test, test_df = self._load_data(
            train_start, train_end, test_start, test_end
        )

        logger.info(f"Training data shape: {X_train.shape}")
        logger.info(f"Test data shape: {X_test.shape}")

        # Train model
        model = self._train_model(X_train, y_train)
        logger.info("Model training completed")

        # Generate predictions
        predictions = self._generate_predictions(model, X_test, test_df)
        logger.info(f"Predictions generated for {len(predictions)} rows")

        # Run backtest
        portfolio, metrics = self._backtest(predictions, {
            'train_start': train_start,
            'train_end': train_end,
            'test_start': test_start,
            'test_end': test_end,
        })

        # Log metrics
        logger.info("Backtest metrics:")
        for k, v in metrics.items():
            logger.info(f"  {k}: {v:.4f}")

        return {
            'predictions': predictions,
            'portfolio': portfolio,
            'metrics': metrics,
            'model': model,
        }
