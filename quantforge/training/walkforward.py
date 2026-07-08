import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any
import json

from quantforge.models.factory import build  # Changed import
from quantforge.training.monthly_loop import MonthlyLoop
from quantforge.data.loader import DataLoader
from quantforge.data.date_utils import generate_walkforward_periods

logger = logging.getLogger(__name__)


class WalkForward:
    """Walk-forward backtesting framework."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.data_loader = DataLoader(config)

        self.model_manager = build(self.config)

        # Initialize monthly loop with model manager
        self.loop = MonthlyLoop(config, self.model_manager)

        self.results = []
        self.periods = None

    def _generate_periods(self):
        """Generate walk-forward periods based on configuration."""
        train_length = self.config.get('train_length', 60)  # months
        test_length = self.config.get('test_length', 1)     # months
        step_size = self.config.get('step_size', 1)         # months

        # Get date range from data
        df = self.data_loader.load()
        min_date = df['Date'].min()
        max_date = df['Date'].max()

        # Generate periods
        self.periods = generate_walkforward_periods(
            min_date=min_date,
            max_date=max_date,
            train_length=train_length,
            test_length=test_length,
            step_size=step_size,
        )

        logger.info(f"Generated {len(self.periods)} walk-forward periods")

        return self.periods

    def _save_checkpoint(self, period_idx: int, results: Dict):
        """Save checkpoint for a period."""
        checkpoint_file = Path(self.config.get('checkpoint_file', 'checkpoint.csv'))

        if checkpoint_file.exists():
            existing = pd.read_csv(checkpoint_file)
        else:
            existing = pd.DataFrame()

        # Convert period info to DataFrame
        period_data = {
            'period_idx': [period_idx],
            'train_start': [results.get('train_start')],
            'train_end': [results.get('train_end')],
            'test_start': [results.get('test_start')],
            'test_end': [results.get('test_end')],
            'return': [results.get('metrics', {}).get('return', np.nan)],
            'sharpe_ratio': [results.get('metrics', {}).get('sharpe_ratio', np.nan)],
            'max_drawdown': [results.get('metrics', {}).get('max_drawdown', np.nan)],
            'volatility': [results.get('metrics', {}).get('volatility', np.nan)],
        }

        period_df = pd.DataFrame(period_data)

        if existing.empty:
            combined = period_df
        else:
            combined = pd.concat([existing, period_df], ignore_index=True)

        combined.to_csv(checkpoint_file, index=False)
        logger.info(f"Checkpoint saved to {checkpoint_file}")

    def _save_predictions(self, results: Dict, period_idx: int):
        """Save predictions for a period."""
        pred_file = Path(self.config.get('prediction_file', 'predictions.csv'))

        if 'predictions' in results:
            predictions = results['predictions'].copy()
            predictions['period_idx'] = period_idx

            if pred_file.exists():
                existing = pd.read_csv(pred_file)
                combined = pd.concat([existing, predictions], ignore_index=True)
            else:
                combined = predictions

            combined.to_csv(pred_file, index=False)
            logger.info(f"Predictions saved to {pred_file}")

    def run(self, save_predictions: bool = True, save_checkpoints: bool = True):
        """Run the complete walk-forward analysis."""
        logger.info("Starting walk-forward analysis")

        # Generate periods if not already done
        if self.periods is None:
            self._generate_periods()

        if not self.periods:
            logger.warning("No walk-forward periods generated")
            return

        total_periods = len(self.periods)

        for idx, period in enumerate(self.periods):
            logger.info(f"Processing period {idx+1}/{total_periods}")

            try:
                train_start = period['train_start']
                train_end = period['train_end']
                test_start = period['test_start']
                test_end = period['test_end']

                # Run monthly loop for this period
                results = self.loop.run(
                    train_start=train_start,
                    train_end=train_end,
                    test_start=test_start,
                    test_end=test_end,
                )

                # Add period info to results
                results['train_start'] = train_start
                results['train_end'] = train_end
                results['test_start'] = test_start
                results['test_end'] = test_end
                results['period_idx'] = idx

                self.results.append(results)

                # Save checkpoints if requested
                if save_checkpoints:
                    self._save_checkpoint(idx, results)

                # Save predictions if requested
                if save_predictions:
                    self._save_predictions(results, idx)

            except Exception as e:
                logger.error(f"Error in period {idx+1}: {e}")
                # Continue with next period
                continue

        logger.info("Walk-forward analysis completed")

        return self.results

    def aggregate_results(self) -> Dict:
        """Aggregate results from all periods."""
        if not self.results:
            logger.warning("No results to aggregate")
            return {}

        aggregated = {
            'total_periods': len(self.results),
            'return_mean': np.mean([r['metrics'].get('return', np.nan) for r in self.results]),
            'return_std': np.std([r['metrics'].get('return', np.nan) for r in self.results]),
            'sharpe_mean': np.mean([r['metrics'].get('sharpe_ratio', np.nan) for r in self.results]),
            'max_drawdown_mean': np.mean([r['metrics'].get('max_drawdown', np.nan) for r in self.results]),
            'volatility_mean': np.mean([r['metrics'].get('volatility', np.nan) for r in self.results]),
        }

        logger.info("Aggregated results:")
        for k, v in aggregated.items():
            if isinstance(v, float):
                logger.info(f"  {k}: {v:.4f}")
            else:
                logger.info(f"  {k}: {v}")

        return aggregated

    def save_results(self, output_dir: str):
        """Save all results to disk."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save aggregate results
        aggregated = self.aggregate_results()
        with open(output_dir / 'aggregated_results.json', 'w') as f:
            json.dump(aggregated, f, indent=2, default=str)

        # Save detailed results
        for i, result in enumerate(self.results):
            period_dir = output_dir / f"period_{i:03d}"
            period_dir.mkdir(parents=True, exist_ok=True)

            # Save metrics
            with open(period_dir / 'metrics.json', 'w') as f:
                json.dump(result['metrics'], f, indent=2)

            # Save predictions
            if 'predictions' in result:
                result['predictions'].to_csv(
                    period_dir / 'predictions.csv',
                    index=False
                )

            # Save portfolio
            if 'portfolio' in result:
                result['portfolio'].to_csv(
                    period_dir / 'portfolio.csv',
                    index=False
                )

        logger.info(f"All results saved to {output_dir}")
