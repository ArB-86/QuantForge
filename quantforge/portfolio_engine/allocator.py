import pandas as pd
import numpy as np

def build_portfolio(oos_df: pd.DataFrame, method='inverse_vol', score_column='Prediction', top_k=10, buffer_k=15, rebalance_freq=5):
    portfolio_rows = []
    previous_holdings = set()
    current_weights = pd.Series(dtype=float)
    
    unique_dates = sorted(oos_df['Date'].unique())
    
    for i, date in enumerate(unique_dates):
        group = oos_df[oos_df['Date'] == date]
        
        # Rebalance only every 'rebalance_freq' days, otherwise carry forward previous weights
        if i % rebalance_freq != 0 and not current_weights.empty and previous_holdings:
            # Carry forward weights for tickers still in previous holdings
            held_group = group[group['Ticker'].isin(previous_holdings)].copy()
            if not held_group.empty:
                # Maintain proportional weights from last rebalance
                held_group['Weight'] = [current_weights.get(t, 0.0) for t in held_group['Ticker']]
                total_w = held_group['Weight'].sum()
                if total_w > 0:
                    held_group['Weight'] /= total_w
                else:
                    held_group['Weight'] = 1.0 / len(held_group)
                portfolio_rows.append(held_group[['Date', 'Ticker', 'TARGET_5D', 'RET_1D', 'VOL_20D', 'Raw_Prediction', 'Prediction', 'Weight']])
                continue

        sorted_group = group.sort_values(by=score_column, ascending=False).reset_index(drop=True)
        sorted_group['Rank'] = sorted_group.index + 1
        
        if not previous_holdings:
            selected = sorted_group[sorted_group['Rank'] <= top_k].copy()
        else:
            is_held = sorted_group['Ticker'].isin(previous_holdings)
            is_qualified = sorted_group['Rank'] <= buffer_k
            selected_mask = is_held & is_qualified
            
            num_needed = top_k - selected_mask.sum()
            if num_needed > 0:
                unheld_mask = ~selected_mask
                top_unheld = sorted_group[unheld_mask].head(num_needed)
                selected_mask = selected_mask | sorted_group['Ticker'].isin(top_unheld['Ticker'])
                
            selected = sorted_group[selected_mask].copy()
            if len(selected) > top_k:
                selected = selected.head(top_k)
                
        previous_holdings = set(selected['Ticker'])
        
        if len(selected) == 0:
            continue
            
        if method == 'inverse_vol':
            inv_vol = 1.0 / selected['VOL_20D'].clip(lower=0.01)
            selected['Weight'] = inv_vol / inv_vol.sum()
        elif method == 'score_weight':
            scores = selected[score_column].clip(lower=0.0)
            if scores.sum() > 0:
                selected['Weight'] = scores / scores.sum()
            else:
                selected['Weight'] = 1.0 / len(selected)
        else:
            selected['Weight'] = 1.0 / len(selected)
            
        current_weights = pd.Series(selected['Weight'].values, index=selected['Ticker'].values)
        portfolio_rows.append(selected[['Date', 'Ticker', 'TARGET_5D', 'RET_1D', 'VOL_20D', 'Raw_Prediction', 'Prediction', 'Weight']])
        
    if not portfolio_rows:
        return pd.DataFrame()
        
    return pd.concat(portfolio_rows, ignore_index=True)
