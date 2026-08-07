import pandas as pd
import numpy as np

def build_portfolio(oos_df: pd.DataFrame, method='inverse_vol', score_column='Prediction', top_k=10, buffer_k=15):
    portfolio_rows = []
    previous_holdings = set()
    
    for date, group in oos_df.groupby('Date'):
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
            # Inverse volatility weighting to minimize portfolio variance
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
            
        portfolio_rows.append(selected[['Date', 'Ticker', 'TARGET_5D', 'RET_1D', 'VOL_20D', 'Raw_Prediction', 'Prediction', 'Weight']])
        
    if not portfolio_rows:
        return pd.DataFrame()
        
    return pd.concat(portfolio_rows, ignore_index=True)
