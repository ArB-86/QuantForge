import pandas as pd

def build_portfolio(oos_df: pd.DataFrame, method='score_weight', score_column='Prediction', top_k=10):
    portfolio_rows = []
    
    for date, group in oos_df.groupby('Date'):
        # Sort by prediction descending
        sorted_group = group.sort_values(by=score_column, ascending=False)
        
        # Select top K stocks to diversify and reduce concentration risk
        top_stocks = sorted_group.head(top_k).copy()
        
        if len(top_stocks) == 0:
            continue
            
        if method == 'score_weight':
            # Weight proportional to predicted score, bounded and normalized
            scores = top_stocks[score_column].clip(lower=0.0)
            if scores.sum() > 0:
                top_stocks['Weight'] = scores / scores.sum()
            else:
                top_stocks['Weight'] = 1.0 / len(top_stocks)
        else:
            # Equal weight across top K
            top_stocks['Weight'] = 1.0 / len(top_stocks)
            
        portfolio_rows.append(top_stocks[['Date', 'Ticker', 'TARGET_5D', 'RET_1D', 'VOL_20D', 'Raw_Prediction', 'Prediction', 'Weight']])
        
    if not portfolio_rows:
        return pd.DataFrame()
        
    portfolio_df = pd.concat(portfolio_rows, ignore_index=True)
    return portfolio_df
