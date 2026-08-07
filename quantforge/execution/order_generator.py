import pandas as pd

class OrderGenerator:
    def __init__(self, portfolio_value: float):
        self.portfolio_value = portfolio_value

    def generate_orders(self, target_weights: pd.DataFrame, current_prices: dict) -> pd.DataFrame:
        print(f'Generating orders for Portfolio Value: ?{self.portfolio_value:,.2f}')
        orders = []
        for _, row in target_weights.iterrows():
            ticker = row['Ticker']
            weight = row['Weight']
            
            if ticker not in current_prices:
                print(f'WARNING: No current price for {ticker}. Skipping.')
                continue
                
            price = current_prices[ticker]
            target_value = self.portfolio_value * weight
            target_shares = int(target_value // price)  # Floor division for full shares
            
            orders.append({
                'Ticker': ticker,
                'Target_Shares': target_shares,
                'Execution_Price': price,
                'Capital_Allocated': target_shares * price,
                'Target_Weight': weight
            })
        
        return pd.DataFrame(orders)
