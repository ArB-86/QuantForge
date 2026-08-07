import pandas as pd

class OrderExecutor:
    def __init__(self, portfolio_value=1000000.0):
        self.portfolio_value = portfolio_value

    def generate_orders(self, portfolio_df: pd.DataFrame) -> pd.DataFrame:
        if portfolio_df.empty:
            print('Portfolio dataframe is empty. No orders to generate.')
            return pd.DataFrame()

        print(f'Generating orders for Portfolio Value: ?{self.portfolio_value:,.2f}')
        
        orders = []
        for _, row in portfolio_df.iterrows():
            ticker = row['Ticker']
            weight = row['Weight']
            
            capital_allocated = self.portfolio_value * weight
            
            # Extract true price if available, otherwise check common price columns
            price = row.get('Close', row.get('Execution_Price', 0.0))
            if pd.isna(price) or price <= 0:
                price = 1000.0  # Fallback only if data is missing
                
            target_shares = int(capital_allocated // price)
            if target_shares <= 0:
                target_shares = 1
                
            orders.append({
                'Ticker': ticker,
                'Target_Shares': target_shares,
                'Execution_Price': price,
                'Capital_Allocated': target_shares * price,
                'Target_Weight': weight
            })
            
        return pd.DataFrame(orders)
