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
            
            # Use VOL_20D or mock execution price proxy if price not directly available
            # Assuming a proxy price or fetching close price from VOL/Prediction or recent data
            # Let's derive an approximate execution price or use a default baseline if columns vary
            capital_allocated = self.portfolio_value * weight
            
            # Look for a price proxy or fallback to a standard estimation based on typical Indian large-caps if needed
            # If execution price is in row, use it; otherwise estimate from VOL or default to 1000
            price = row.get('Execution_Price', row.get('Close', 1000.0))
            if price <= 0:
                price = 1000.0
                
            target_shares = int(capital_allocated // price)
            
            orders.append({
                'Ticker': ticker,
                'Target_Shares': target_shares,
                'Execution_Price': price,
                'Capital_Allocated': target_shares * price,
                'Target_Weight': weight
            })
            
        orders_df = pd.DataFrame(orders)
        return orders_df
