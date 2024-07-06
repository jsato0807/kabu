import numpy as np

def traripi_backtest_debug(data, initial_funds, grid_start, grid_end, num_traps, profit_width, order_size, strategy='standard', density=1):
    """
    Perform Trailing Stop strategy backtest on given data.
    """
    effective_margin = initial_funds
    realized_profit = 0
    positions = []
    trades = []

    if strategy == 'long_only':
        grids = np.linspace(grid_start, grid_end, num=num_traps)
        last_price = None  # Variable to store the last processed price

        for i in range(len(data)):
            date = data.index[i]
            price = data.iloc[i]

            if last_price is not None:
                # Check if price has crossed any grid between last_price and price
                if price > last_price:
                    for grid in grids:
                        if last_price <= grid < price:
                            if effective_margin >= order_size * grid:
                                effective_margin -= order_size * grid
                                positions.append([order_size, i, 'Buy', grid, -grid * order_size])  # Add initial unrealized profit as 0
                                trades.append((date, price, 'Buy'))
                                break  # Exit loop once position is taken
                elif price < last_price:
                    for grid in grids:
                        if last_price >= grid > price:
                            if effective_margin >= order_size * grid:
                                effective_margin -= order_size * grid
                                positions.append([order_size, i, 'Buy', grid, -grid * order_size])  # Add initial unrealized profit as 0
                                trades.append((date, price, 'Buy'))
                                break  # Exit loop once position is taken

            # Position closure processing
            for pos in positions[:]:
                if pos[2] == 'Buy' and pos[1] < len(data) - 1:
                    if price - pos[3] >= profit_width:
                        # Calculate profit for closed positions
                        profit = order_size * profit_width * (abs(price - pos[3]) // profit_width)
                        effective_margin += profit + order_size * pos[3]		#sell back the position you got in the process 'Check if price has crossed any grid between last_price and price'
                        realized_profit += profit
                        pos[2] = 'Sell-Closed'
                        trades.append((date, price, 'Sell'))
                        positions.remove(pos)  # Remove the closed position
                    else:
                        # Update unrealized profit for open positions
                        unrealized_profit = order_size * (price - pos[3])
                        effective_margin += unrealized_profit -  pos[4]  # Adjust for previous unrealized profit
                        pos[4] = unrealized_profit  # Store current unrealized profit in the position

            # Update last_price for the next iteration
            last_price = price

    # 他の戦略も同様に修正してください...

    # Calculate position value
    position_value = sum(size * (data.iloc[-1] - grid) if 'Buy' in status and not status.endswith('Closed') else
                         -size * (data.iloc[-1] - grid) if 'Sell' in status and not status.endswith('Closed') else
                         0 for size, _, status, grid, _ in positions)

    # Calculate effective margin including position value
    effective_margin_with_positions = effective_margin

    # Calculate margin deposit
    margin_deposit = initial_funds + realized_profit

    return effective_margin_with_positions, margin_deposit, realized_profit, position_value, trades
