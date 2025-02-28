import datetime
import sys
import json
import argparse
import os

sys.path.append("./Live-Tools-V2")

import asyncio
from utilities.hyperliquid_perp import PerpHyperliquid
from secret import ACCOUNTS
import ta

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

def load_config(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

async def main():
    parser = argparse.ArgumentParser(description='Load pairs configuration from a JSON file.')
    parser.add_argument('--pairs', default='pairs.json', type=str, help='The path to the JSON configuration pairs file (default: pairs.json)')
    parser.add_argument('--account', default='hyperliquid1', type=str, help='The name of the hyperliquid (sub)account')
    args = parser.parse_args()
    script_dir = os.path.dirname(os.path.realpath(__file__))
    root_dir = os.path.abspath(os.path.join(script_dir, '../../'))
    # Si un fichier est spécifié, vérifier s'il est relatif ou absolu
    if not os.path.isabs(args.pairs):
        # Construire le chemin complet depuis le répertoire du script
        args.pairs = os.path.join(root_dir, args.pairs)
    params = load_config(args.pairs)
    
    try:
        account = ACCOUNTS[args.account]
    except KeyError:
        print(f"Account '{args.account}' not found in the secret.py")
        return
    print(f"Account Name: {args.account}")

    margin_mode = "isolated"  # isolated or crossed
    leverage = 10
    hedge_mode = True  # Warning, set to False if you are in one way mode

    tf = "15m"
    sl = 0.3

    exchange = PerpHyperliquid(
        public_adress=account["public_adress"],
        private_key=account["private_key"],
    )
    invert_side = {"long": "sell", "short": "buy"}
    print(
        f"--- Execution started at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---"
    )
    try:
        markets = await exchange.load_markets()
        for pair in params.copy():
            info = exchange.get_pair_info(pair)
            if info is None:
                print(f"Pair {pair} not found, removing from params...")
                del params[pair]

        pairs = list(params.keys())

        try:
            print(f"Setting {margin_mode} x{leverage} on {len(pairs)} pairs...")
            tasks = [
                exchange.set_margin_mode_and_leverage(pair, margin_mode, leverage)
                for pair in pairs
            ]
            await asyncio.gather(*tasks)  # set leverage and margin mode for all pairs
        except Exception as e:
            print(e)

        print(f"Getting data and indicators on {len(pairs)} pairs...")
        tasks = [exchange.get_last_ohlcv(pair, tf, 50) for pair in pairs]
        dfs = await asyncio.gather(*tasks)
        df_list = dict(zip(pairs, dfs))

        for pair in df_list:
            current_params = params[pair]
            df = df_list[pair]
            if current_params["src"] == "close":
                src = df["close"]
            elif current_params["src"] == "ohlc4":
                src = (df["close"] + df["high"] + df["low"] + df["open"]) / 4

            df["ma_base"] = ta.trend.sma_indicator(
                close=src, window=current_params["ma_base_window"]
            )
            high_envelopes = [
                round(1 / (1 - e) - 1, 3) for e in current_params["envelopes"]
            ]
            for i in range(1, len(current_params["envelopes"]) + 1):
                df[f"ma_high_{i}"] = df["ma_base"] * (1 + high_envelopes[i - 1])
                df[f"ma_low_{i}"] = df["ma_base"] * (
                    1 - current_params["envelopes"][i - 1]
                )

            df_list[pair] = df

        usdc_balance = await exchange.get_balance()
        usdc_balance = usdc_balance.total
        print(f"Balance: {round(usdc_balance, 2)} USD")
        
        tasks = [exchange.get_open_trigger_orders(pair) for pair in pairs]
        print(f"Getting open trigger orders...")
        trigger_orders = await asyncio.gather(*tasks)
        print("trigger_orders:", trigger_orders)
        trigger_order_list = dict(
            zip(pairs, trigger_orders)
        )  # Get all open trigger orders by pair

        tasks = []
        for pair in df_list:
            params[pair]["canceled_orders_buy"] = len(
                [
                    order
                    for order in trigger_order_list[pair]
                    if (order.side == "buy" and order.reduce is False)
                ]
            )
            params[pair]["canceled_orders_sell"] = len(
                [
                    order
                    for order in trigger_order_list[pair]
                    if (order.side == "sell" and order.reduce is False)
                ]
            )
            tasks.append(
                exchange.cancel_trigger_orders(
                    pair, [order.id for order in trigger_order_list[pair]]
                )
            )
        print(f"Canceling trigger orders...")
        await asyncio.gather(*tasks)  # Cancel all trigger orders
        
        tasks = [exchange.get_open_orders(pair) for pair in pairs]
        print(f"Getting open orders...")
        orders = await asyncio.gather(*tasks)
        print("orders", orders)
        order_list = dict(zip(pairs, orders))  # Get all open orders by pair

        print("order_list", order_list)
        tasks = []
        for pair in df_list:
            print("pair", pair)
            params[pair]["canceled_orders_buy"] = params[pair][
                "canceled_orders_buy"
            ] + len(
                [
                    order
                    for order in order_list[pair]
                    if (order.side == "buy" and order.reduce is False)
                ]
            )
            params[pair]["canceled_orders_sell"] = params[pair][
                "canceled_orders_sell"
            ] + len(
                [
                    order
                    for order in order_list[pair]
                    if (order.side == "sell" and order.reduce is False)
                ]
            )
            tasks.append(
                exchange.cancel_orders(pair, [order.id for order in order_list[pair]])
            )

        print(f"Canceling limit orders...")
        await asyncio.gather(*tasks)  # Cancel all orders

        print(f"Getting live positions...")
        positions = await exchange.get_open_positions(pairs)
        print("positions:", positions)
        tasks_close = []
        tasks_open = []
        for position in positions:
            print(
                f"Current position on {position.pair} {position.side} - {position.size} ~ {position.usd_size} $"
            )
            row = df_list[position.pair].iloc[-2]
            print("margin_mode:", margin_mode, "row[ma_base]:", row["ma_base"])
            tasks_close.append(
                exchange.place_order(
                    pair=position.pair,
                    side=invert_side[position.side],
                    price=row["ma_base"],
                    size=exchange.amount_to_precision(position.pair, position.size),
                    type="limit",
                    reduce=True,
                    margin_mode=margin_mode,
                    hedge_mode=hedge_mode,
                    error=False,
                )
            )
            if position.side == "long":
                sl_side = "sell"
                sl_price = exchange.price_to_precision(
                    position.pair, position.entry_price * (1 - sl)
                )
            elif position.side == "short":
                sl_side = "buy"
                sl_price = exchange.price_to_precision(
                    position.pair, position.entry_price * (1 + sl)
                )
            tasks_close.append(
                exchange.place_trigger_order(
                    pair=position.pair,
                    side=sl_side,
                    trigger_price=sl_price,
                    price=sl_price,
                    size=exchange.amount_to_precision(position.pair, position.size),
                    type="market",
                    reduce=True,
                    margin_mode=margin_mode,
                    hedge_mode=hedge_mode,
                    error=False,
                )
            )
            for i in range(
                len(params[position.pair]["envelopes"])
                - params[position.pair]["canceled_orders_buy"],
                len(params[position.pair]["envelopes"]),
            ):
                tasks_open.append(
                    exchange.place_trigger_order(
                        pair=position.pair,
                        side="buy",
                        price=exchange.price_to_precision(
                            position.pair, row[f"ma_low_{i+1}"]
                        ),
                        trigger_price=exchange.price_to_precision(
                            position.pair, row[f"ma_low_{i+1}"] * 1.005
                        ),
                        size=exchange.amount_to_precision(
                            position.pair,
                            (
                                (params[position.pair]["size"] * usdc_balance)
                                / len(params[position.pair]["envelopes"])
                                * leverage
                            )
                            / row[f"ma_low_{i+1}"],
                        ),
                        type="limit",
                        reduce=False,
                        margin_mode=margin_mode,
                        hedge_mode=hedge_mode,
                        error=False,
                    )
                )
            for i in range(
                len(params[position.pair]["envelopes"])
                - params[position.pair]["canceled_orders_sell"],
                len(params[position.pair]["envelopes"]),
            ):
                tasks_open.append(
                    exchange.place_trigger_order(
                        pair=position.pair,
                        side="sell",
                        trigger_price=exchange.price_to_precision(
                            position.pair, row[f"ma_high_{i+1}"] * 0.995
                        ),
                        price=exchange.price_to_precision(
                            position.pair, row[f"ma_high_{i+1}"]
                        ),
                        size=exchange.amount_to_precision(
                            position.pair,
                            (
                                (params[position.pair]["size"] * usdc_balance)
                                / len(params[position.pair]["envelopes"])
                                * leverage
                            )
                            / row[f"ma_high_{i+1}"],
                        ),
                        type="limit",
                        reduce=False,
                        margin_mode=margin_mode,
                        hedge_mode=hedge_mode,
                        error=False,
                    )
                )

        print(f"Placing {len(tasks_close)} close SL / limit order...")
        await asyncio.gather(*tasks_close)  # Limit orders when in positions
        
        pairs_not_in_position = [
            pair
            for pair in pairs
            if pair not in [position.pair for position in positions]
        ]
        print("pairs_not_in_position:", pairs_not_in_position)
        for pair in pairs_not_in_position:
            row = df_list[pair].iloc[-2]
            for i in range(len(params[pair]["envelopes"])):
                if "long" in params[pair]["sides"]:
                    tasks_open.append(
                        exchange.place_trigger_order(
                            pair=pair,
                            side="buy",
                            price=exchange.price_to_precision(
                                pair, row[f"ma_low_{i+1}"]
                            ),
                            trigger_price=exchange.price_to_precision(
                                pair, row[f"ma_low_{i+1}"] * 1.005
                            ),
                            size=exchange.amount_to_precision(
                                pair,
                                (
                                    (params[pair]["size"] * usdc_balance)
                                    / len(params[pair]["envelopes"])
                                    * leverage
                                )
                                / row[f"ma_low_{i+1}"],
                            ),
                            type="limit",
                            reduce=False,
                            margin_mode=margin_mode,
                            hedge_mode=hedge_mode,
                            error=False,
                        )
                    )
                if "short" in params[pair]["sides"]:
                    tasks_open.append(
                        exchange.place_trigger_order(
                            pair=pair,
                            side="sell",
                            trigger_price=exchange.price_to_precision(
                                pair, row[f"ma_high_{i+1}"] * 0.995
                            ),
                            price=exchange.price_to_precision(
                                pair, row[f"ma_high_{i+1}"]
                            ),
                            size=exchange.amount_to_precision(
                                pair,
                                (
                                    (params[pair]["size"] * usdc_balance)
                                    / len(params[pair]["envelopes"])
                                    * leverage
                                )
                                / row[f"ma_high_{i+1}"],
                            ),
                            type="limit",
                            reduce=False,
                            margin_mode=margin_mode,
                            hedge_mode=hedge_mode,
                            error=False,
                        )
                    )

        print(f"Placing {len(tasks_open)} open limit order...")
        await asyncio.gather(*tasks_open)  # Limit orders when not in positions

        await exchange.close()
        print(
            f"--- Execution finished at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---"
        )
    except Exception as e:
        await exchange.close()
        raise e


if __name__ == "__main__":
    asyncio.run(main())
      