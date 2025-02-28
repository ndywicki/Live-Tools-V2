

from decimal import ROUND_DOWN, Decimal, getcontext
import math
import time
from typing import List, Optional
import ccxt.async_support as ccxt
import pandas as pd
from pydantic import BaseModel


class UsdtBalance(BaseModel):
    total: float
    free: float
    used: float


class Info(BaseModel):
    success: bool
    message: str


class Order(BaseModel):
    id: str
    pair: str
    type: str
    side: str
    price: float
    size: float
    reduce: bool
    filled: float
    remaining: float
    timestamp: int


class TriggerOrder(BaseModel):
    id: str
    pair: str
    type: str
    side: str
    price: float
    trigger_price: float
    size: float
    reduce: bool
    timestamp: int


class Position(BaseModel):
    pair: str
    side: str
    size: float
    usd_size: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    liquidation_price: float
    margin_mode: str
    leverage: int
    hedge_mode: bool
    open_timestamp: int = 0
    take_profit_price: float | None = None
    stop_loss_price: float | None = None

class Market(BaseModel):
    internal_pair: str
    base: str
    quote: str
    price_precision: float
    contract_precision: float
    contract_size: Optional[float] = 1.0
    min_contracts: float
    max_contracts: Optional[float] = float('inf')
    min_cost: Optional[float] = 0.0
    max_cost: Optional[float] = float('inf')
    coin_index: Optional[int] = 0
    market_price: Optional[float] = 0.0


def get_price_precision(price: float) -> float:
    log_price = math.log10(price)
    order = math.floor(log_price)
    precision = 10 ** (order - 4)
    return precision
    
def number_to_str(n: float) -> str:
    s = format(n, 'f')
    s = s.rstrip('0')
    if s.endswith('.'):
        s = s[:-1]
    
    return s


class PerpHyperliquid:
    def __init__(self, public_adress=None, private_key=None):
        hyperliquid_auth_object = {
            "walletAddress": public_adress,
            "privateKey": private_key,
        }
        self.public_adress = public_adress
        getcontext().prec = 10
        if hyperliquid_auth_object["privateKey"] == None:
            self._auth = False
            self._session = ccxt.hyperliquid()
        else:
            self._auth = True
            self._session = ccxt.hyperliquid(hyperliquid_auth_object)
        self.market: dict[str, Market] = {}
        # self._session.verbose = True

    async def close(self):
        await self._session.close()

    def get_session(self):
        return self._session

    # async def load_markets(self) -> dict[str, Market]:
    #     data = await self._session.publicPostInfo(params={
    #         "type": "metaAndAssetCtxs",
    #     })
    #     resp = {}
    #     for i in range(0,len(data[0]["universe"])):
    #         mark_price = float(data[1][i]["markPx"])
    #         object = data[0]["universe"][i]
    #         size_decimals = int(object["szDecimals"])
    #         resp[object["name"]+"/USD"] = Market(
    #             internal_pair=object["name"],
    #             base=object["name"],
    #             quote="USD",
    #             price_precision=get_price_precision(mark_price),
    #             contract_precision=1/(10**(size_decimals)),
    #             min_contracts=1/(10**(size_decimals)),
    #             min_cost=10,
    #             coin_index=i,
    #             market_price=mark_price,
    #         )
    #     self.market = resp
    #     return resp
    async def load_markets(self):
        self.market = await self._session.load_markets()

    # def ext_pair_to_pair(self, ext_pair) -> str:
    #     print("ext_pair", ext_pair)
    #     print("self.market[ext_pair]", self.market[ext_pair])
    #     return self.market[ext_pair].internal_pair
  
    # def pair_to_ext_pair(self, pair) -> str:
    #     return pair+"/USD"

    def ext_pair_to_pair(self, ext_pair) -> str:
        return f"{ext_pair}:USDC"

    def pair_to_ext_pair(self, pair) -> str:
        return pair.replace(":USDC", "")
    
    def ext_pair_to_base(self, ext_pair) -> str:
        return ext_pair.split("/")[0]

    # def get_pair_info(self, ext_pair) -> str:
    #     pair = ext_pair #self.ext_pair_to_pair(ext_pair)
    #     # print("pair", pair)
    #     # print("self.market", self.market)
    #     if pair in self.market:
    #         return self.market[pair]
    #     else:
    #         return None
    def get_pair_info(self, ext_pair) -> str:
        pair = self.ext_pair_to_pair(ext_pair)
        # print("get_pair_info pair", pair)
        # print("self.market", self.market)
        if pair in self.market:
            return self.market[pair]
        else: 
            return None
        
    def size_to_precision(self, pair: str, size: float) -> float:
        size_precision = self.market[pair].contract_precision
        decimal_precision = Decimal(str(size_precision))
        rounded_size = Decimal(str(size)).quantize(decimal_precision, rounding=ROUND_DOWN)
        return float(rounded_size)
    
    def amount_to_precision(self, pair: str, amount: float) -> float:
        pair = self.ext_pair_to_pair(pair)
        try:
            return self._session.amount_to_precision(pair, amount)
        except Exception as e:
            return 0

    def price_to_precision(self, pair: str, price: float) -> float:
        pair = self.ext_pair_to_pair(pair)
        return self._session.price_to_precision(pair, price)

    async def get_last_ohlcv(self, pair, timeframe, limit=1000) -> pd.DataFrame:
        if limit > 5000:
            limit = 5000
        base_pair = self.ext_pair_to_base(pair)
        ts_dict = {
            "1m": 1 * 60 * 1000,
            "5m": 5 * 60 * 1000,
            "15m": 15 * 60 * 1000,
            "1h": 60 * 60 * 1000,
            "2h": 2 * 60 * 60 * 1000,
            "4h": 4 * 60 * 60 * 1000,
            "1d": 24 * 60 * 60 * 1000,
        }
        end_ts = int(time.time() * 1000)
        start_ts = end_ts - ((limit-1) * ts_dict[timeframe])
        data = await self._session.publicPostInfo(params={
            "type": "candleSnapshot",
            "req": {
                "coin": base_pair,
                "interval": timeframe,
                "startTime": start_ts,
                "endTime": end_ts,
            },
        })
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['t'].astype(float), unit='ms')
        df.set_index('date', inplace=True)
        df = df[['o', 'h', 'l', 'c', 'v']].astype(float)
        df.rename(columns={
            'o': 'open',
            'h': 'high',
            'l': 'low',
            'c': 'close',
            'v': 'volume'
        }, inplace=True)

        return df

    async def get_balance(self) -> UsdtBalance:
        data = await self._session.publicPostInfo(params={
            "type": "clearinghouseState",
            "user": self.public_adress,
        })
        total = float(data["marginSummary"]["accountValue"])
        used = float(data["marginSummary"]["totalMarginUsed"])
        free = total - used
        return UsdtBalance(
            total=total,
            free=free,
            used=used,
        )

    # async def set_margin_mode_and_leverage(self, pair, margin_mode, leverage):
    #     if margin_mode not in ["cross", "isolated"]:
    #         raise Exception("Margin mode must be either 'cross' or 'isolated'")
    #     asset_index = self.market[pair].coin_index
    #     try:
    #         nonce = int(time.time() * 1000)
    #         req_body = {}
    #         action = {
    #             "type": "updateLeverage",
    #             "asset": asset_index,
    #             "isCross": margin_mode == "cross",
    #             "leverage": leverage,
    #         }
    #         signature = self._session.sign_l1_action(action, nonce)
    #         req_body["action"] = action
    #         req_body["nonce"] = nonce
    #         req_body["signature"] = signature
    #         await self._session.private_post_exchange(params=req_body)
    #     except Exception as e:
    #         raise e

    #     return Info(
    #         success=True,
    #         message=f"Margin mode and leverage set to {margin_mode} and {leverage}x",
    #     )

    async def set_margin_mode_and_leverage(self, pair, margin_mode, leverage):
        if margin_mode not in ["cross", "isolated"]:
            raise Exception("Margin mode must be either 'cross' or 'isolated'")
        pair = self.ext_pair_to_pair(pair)
        try:
            await self._session.set_leverage(
                leverage,
                pair,
                params={
                    "open_type": margin_mode,
                    "marginMode": margin_mode,
                },
            )
        except Exception as e:
            raise e

    # async def get_open_positions(self, pairs=[]) -> List[Position]:
    #     data = await self._session.publicPostInfo(params={
    #         "type": "clearinghouseState",
    #         "user": self.public_adress,
    #     })
    #     # return data
    #     positions_data = data["assetPositions"]
    #     positions = []
    #     for position_data in positions_data:
    #         position = position_data["position"]
    #         if self.pair_to_ext_pair(position["coin"]) not in pairs and len(pairs) > 0:
    #             continue
    #         type_mode = position_data["type"]
    #         hedge_mode = True if type_mode != "oneWay" else False
    #         size = float(position["szi"])
    #         side = "long" if size > 0 else "short"
    #         size = abs(size)
    #         usd_size = float(position["positionValue"])
    #         current_price = usd_size / size
    #         positions.append(
    #             Position(
    #                 pair=self.pair_to_ext_pair(position["coin"]),
    #                 side=side,
    #                 size=size,
    #                 usd_size=usd_size,
    #                 entry_price=float(position["entryPx"]),
    #                 current_price=current_price,
    #                 unrealized_pnl=float(position["unrealizedPnl"]),
    #                 liquidation_price=float(position["liquidationPx"]),
    #                 margin_mode=position["leverage"]["type"],
    #                 leverage=position["leverage"]["value"],
    #                 hedge_mode=hedge_mode,
    #             )
    #         )

    #     return positions

    # async def place_order(
    #     self,
    #     pair,
    #     side,
    #     price,
    #     size,
    #     type="limit",
    #     reduce=False,
    #     error=True,
    #     market_max_spread=0.1,
    # ) -> Order:
    #     if price is None:
    #         price = self.market[pair].market_price
    #     try:
    #         asset_index = self.market[pair].coin_index
    #         nonce = int(time.time() * 1000)
    #         is_buy = side == "buy"
    #         req_body = {}
    #         if type == "market":
    #             if side == "buy":
    #                 price = price * (1 + market_max_spread)
    #             else:
    #                 price = price * (1 - market_max_spread)

    #         print(number_to_str(self.price_to_precision(pair, price)))
    #         action = {
    #             "type": "order",
    #             "orders": [{
    #                 "a": asset_index,
    #                 "b": is_buy,
    #                 "p": number_to_str(self.price_to_precision(pair, price)),
    #                 "s": number_to_str(self.size_to_precision(pair, size)),
    #                 "r": reduce,
    #                 "t": {"limit":{"tif": "Gtc"}}
    #             }],
    #             "grouping": "na",
    #             "brokerCode": 1,
    #         }
    #         signature = self._session.sign_l1_action(action, nonce)
    #         req_body["action"] = action
    #         req_body["nonce"] = nonce
    #         req_body["signature"] = signature
    #         resp = await self._session.private_post_exchange(params=req_body)
            
    #         order_resp = resp["response"]["data"]["statuses"][0]
    #         order_key = list(order_resp.keys())[0]
    #         order_id = resp["response"]["data"]["statuses"][0][order_key]["oid"]

    #         order = await self.get_order_by_id(order_id)

    #         if order_key == "filled":
    #             order_price = resp["response"]["data"]["statuses"][0][order_key]["avgPx"]
    #             order.price = float(order_price)
            
    #         return order
    #     except Exception as e:
    #         if error:
    #             raise e
    #         else:
    #             print(e)
    #             return None
    async def place_order(
        self,
        pair,
        side,
        price,
        size,
        type="limit",
        reduce=False,
        margin_mode="crossed",
        hedge_mode=False,
        error=False,
    ) -> Order:
        try:
          
            pair = self.ext_pair_to_pair(pair)
            trade_side = "Open" if reduce is False else "Close"
            margin_mode = "cross" if margin_mode == "crossed" else "isolated"
            print("pair:", pair, "trade_side:", trade_side)
            print("margin_mode:", margin_mode, "price:", price, "size:", size, "pair:", pair, "type:", type, "reduce:", reduce, "hedge_mode:", hedge_mode)
            resp = await self._session.create_order(
                symbol=pair,
                type=type,
                side=side,
                amount=size,
                price=price,
                params={
                    "reduceOnly": reduce,
                    "tradeSide": trade_side,
                    "marginMode": margin_mode,
                    "hedged": hedge_mode,
                },
            )
            print("create_order resp:", resp)
            order_id = resp["id"]
            # pair = self.pair_to_ext_pair(resp["symbol"])
            order = await self.get_order_by_id(order_id, pair)
            print("order:", order)
            return order
        except Exception as e:
            print(f"Error {type} {side} {size} {pair} - Price {price} - Error => {str(e)}")
            if error:
                raise e
            else:
                return None

    async def place_trigger_order(
        self,
        pair,
        side,
        price,
        trigger_price,
        size,
        type="limit",
        reduce=False,
        margin_mode="crossed",
        hedge_mode=False,
        error=False,
    ) -> Info:
        try:
            pair = self.ext_pair_to_pair(pair)
            trade_side = "Open" if reduce is False else "Close"
            margin_mode = "cross" if margin_mode == "crossed" else "isolated"
            trigger_order = await self._session.create_trigger_order(
                symbol=pair,
                type=type,
                side=side,
                amount=size,
                price=price,
                triggerPrice=trigger_price,
                params={
                    "reduceOnly": reduce,
                    "tradeSide": trade_side,
                    "marginMode": margin_mode,
                    "hedged": hedge_mode,
                },
            )
            resp = Info(success=True, message="Trigger Order set up")
            return resp
        except Exception as e:
            print(f"Error {type} {side} {size} {pair} - Trigger {trigger_price} - Price {price} - Error => {str(e)}")
            if error:
                raise e
            else:
                return None


    # async def get_order_by_id(self, order_id) -> Order:
    #     order_id = int(order_id)
    #     data = await self._session.publicPostInfo(params={
    #         "user": self.public_adress,
    #         "type": "orderStatus",
    #         "oid": order_id,
    #     })
    #     order = data["order"]["order"]
    #     side_map = {
    #         "A": "sell",
    #         "B": "buy",
    #     }
    #     return Order(
    #         id=str(order_id),
    #         pair=self.pair_to_ext_pair(order["coin"]),
    #         type=order["orderType"].lower(),
    #         side=side_map[order["side"]],
    #         price=float(order["limitPx"]),
    #         size=float(order["origSz"]),
    #         reduce=order["reduceOnly"],
    #         filled=float(order["origSz"]) - float(order["sz"]),
    #         remaining=float(order["sz"]),
    #         timestamp=int(order["timestamp"]),
    #     )
    async def get_order_by_id(self, order_id, pair) -> Order:
        pair = self.ext_pair_to_pair(pair)
        resp = await self._session.fetch_order(order_id, pair)
        return Order(
            id=resp["id"],
            pair=self.pair_to_ext_pair(resp["symbol"]),
            type=resp["type"],
            side=resp["side"],
            price=resp["price"],
            size=resp["amount"],
            reduce=resp["reduceOnly"],
            filled=resp["filled"],
            remaining=resp["remaining"],
            timestamp=resp["timestamp"],
        )

    # async def cancel_orders(self, pair, ids=[]):
    #     try:
    #         asset_index = self.market[pair].coin_index
    #         nonce = int(time.time() * 1000)
    #         req_body = {}
    #         orders_action = []
    #         for order_id in ids:
    #             orders_action.append({
    #                 "a": asset_index,
    #                 "o": int(order_id),
    #             })
    #         action = {
    #             "type": "cancel",
    #             "cancels": orders_action,
    #         }
    #         signature = self._session.sign_l1_action(action, nonce)
    #         req_body["action"] = action
    #         req_body["nonce"] = nonce
    #         req_body["signature"] = signature
    #         resp = await self._session.private_post_exchange(params=req_body)
    #         return Info(success=True, message=f"Orders cancelled")
    #     except Exception as e:
    #         return Info(success=False, message="Error or no orders to cancel")
    async def cancel_orders(self, pair, ids=[]):
        try:
            pair = self.ext_pair_to_pair(pair)
            resp = await self._session.cancel_orders(
                ids=ids,
                symbol=pair,
            )
            return Info(success=True, message=f"{len(resp)} Orders cancelled")
        except Exception as e:
            return Info(success=False, message="Error or no orders to cancel")
        
    async def get_open_trigger_orders(self, pair) -> List[TriggerOrder]:
        pair = self.ext_pair_to_pair(pair)
        print("pair", pair)
        # resp = await self._session.fetch_open_orders(pair, params={"stop": True})
        resp = await self._session.fetch_open_orders('HYPE/USDC:USDC')
        print(resp)
        return_orders = []
        # for order in resp:
        #     reduce = True if order["info"]["tradeSide"] == "close" else False
        #     price = order["price"] if order["price"] else 0.0
        #     return_orders.append(
        #         TriggerOrder(
        #             id=order["id"],
        #             pair=self.pair_to_ext_pair(order["symbol"]),
        #             type=order["type"],
        #             side=order["side"],
        #             price=price,
        #             trigger_price=order["triggerPrice"],
        #             size=order["amount"],
        #             reduce=reduce,
        #             timestamp=order["timestamp"],
        #         )
        #     )
        return return_orders

    async def get_open_orders(self, pair) -> List[Order]:
        pair = self.ext_pair_to_pair(pair)
        resp = await self._session.fetch_open_orders(pair)
        # print("get_open_orders resp", resp)
        return_orders = []
        for order in resp:
            return_orders.append(
                Order(
                    id=order["id"],
                    pair=self.pair_to_ext_pair(order["symbol"]),
                    type=order["type"],
                    side=order["side"],
                    price=order["price"],
                    size=order["amount"],
                    reduce=order["reduceOnly"],
                    filled=order["filled"],
                    remaining=order["remaining"],
                    timestamp=order["timestamp"],
                )
            )
        return return_orders
      
    async def get_open_positions(self, pairs) -> List[Position]:
        pairs = [self.ext_pair_to_pair(pair) for pair in pairs]
        resp = await self._session.fetch_positions(symbols=pairs)
        print("fetch_positions resp:", resp)
        return_positions = []
        for position in resp:
            liquidation_price = 0
            take_profit_price = None
            stop_loss_price = None
            hedge_mode = False
            if "liquidationPrice" in position:
                liquidation_price = position["liquidationPrice"]
            if "takeProfitPrice" in position:
                take_profit_price = position["takeProfitPrice"]
            if "stopLossPrice" in position:
                stop_loss_price = position["stopLossPrice"]
            if "hedged" in position:
                hedge_mode = True

            # Use entryPrice if markPrice is not available
            current_price = position["markPrice"] if position["markPrice"] is not None else position["entryPrice"]
        
            return_positions.append(
                Position(
                    pair=self.pair_to_ext_pair(position["symbol"]),
                    side=position["side"],
                    size=Decimal(position["contracts"])
                    * Decimal(position["contractSize"]),
                    usd_size=round(
                        position["notional"],
                        2,
                    ),
                    entry_price=position["entryPrice"],
                    current_price=Decimal(current_price),
                    unrealized_pnl=position["unrealizedPnl"],
                    liquidation_price=liquidation_price,
                    leverage=position["leverage"],
                    margin_mode=position["marginMode"],
                    hedge_mode=hedge_mode,
                    # open_timestamp=position["info"]["open_timestamp"],
                    take_profit_price=take_profit_price,
                    stop_loss_price=stop_loss_price,
                )
            )
        return return_positions
      
    async def cancel_trigger_orders(self, pair, ids=[]):
      try:
          pair = self.ext_pair_to_pair(pair)
          resp = await self._session.cancel_orders(
              ids=ids, symbol=pair, params={"stop": True}
          )
          return Info(success=True, message=f"{len(resp)} Trigger Orders cancelled")
      except Exception as e:
          return Info(success=False, message="Error or no orders to cancel")