import torch
import torch.nn as nn
import random
import math
import numpy as np

# GWrapper クラス（既存のまま使用）
class GWrapper(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, h_fn, g_fn):
        with torch.no_grad():
            h_out = h_fn(x)
        h_out_detached = h_out.detach().requires_grad_()
        with torch.enable_grad():
            y = g_fn(h_out_detached)
        ctx.save_for_backward(h_out_detached, y)
        ctx.g_fn = g_fn
        return y

    @staticmethod
    def backward(ctx, grad_output):
        h_out_detached, y = ctx.saved_tensors
        g_fn = ctx.g_fn
        grad_h = torch.autograd.grad(
            y, h_out_detached, grad_output, retain_graph=True, allow_unused=True
        )[0]
        return grad_h, None, None

# MarketGenerator = h(x)
class MarketGenerator(nn.Module):
    def __init__(self, input_dim=4, hidden_dim1=128,hidden_dim2=64,output_dim=3):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            #nn.Linear(hidden_dim1,hidden_dim2),
            nn.Linear(hidden_dim1,output_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.model(x)  # h(x)


# RLAgent = g(h)
class RLAgent(nn.Module):
    def __init__(self, initial_cash, input_dim=2, hidden_dim1=128,hidden_dim2=64, output_dim=4):
        self.positions = torch.empty(0)
        self.closed_positions = []
        self.effective_margin = torch.tensor(initial_cash, dtype=torch.float32, requires_grad=False)
        self.required_margin = 0
        self.margin_deposit = torch.tensor(initial_cash, dtype=torch.float32, requires_grad=False)
        self.realized_profit = torch.tensor(0.0, dtype=torch.float32, requires_grad=False)
        self.long_position = torch.tensor(0.0, dtype=torch.float32, requires_grad=False)
        self.short_position = torch.tensor(0.0, dtype=torch.float32, requires_grad=False)
        self.unfulfilled_long_open = torch.tensor(0.0, dtype=torch.float32, requires_grad=False)
        self.unfulfilled_short_open = torch.tensor(0.0, dtype=torch.float32, requires_grad=False)
        self.unfulfilled_long_close = torch.tensor(0.0, dtype=torch.float32, requires_grad=False)
        self.unfulfilled_short_close = torch.tensor(0.0, dtype=torch.float32, requires_grad=False)
        self.margin_maintenance_rate = np.inf
        self.margin_maintenance_flag = False
        self.effective_margin_max = -np.inf
        self.effective_margin_min = np.inf

        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            #nn.Linear(hidden_dim1,hidden_dim2),
            nn.Linear(hidden_dim1,output_dim),
            nn.ReLU()
        )

    def forward(self, h):
        return self.model(h)  # g(h)
    
    def _remove_position(self, index):
        print(f"positions_index before removing a position:{tf.shape(self.positions.stack())[0]}")
        valid_indices = [i for i in range(len(self.positions)) if i != index]


        # 🔥 削除済みの `pos_id` をリストとして保存
        self.valid_pos_ids = valid_indices.copy()

        filtered_positions = self.positions.gather(valid_indices)

        # 新しい `TensorArray` に入れ直す
        new_positions = []
        for i in range(np.shape(filtered_positions)[0]):
            new_positions = new_positions.write(i, filtered_positions[i])

        self.positions = new_positions
        print(f"after removing")
        print(self.positions.stack())
        print(f"positions_index after removing a position:{np.shape(self.positions.stack())[0]}")





    def process_new_order(self, long_order_size, short_order_size, current_price, margin_rate):
        #trade_type = tf.Variable(trade_type, name="trade_type",dtype=tf.float32,trainable=True)
        if long_order_size > 0 and short_order_size > 0:
            order_margin = ((long_order_size + short_order_size + self.unfulfilled_short_open + self.unfulfilled_long_open) * current_price * margin_rate)
            #self.margin_maintenance_flag, self.margin_maintenance_rate = update_margin_maintenance_rate(self.effective_margin, self.required_margin)
            if self.margin_maintenance_flag:
                #print(f"Margin cut triggered during {'Buy' if trade_type.numpy() == 1.0 else 'Sell'} order processing.")
                print(f"Margin cut triggered during order processing.")
                return
            order_capacity = (self.effective_margin - (self.required_margin+ order_margin)).numpy()
            if order_capacity < 0:
                #print(f"Cannot process {'Buy' if trade_type.numpy() == 1.0 else 'Sell'} order due to insufficient order capacity.")
                print(f"Cannot process order due to insufficient order capacity.")
                #if trade_type == 1:
                #self.unfulfilled_long_open += order_size
                new_unfulfilled_buy_orders = self.unfulfilled_long_open + long_order_size
                self.unfulfilled_long_open = new_unfulfilled_buy_orders
                #elif trade_type == -1:
                #self.unfulfilled_short_open += order_size
                new_unfulfilled_sell_orders = self.unfulfilled_short_open + short_order_size
                self.unfulfilled_short_open = new_unfulfilled_sell_orders
                # when cannot buy or sell, you need effective_margin which itself is unchanging and depends on the output of generator and discriminators
                update_effective_margin = self.effective_margin + current_price * (long_order_size+short_order_size) * 0
                self.effective_margin = update_effective_margin
                return
            if self.margin_maintenance_rate > 100 and order_capacity > 0:
                #if trade_type == 1:
                #order_margin -= (order_size + self.unfulfilled_long_open) * current_price * margin_rate
                #self.unfulfilled_long_open.assign(0.0)
                #add_required_margin = (order_size + self.unfulfilled_long_open) * current_price * margin_rate
                #elif trade_type == -1:
                #order_margin -= (order_size + self.unfulfilled_short_open) * current_price * margin_rate
                #self.unfulfilled_short_open.assign(0.0)
                long_add_required_margin = (long_order_size + self.unfulfilled_long_open) * current_price * margin_rate
                short_add_required_margin = (short_order_size + self.unfulfilled_short_open) * current_price * margin_rate
                self.required_margin += long_add_required_margin + short_add_required_margin
                #order_size = tf.add(order_size, self.unfulfilled_long_open if trade_type == 1 else self.unfulfilled_short_open)
                if long_order_size > 0:
                    long_order_size += self.unfulfilled_long_open
                    new_unfulfilled_long_open = self.unfulfilled_long_open - self.unfulfilled_long_open
                    self.unfulfilled_long_open = new_unfulfilled_long_open
                    pos = [long_order_size, 1, current_price, 0.0, long_add_required_margin, 0.0]
                    print(f"Opened Buy position at {current_price}, required_margin:{self.required_margin}")
                    self.positions.append(pos)
                    #new_order_size = order_size + self.unfulfilled_long_open
                    #order_size = new_order_size
                if short_order_size > 0:
                    short_order_size += self.unfulfilled_short_open
                    new_unfulfilled_short_open = self.unfulfilled_short_open - self.unfulfilled_short_open
                    self.unfulfilled_short_open = new_unfulfilled_short_open
                    pos = [short_order_size,-1, current_price, 0.0, short_add_required_margin, 0.0]
                    print(f"Opened Sell position at {current_price}, required_margin:{self.required_margin}")
                    self.positions.append(pos)
                    #new_order_size = order_size + self.unfulfilled_short_open
                    #order_size = new_order_size
                #pos = tf.stack([order_size, trade_type, current_price, tf.Variable(0.0,name="unrealized_profit",dtype=tf.float32,trainable=True), add_required_margin, tf.Variable(0.0,name="profit",dtype=tf.float32,trainable=True)])

                #self.positions = self.positions.write(tf.cast(self.positions_index, tf.int32), pos)

                #self.positions_index.assign_add(1)
                print(f"positions_index in process_new_order:{np.shape(self.positions)[0]}")
                #print(f"Opened position at {current_price}, required margin: {self.required_margin}")
                self.margin_maintenance_flag, self.margin_maintenance_rate = update_margin_maintenance_rate(self.effective_margin,self.required_margin)
                if self.margin_maintenance_flag:
                    print(f"margin maintenance rate is {self.margin_maintenance_rate},so loss cut is executed in process_new_order, effective_margin: {self.effective_margin}")
                    return


    def process_position_closure(self,long_close_position,short_close_position,current_price):
        if self.margin_maintenance_flag:
            #print(f"Margin cut triggered during {'Buy' if trade_type.numpy() == 1.0 else 'Sell'} order processing.")
            print(f"Margin cut triggered during position closure.")
            return
        new_unfulfilled_long_close = self.unfulfilled_long_close + long_close_position
        self.unfulfilled_long_close = new_unfulfilled_long_close
        new_unfulfilled_short_close = self.unfulfilled_short_close + short_close_position
        self.unfulfilled_short_close = new_unfulfilled_short_close

        # --- ポジション決済処理 ---
        pos_id_max = int(np.shape(self.positions)[0] - 1)  # 現在の最大 ID
        to_be_removed = []
        for pos_id in range(pos_id_max + 1):  # 最大 ID までの範囲を網羅
            try:
                pos = self.positions[pos_id]
                print(f"In the position closure process, pos_id in update of position value{pos_id}: {pos.numpy()}")
                if tf.reduce_all(pos == 0.0):
                    print(f"this pos_id is all 0.0 so skipped")
                    continue
            except:
                continue
            size, pos_type, open_price, unrealized_profit, margin, realized_profit = pos

            if pos_type.numpy() == 1.0 and self.unfulfilled_long_close > 0:
                fulfilled_size = tf.minimum(self.unfulfilled_long_close,size)
                profit = fulfilled_size * (current_price - open_price)
            elif pos_type.numpy() == 1.0 and self.unfulfilled_long_close == 0:
                pos = [size, pos_type, open_price, unrealized_profit, margin, realized_profit]
                self.positions.append(pos)
                print("unfulfilled_long_close == 0 so continue")
                continue
            elif pos_type.numpy() == -1.0 and self.unfulfilled_short_close > 0:
                fulfilled_size = tf.minimum(self.unfulfilled_short_close,size)
                profit = fulfilled_size * (open_price - current_price)
            elif pos_type.numpy() == -1.0 and self.unfulfilled_short_close == 0:
                pos = [size, pos_type, open_price, unrealized_profit, margin, realized_profit]
                self.positions.append(pos)
                print("unfulfilled_short_close == 0 so continue")
                continue
            

            # 決済ロジック
            #self.effective_margin.assign_add(profit - unrealized_profit)
            update_effective_margin = self.effective_margin + profit - unrealized_profit
            self.effective_margin = update_effective_margin
            self.effective_margin_max, self.effective_margin_min = check_min_max_effective_margin(self.effective_margin, self.effective_margin_max, self.effective_margin_min)
            print(f"profit:{profit}")
            print(f"unrealized_profit:{unrealized_profit}")
            #exit()
            self.margin_deposit.assign_add(profit)
            self.realized_profit.assign_add(profit)
            add_required_margin = - margin * (fulfilled_size/size)
            self.required_margin += add_required_margin.numpy()
            print(f"fulfill_size:{fulfilled_size}, current_price:{current_price},open_price:{open_price}, effective_margin:{self.effective_margin}, required_margin:{self.required_margin}")

            # 部分決済または完全決済の処理
            size -= fulfilled_size
            if pos_type.numpy() == 1.0:
                self.unfulfilled_long_close -= fulfilled_size
            if pos_type.numpy() == -1.0:
                self.unfulfilled_short_close -= fulfilled_size
            #new_size = size - fulfilled_size
            #size = new_size
            if size > 0:  # 部分決済の場合
                print(f"partial payment was executed")
                pos = [size, pos_type, open_price, 0, add_required_margin, 0]#once substract unrealized_profit from effective_margin, you need not do it again in the process of update pos, so you have to set unrealized_profit to 0.
                self.positions.append(pos)
                print(self.positions)
                pos = [fulfilled_size, pos_type, open_price, 0, 0, profit]
                self.closed_positions.append(pos)

            else:  # 完全決済の場合
                print(f"all payment was executed")
                pos = [fulfilled_size, pos_type, open_price, 0, 0, profit] #we hope margin+add_required_margin==0
                #print(f"all payment: margin+add_required_margin:{margin+add_required_margin}")
                self.closed_positions.append(pos)

                to_be_removed.append(pos_id)
                if self.positions.size().numpy() == len(to_be_removed):  #this sentence needs because required_margin must be just 0 when all positions are payed, but actually not be just 0 because of rounding error.
                    self.required_margin = 0
                self.positions.append(pos) #once rewrite the pos which you should remove because you have to write or stack the pos which once you read

            print(f"Closed {'Buy' if pos_type.numpy()==1.0 else ('Sell' if pos_type.numpy() == -1.0 else 'Unknown')} position at {current_price} with profit {profit} ,grid {open_price}, Effective Margin: {self.effective_margin}, Required Margin: {self.required_margin}")

            #if pos_type.numpy() == 1.0:
            #    #self.unfulfilled_long_open.assign(long_close_position - fulfilled_size)
            #    self.unfulfilled_long_open = long_close_position - fulfilled_size
            #elif pos_type.numpy() == -1.0:
            #    #self.unfulfilled_short_open.assign(short_close_position - fulfilled_size)
            #    self.unfulfilled_short_open = short_close_position - fulfilled_size

            self.margin_maintenance_flag, self.margin_maintenance_rate = update_margin_maintenance_rate(self.effective_margin,self.required_margin)
            if self.margin_maintenance_flag:
                print(f"margin maintenance rate is {self.margin_maintenance_rate},so loss cut is executed in position closure process, effective_margin: {self.effective_margin}")
                continue
        for pos_id in sorted(to_be_removed, reverse=True):  # 降順で削除（pos_id がズレないように）
            self._remove_position(pos_id)


    def process_position_update(self, current_price, required_margin_rate):
        if self.margin_maintenance_flag:
            #print(f"Margin cut triggered during {'Buy' if trade_type.numpy() == 1.0 else 'Sell'} order processing.")
            print(f"Margin cut triggered during position update.")
            return
        """
        資産とポジションの更新を実行
        """
        # --- 含み益の更新 ---
        pos_id_max = int(np.shape(self.positions)[0] - 1)  # 現在の最大 ID
        for pos_id in range(pos_id_max + 1):  # 最大 ID までの範囲を網羅
            try:
                pos = self.positions[pos_id]
                print(f"In the update position value, pos_id in update of position value{pos_id}: {pos.numpy()}")
                if np.all(pos == 0.0):
                    print(f"this pos_id is all 0.0 so skipped")
                    continue
            except:
                continue
            size, pos_type, open_price, before_unrealized_profit, margin, _ = pos
            print(f"unstacked pos_id:{pos_id}")

            #print(f"# update of position values")
            #print(f"current_price:{current_price}")
            #print(f"open_price:{open_price}")
            #print(f"size:{size}")
            unrealized_profit = size * (current_price - open_price) if pos_type.numpy() == 1.0 else size * (open_price - current_price)
            print(f"unrealized_profit in update of unrealized profit: {unrealized_profit}, pos_type:{pos_type}")
            #self.effective_margin.assign(self.effective_margin + unrealized_profit - before_unrealized_profit)
            print(f"right before updating effective_margin:{self.effective_margin}")
            update_effective_margin = self.effective_margin + unrealized_profit - before_unrealized_profit
            self.effective_margin = update_effective_margin
            self.effective_margin_max, self.effective_margin_min = check_min_max_effective_margin(self.effective_margin, self.effective_margin_max, self.effective_margin_min)
            print(f"right after updating effective_margin:{self.effective_margin}")
            #exit()
            add_required_margin = -margin + current_price * size * required_margin_rate
            self.required_margin += add_required_margin.numpy()

            pos = [size, pos_type, open_price, unrealized_profit,add_required_margin,0]
            #print(f"pos_id:{type(pos_id)}")
            #print(f"size:{type(size)}")
            #print(f"pos_type:{type(pos_type)}")
            #print(f"open_price:{type(open_price)}")
            #print(f"unrealized_profit:{type(unrealized_profit)}")
            #print(f"margin:{type(margin)}")
            #print(f"add_required_margin:{type(add_required_margin)}")
            #exit()
            self.positions.append(pos)
            print(f"unrealized_profit:{unrealized_profit}, before_unrealized_profit:{before_unrealized_profit}")
            print(f"updated effective margin against price {current_price} , effective Margin: {self.effective_margin}, required_margin:{self.required_margin}, pos_id:{pos_id}")
            print(self.positions)

            self.margin_maintenance_flag, self.margin_maintenance_rate = update_margin_maintenance_rate(self.effective_margin,self.required_margin)
            if self.margin_maintenance_flag:
                print(f"margin maintenance rate is {self.margin_maintenance_rate},so loss cut is executed in position update process, effective_margin: {self.effective_margin}")

        # --- 強制ロスカットのチェック ---
        self.margin_maintenance_flag, _ = update_margin_maintenance_rate(self.effective_margin, self.required_margin)
        if self.margin_maintenance_flag:
            print("Forced margin cut triggered.")
            pos_id_max = int(np.shape(self.positions)[0] - 1)  # 現在の最大 ID
            to_be_removed = []
            print(f"pos_id_max right before forced margin cut triggered: {pos_id_max}")
            for pos_id in range(pos_id_max + 1):  # 最大 ID までの範囲を網羅
                try:
                    pos = self.positions.read(pos_id)
                    print(f"In the loss cut, pos_id in update of position value{pos_id}: {pos.numpy()}")
                    if np.all(pos == 0.0):
                        print(f"this pos_id is all 0.0 so skipped")
                        continue
                except:
                    continue
                size, pos_type, open_price, before_unrealized_profit, margin, _ = pos

                if pos_type.numpy() == 1.0:
                    profit = (current_price - open_price) * size  # 現在の損失計算
                elif pos_type.numpy() == -1.0:
                    profit = -(current_price - open_price) * size
                #self.effective_margin.assign(self.effective_margin + profit - before_unrealized_profit) # 損失分を証拠金に反映
                update_effective_margin = self.effective_margin + profit - before_unrealized_profit
                self.effective_margin = update_effective_margin
                self.effective_margin_max, self.effective_margin_min = check_min_max_effective_margin(self.effective_margin, self.effective_margin_max, self.effective_margin_min)
                self.margin_deposit.assign(self.margin_deposit + profit)
                #self.margin_deposit += profit
                self.realized_profit.assign(self.realized_profit + profit)
                #self.realized_profit += profit
                self.required_margin -= margin

                to_be_removed.append(pos_id)
                if self.positions.size().numpy() == len(to_be_removed):  #this sentence needs because required_margin must be just 0 when all positions are payed, but actually not be just 0 because of rounding error.
                    self.required_margin = 0
                self.positions.append(pos) #once rewrite the pos which you should remove because you have to write or stack the pos which once you read

                pos = [size, pos_type, open_price, 0, 0, profit]
                self.closed_positions.append(pos)
                print(f"Forced Closed at currnt_price:{current_price} with open_price:{open_price}, Effective Margin: {self.effective_margin}")
            #self.required_margin = 0.0

            for pos_id in sorted(to_be_removed, reverse=True):  # 降順で削除（pos_id がズレないように）
                self._remove_position(pos_id)

            # 全ポジションをクリア
            self.positions = []

            self.required_margin = 0
            _, self.margin_maintenance_rate = update_margin_maintenance_rate(self.effective_margin,self.required_margin) 

        #"""


    """
    def train(self, tape,generation,actions):
        long_order_size, short_order_size, long_close_position, short_close_position = tf.unstack(actions)
        loss = -(tf.math.log(current_price + 1e-6) \
     + tf.math.log(long_order_size + 1e-6) \
     + tf.math.log(short_order_size + 1e-6) \
     + tf.math.log(long_close_position + 1e-6) \
     + tf.math.log(short_close_position + 1e-6))
        if generation == generations // 2:
            loss = -self.effective_margin  # 総資産の最大化
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
    """



def match_orders(agents, actions, current_price, required_margin_rate):
    """
    各エージェントの新規注文 (ロング・ショート) および決済注文 (ロング・ショート) を需給に基づいて処理する。
    - 相殺も考慮し、未決済注文を適切に処理する。
    """

    # 🔹 1️⃣ 各エージェントの注文を取得
    long_open_orders = []  # 新規ロング注文
    short_open_orders = []  # 新規ショート注文
    long_close_orders = []  # ロング決済注文
    short_close_orders = []  # ショート決済注文

    for agent, action in zip(agents, actions):
        action_flat = np.reshape(action, [-1])  # 形状 (4,) に変換
        long_open_position, short_open_position, long_close_position, short_close_position = action_flat

        long_open_orders.append(long_open_position)
        short_open_orders.append(short_open_position)
        long_close_orders.append(long_close_position)
        short_close_orders.append(short_close_position)

    # 🔹 2️⃣ 需給に基づいたマッチング処理
    total_long_open = sum(long_open_orders)
    total_short_open = sum(short_open_orders)
    total_long_close = sum(long_close_orders)
    total_short_close = sum(short_close_orders)

    executed_open_volume = min(total_long_open, total_short_open)  # 売りと買いのどちらか小さい方を全約定
    executed_close_volume = min(total_long_close, total_short_close)  # ロング・ショート決済のどちらか小さい方を全約定

    # ✅ エージェントごとの注文を比例配分して処理
    i = 0
    for agent, long_open_size, short_open_size, long_close_size, short_close_size in zip(
            agents, long_open_orders, short_open_orders, long_close_orders, short_close_orders):

        long_open_ratio = long_open_size / total_long_open if total_long_open > 0 else 0
        short_open_ratio = short_open_size / total_short_open if total_short_open > 0 else 0
        long_close_ratio = long_close_size / total_long_close if total_long_close > 0 else 0
        short_close_ratio = short_close_size / total_short_close if total_short_close > 0 else 0

        executed_long_open = executed_open_volume * long_open_ratio
        executed_short_open = executed_open_volume * short_open_ratio
        executed_long_close = executed_close_volume * long_close_ratio
        executed_short_close = executed_close_volume * short_close_ratio

        print("\n")
        print(f"process_new_order of {i}th agent by ordering new positions")
        agent.process_new_order(executed_long_open, executed_short_open, current_price, required_margin_rate)
        print("\n")
        print(f"process_position_closure of {i}th agent by closing positions")
        agent.process_position_closure(executed_long_close, executed_short_close, current_price)
        i += 1

    # 🔹 3️⃣ 新規注文と決済注文の未約定部分を相殺
    remaining_long_open = total_long_open - executed_open_volume
    remaining_short_open = total_short_open - executed_open_volume
    remaining_long_close = total_long_close - executed_close_volume
    remaining_short_close = total_short_close - executed_close_volume

    total_buy_side = remaining_long_open + remaining_short_close
    total_sell_side = remaining_short_open + remaining_long_close

    executed_cross_volume = min(total_buy_side, total_sell_side)

    if executed_cross_volume > 0:
        executed_longs = executed_cross_volume * (remaining_long_open / total_buy_side) if total_buy_side > 0 else 0
        executed_shorts = executed_cross_volume * (remaining_short_close / total_buy_side) if total_buy_side > 0 else 0
        executed_sells = executed_cross_volume * (remaining_short_open / total_sell_side) if total_sell_side > 0 else 0
        executed_buys = executed_cross_volume * (remaining_long_close / total_sell_side) if total_sell_side > 0 else 0


        final_remaining_long_open = remaining_long_open - executed_longs
        final_remaining_short_open = remaining_short_open - executed_shorts
        final_remaining_long_close = remaining_long_close - executed_buys
        final_remaining_short_close = remaining_short_close - executed_sells


        # ✅ 相殺をエージェントごとに比例配分
        i = 0
        for agent, long_open_size, short_open_size, long_close_size, short_close_size in zip(
                agents, long_open_orders, short_open_orders, long_close_orders, short_close_orders):

            long_open_ratio = long_open_size / total_long_open if total_long_open > 0 else 0
            short_open_ratio = short_open_size / total_short_open if total_short_open > 0 else 0
            long_close_ratio = long_close_size / total_long_close if total_long_close > 0 else 0
            short_close_ratio = short_close_size / total_short_close if total_short_close > 0 else 0

            executed_long_open = executed_longs * long_open_ratio
            executed_short_open = executed_shorts * short_open_ratio
            executed_long_close = executed_buys * long_close_ratio
            executed_short_close = executed_sells * short_close_ratio

            print("\n")
            print(f"process_new_order of {i}th agent by offsetting remaining orders")
            agent.process_new_order(executed_long_open, executed_short_open, current_price, required_margin_rate)
            print("\n")
            print(f"process_position_closure of {i}th agent by offsetting remaining orders")
            agent.process_position_closure(executed_long_close, executed_short_close, current_price)
            i += 1


            # ✅ `validate_unfulfilled_orders()` を呼び出す前に計算
            #previous_unfulfilled_long_open = sum(agent.unfulfilled_long_open for agent in agents)
            #previous_unfulfilled_short_open = sum(agent.unfulfilled_short_open for agent in agents)
            #previous_unfulfilled_long_close = sum(agent.unfulfilled_long_close for agent in agents)
            #previous_unfulfilled_short_close = sum(agent.unfulfilled_short_close for agent in agents)

            #update final remaining unfulfilled orders
            new_unfulfilled_long_open = agent.unfulfilled_long_open + final_remaining_long_open * long_open_ratio
            agent.unfulfilled_long_open = new_unfulfilled_long_open

            new_unfulfilled_short_open = agent.unfulfilled_short_open + final_remaining_short_open * short_open_ratio
            agent.unfulfilled_short_open = new_unfulfilled_short_open

            new_unfulfilled_long_close = agent.unfulfilled_long_close + final_remaining_long_close * long_close_ratio
            agent.unfulfilled_long_close = new_unfulfilled_long_close

            new_unfulfilled_short_close = agent.unfulfilled_short_close + final_remaining_short_close * short_close_ratio
            agent.unfulfilled_short_close = new_unfulfilled_short_close

    print(f"executed_open_volume:{executed_open_volume}")
    print(f"executed_close_volume:{executed_close_volume}")
    #validate_unfulfilled_orders(agents, previous_unfulfilled_long_open, previous_unfulfilled_short_open, 
    #                        previous_unfulfilled_long_close, previous_unfulfilled_short_close,
    #                        final_remaining_long_open, final_remaining_short_open, 
    #                        final_remaining_long_close, final_remaining_short_close)

# 実行例
input_dim = 4
hidden_dim = 8
output_dim = 3
num_agents = 5
states = [100.0, 1.0, 0.01]
supply_and_demand = 0
initial_cash = 1000000

log_supply_and_demand = np.sign(supply_and_demand) * math.log(abs(supply_and_demand) + 1e-6)

# generator の入力データ
log_inputs = np.concatenate([
    np.log(np.reshape(states,[-1]) + 1e-6),
    [log_supply_and_demand]
], axis=0)

x = torch.tensor(log_inputs, dtype=torch.float32, requires_grad=True)  # ✅必要

generator = MarketGenerator(input_dim, hidden_dim)

agents = [RLAgent(initial_cash) for _ in range(num_agents)]

generated_states = generator.forward(x)

unlog_generated_states = torch.exp(generated_states) - 1e-6
current_price, current_liquidity, current_slippage = unlog_generated_states

actions = []
for agent in agents:
    log_inputs = torch.log(torch.stack([
    agent.effective_margin + torch.tensor(1e-6),
    current_price + torch.tensor(1e-6)
]))

    action = agent.forward(log_inputs)
    actions.append(action)

#match_orders(agents, actions, current_price, required_margin_rate=0.01)

effective_margins = [agent.effective_margin for agent in agents]
# ランダムなインデックスを取得
random_index = random.randint(0, len(agents) - 1)
# ランダムに選択した effective_margin を取得
selected_margin = effective_margins[random_index]
print(f"selected agent is {random_index}th agent")
#gen_loss = tf.reduce_mean(tf.stack([agent.effective_margin for agent in agents]))
gen_loss = selected_margin
gen_loss.backward()


def g_h(h):
    actions = [agent(h) for agent in agents]
    match_orders(agents, actions, current_price, 0.01)
    return torch.stack([agent.effective_margin for agent in agents])

agent_effective_margins = GWrapper.apply(x,generator,g_h)
for i in range(len(agents)):
    disc_loss = -agent_effective_margins[i]
    disc_loss.backward()

print(f"x.grad: {x.grad}")  # ← None（h'(x) を無視している）
