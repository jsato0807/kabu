import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import os
import random

def set_seed(seed=43):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)  # Python の乱数シード
    np.random.seed(seed)  # NumPy の乱数シード
    tf.random.set_seed(seed)  # TensorFlow の乱数シード

set_seed()

class MarketGenerator(tf.keras.Model):
    def __init__(self, input_dim=4, output_dim=3):
        super().__init__()
        self.model = tf.keras.Sequential([
            layers.Input(shape=(input_dim,)),
            layers.Dense(128, activation='sigmoid'),
            layers.Dense(64, activation='sigmoid'),
            layers.Dense(output_dim, activation='softplus')
        ])
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002)

        # `add_weight()` を使用しつつ、tf.Variable に変換して `GradientTape` で追跡できるようにする
        self.log_scale_factor = self.add_weight(
                name="log_scale_factor",
                shape=(),
                initializer=tf.keras.initializers.Constant(np.log(100)),
                trainable=True
            )

    def generate(self, inputs):
        return self.model(tf.expand_dims(inputs, axis=0)) * self.log_scale_factor

    """
    def train(self, tape, discriminator_performance,generation,actions):
        long_order_size, short_order_size, long_close_position, short_close_position = tf.unstack(actions)
        loss = tf.math.log(current_price + 1e-6) \
     + tf.math.log(long_order_size + 1e-6) \
     + tf.math.log(short_order_size + 1e-6) \
     + tf.math.log(long_close_position + 1e-6) \
     + tf.math.log(short_close_position + 1e-6)
        if generation == generations // 2:
            loss = tf.reduce_mean(discriminator_performance)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
    """


def check_min_max_effective_margin(effective_margin, effective_margin_max, effective_margin_min):
    if effective_margin_max < effective_margin:
        effective_margin_max = effective_margin
    if effective_margin_min > effective_margin:
        effective_margin_min = effective_margin
    return effective_margin_max, effective_margin_min


def update_margin_maintenance_rate(effective_margin, required_margin, margin_cut_threshold=100.0):
    if required_margin != 0:  
        margin_maintenance_rate = (effective_margin / required_margin) * 100
    else:
        margin_maintenance_rate = np.inf

    if margin_maintenance_rate <= margin_cut_threshold:
        print(f"Margin maintenance rate is {margin_maintenance_rate}%, below threshold. Forced losscut triggered.")
        return True, margin_maintenance_rate  # フラグと値を返す
    return False, margin_maintenance_rate  # フラグと値を返す

class RLAgent(tf.Module):
    def __init__(self, initial_cash=100000):
        super().__init__()
        self.positions = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True,clear_after_read=False)
        self.closed_positions = []
        self.positions_index = tf.Variable(0, name="positions_index",dtype=tf.float32,trainable=True)
        self.effective_margin = tf.Variable(initial_cash, name="effective_margin",dtype=tf.float32,trainable=True)
        self.required_margin = 0
        self.margin_deposit = tf.Variable(initial_cash, name="margin_deposit",dtype=tf.float32,trainable=True)
        self.realized_profit = tf.Variable(0.0, name="realized_profit",dtype=tf.float32,trainable=True)
        self.long_position = tf.Variable(0.0, name="long_position",dtype=tf.float32,trainable=True)
        self.short_position = tf.Variable(0.0, name="short_position",dtype=tf.float32,trainable=True)
        self.unfulfilled_long_open = tf.Variable(0.0, name="unfulfilled_long_open",dtype=tf.float32,trainable=True)
        self.unfulfilled_short_open = tf.Variable(0.0, name="unfulfilled_short_open",dtype=tf.float32,trainable=True)
        self.unfulfilled_long_close = tf.Variable(0.0, name="unfulfilled_long_open",dtype=tf.float32,trainable=True)
        self.unfulfilled_short_close = tf.Variable(0.0, name="unfulfilled_short_close",dtype=tf.float32,trainable=True)
        self.margin_maintenance_rate = np.inf
        self.model = tf.keras.Sequential([
            layers.Dense(128, activation='sigmoid', input_dim=2),
            layers.Dense(64, activation='sigmoid'),
            layers.Dense(4, activation='relu')
        ])
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

    def act(self, state):
        # state の形状を統一して結合
        state = [tf.reshape(tf.Variable(s,dtype=tf.float32,trainable=True), [1]) for s in state]
        state = tf.concat(state, axis=0)
        state = tf.reshape(state, [1, -1])  # 形状を統一
        action = self.model(state)
        return action
    
    # `_remove_position()` の修正
    def _remove_position(self, index):
        index = tf.cast(index, tf.int32)
        valid_indices = tf.boolean_mask(
            tf.range(self.positions.size(), dtype=tf.int32),
            tf.not_equal(tf.range(self.positions.size(), dtype=tf.int32), index)
        )

        # 🔥 削除済みの `pos_id` をリストとして保存
        self.valid_pos_ids = tf.gather(valid_indices, tf.range(tf.shape(valid_indices)[0]))

        filtered_positions = self.positions.gather(valid_indices)

        # 新しい `TensorArray` に入れ直す
        new_positions = tf.TensorArray(dtype=tf.float32, size=tf.shape(filtered_positions)[0], dynamic_size=True)
        for i in tf.range(tf.shape(filtered_positions)[0]):
            new_positions = new_positions.write(i, filtered_positions[i])

        self.positions = new_positions
        self.positions_index.assign(tf.cast(tf.shape(filtered_positions)[0], tf.float32))  # 🔥 float32 に変換





    def process_new_order(self, long_order_size, short_order_size, current_price, margin_rate):
        #trade_type = tf.Variable(trade_type, name="trade_type",dtype=tf.float32,trainable=True)
        if long_order_size > 0 and short_order_size > 0:
            order_margin = ((long_order_size + short_order_size + self.unfulfilled_short_open + self.unfulfilled_long_open) * current_price * margin_rate)
            margin_maintenance_flag, self.margin_maintenance_rate = update_margin_maintenance_rate(self.effective_margin, self.required_margin)
            if margin_maintenance_flag:
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
                    pos = tf.stack([long_order_size, tf.Variable(1, name="trade_type",dtype=tf.float32,trainable=True), current_price, tf.Variable(0.0,name="unrealized_profit",dtype=tf.float32,trainable=True), long_add_required_margin, tf.Variable(0.0,name="profit",dtype=tf.float32,trainable=True)])
                    print(f"Opened Buy position at {current_price}, required_margin:{self.required_margin}")
                    self.positions = self.positions.write(tf.cast(self.positions_index, tf.int32), pos)
                    self.positions_index.assign_add(1)
                    #new_order_size = order_size + self.unfulfilled_long_open
                    #order_size = new_order_size
                if short_order_size > 0:
                    short_order_size += self.unfulfilled_short_open
                    new_unfulfilled_short_open = self.unfulfilled_short_open - self.unfulfilled_short_open
                    self.unfulfilled_short_open = new_unfulfilled_short_open
                    pos = tf.stack([short_order_size, tf.Variable(-1, name="trade_type",dtype=tf.float32,trainable=True), current_price, tf.Variable(0.0,name="unrealized_profit",dtype=tf.float32,trainable=True), short_add_required_margin, tf.Variable(0.0,name="profit",dtype=tf.float32,trainable=True)])
                    print(f"Opened Sell position at {current_price}, required_margin:{self.required_margin}")
                    self.positions = self.positions.write(tf.cast(self.positions_index, tf.int32), pos)
                    self.positions_index.assign_add(1)
                    #new_order_size = order_size + self.unfulfilled_short_open
                    #order_size = new_order_size
                #pos = tf.stack([order_size, trade_type, current_price, tf.Variable(0.0,name="unrealized_profit",dtype=tf.float32,trainable=True), add_required_margin, tf.Variable(0.0,name="profit",dtype=tf.float32,trainable=True)])

                #self.positions = self.positions.write(tf.cast(self.positions_index, tf.int32), pos)

                #self.positions_index.assign_add(1)
                print(self.positions_index)
                #print(f"Opened position at {current_price}, required margin: {self.required_margin}")
                print(self.positions.stack())
                margin_maintenance_flag, self.margin_maintenance_rate = update_margin_maintenance_rate(self.effective_margin,self.required_margin)
                if margin_maintenance_flag:
                    print(f"margin maintenance rate is {self.margin_maintenance_rate},so loss cut is executed in process_new_order, effective_margin: {self.effective_margin}")
                    return


    def process_position_closure(self,long_close_position,short_close_position,current_price):
        new_unfulfilled_long_close = self.unfulfilled_long_close + long_close_position
        self.unfulfilled_long_close = new_unfulfilled_long_close
        new_unfulfilled_short_close = self.unfulfilled_short_close + short_close_position
        self.unfulfilled_short_close = new_unfulfilled_short_close

        # --- ポジション決済処理 ---
        pos_id_max = int(self.positions_index - 1)  # 現在の最大 ID
        for pos_id in range(pos_id_max + 1):  # 最大 ID までの範囲を網羅
            try:
                pos = self.positions.read(pos_id)
                print(f"In the position closure process, pos_id in update of position value{pos_id}: {pos.numpy()}")
                if tf.reduce_all(pos == 0.0):
                    print(f"this pos_id is all 0.0 so skipped")
                    continue
            except:
                continue
            size, pos_type, open_price, unrealized_profit, margin, realized_profit = tf.unstack(pos)

            if pos_type.numpy() == 1 and self.unfulfilled_long_close > 0:
                fulfilled_size = tf.minimum(self.unfulfilled_long_close,size)
                profit = fulfilled_size * (current_price - open_price)
            elif pos_type.numpy() == 1 and self.unfulfilled_long_close == 0:
                pos = tf.stack([size, pos_type, open_price, unrealized_profit, margin, realized_profit])
                self.positions = self.positions.write(tf.cast(pos_id, tf.int32), pos)
                print("unfulfilled_long_close == 0 so continue")
                continue
            elif pos_type.numpy() == -1 and self.unfulfilled_short_close > 0:
                fulfilled_size = tf.minimum(self.unfulfilled_short_close,size)
                profit = fulfilled_size * (open_price - current_price)
            elif pos_type.numpy() == -1 and self.unfulfilled_short_close == 0:
                pos = tf.stack([size, pos_type, open_price, unrealized_profit, margin, realized_profit])
                self.positions = self.positions.write(tf.cast(pos_id, tf.int32), pos)
                print("unfulfilled_short_close == 0 so continue")
                continue
            

            # 決済ロジック
            #self.effective_margin.assign_add(profit - unrealized_profit)
            update_effective_margin = self.effective_margin + profit - unrealized_profit
            self.effective_margin = update_effective_margin
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
                pos = tf.stack([size, pos_type, open_price, 0, margin+add_required_margin, 0])#once substract unrealized_profit from effective_margin, you need not do it agagin in the process of update pos, so you have to set unrealized_profit to 0.
                self.positions = self.positions.write(tf.cast(pos_id, tf.int32), pos)
                print(self.positions.stack())
                pos = [fulfilled_size, pos_type, open_price, 0, 0, profit]
                self.closed_positions.append(pos)

            else:  # 完全決済の場合
                print(f"all payment was executed")
                pos = tf.stack([fulfilled_size, pos_type, open_price, tf.Variable(0,dtype=tf.float32), 0, profit]) #we hope margin+add_required_margin==0
                #print(f"all payment: margin+add_required_margin:{margin+add_required_margin}")
                self.closed_positions.append(pos)

                self._remove_position(pos_id)
                print("after removeing position")
                print(self.positions.stack())
            print(f"Closed {'Buy' if pos_type.numpy()==1 else ('Sell' if pos_type.numpy() == -1 else 'Unknown')} position at {current_price} with profit {profit} ,grid {open_price}, Effective Margin: {self.effective_margin}, Required Margin: {self.required_margin}")

            if self.positions.size() == 0:  #this sentence needs because required_margin must be just 0 when all positions are payed, but actually not be just 0 because of rounding error.
                self.required_margin = 0
            #if pos_type.numpy() == 1.0:
            #    #self.unfulfilled_long_open.assign(long_close_position - fulfilled_size)
            #    self.unfulfilled_long_open = long_close_position - fulfilled_size
            #elif pos_type.numpy() == -1.0:
            #    #self.unfulfilled_short_open.assign(short_close_position - fulfilled_size)
            #    self.unfulfilled_short_open = short_close_position - fulfilled_size

            margin_maintenance_flag, self.margin_maintenance_rate = update_margin_maintenance_rate(self.effective_margin,self.required_margin)
            if margin_maintenance_flag:
                print(f"margin maintenance rate is {self.margin_maintenance_rate},so loss cut is executed in position closure process, effective_margin: {self.effective_margin}")
                return


    def process_position_update(self, current_price, required_margin_rate):
        """
        資産とポジションの更新を実行
        """
        # --- 含み益の更新 ---
        pos_id_max = int(self.positions_index)  # 現在の最大 ID
        for pos_id in range(pos_id_max + 1):  # 最大 ID までの範囲を網羅
            try:
                pos = self.positions.read(pos_id)
                print(f"In the update position value, pos_id in update of position value{pos_id}: {pos.numpy()}")
                if tf.reduce_all(pos == 0.0):
                    print(f"this pos_id is all 0.0 so skipped")
                    continue
            except:
                continue
            size, pos_type, open_price, before_unrealized_profit, margin, _ = tf.unstack(pos)
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
            print(f"right after updating effective_margin:{self.effective_margin}")
            #exit()
            add_required_margin = -margin + current_price * size * required_margin_rate
            self.required_margin += add_required_margin.numpy()

            pos = tf.stack([size, pos_type, open_price, unrealized_profit,margin+add_required_margin,0])
            #print(f"pos_id:{type(pos_id)}")
            #print(f"size:{type(size)}")
            #print(f"pos_type:{type(pos_type)}")
            #print(f"open_price:{type(open_price)}")
            #print(f"unrealized_profit:{type(unrealized_profit)}")
            #print(f"margin:{type(margin)}")
            #print(f"add_required_margin:{type(add_required_margin)}")
            #exit()
            self.positions = self.positions.write(tf.cast(pos_id, tf.int32), pos)
            print(f"unrealized_profit:{unrealized_profit}, before_unrealized_profit:{before_unrealized_profit}")
            print(f"updated effective margin against price {current_price} , effective Margin: {self.effective_margin}, required_margin:{self.required_margin}, pos_id:{pos_id}")
            print(self.positions.stack())

            margin_maintenance_flag, self.margin_maintenance_rate = update_margin_maintenance_rate(self.effective_margin,self.required_margin)
            if margin_maintenance_flag:
                print(f"margin maintenance rate is {self.margin_maintenance_rate},so loss cut is executed in position  process, effective_margin: {self.effective_margin}")
                return

        # --- 強制ロスカットのチェック ---
        margin_maintenance_flag, _ = update_margin_maintenance_rate(self.effective_margin, self.required_margin)
        if margin_maintenance_flag:
            print("Forced margin cut triggered.")
            pos_id_max = int(self.positions_index - 1)  # 現在の最大 ID
            for pos_id in range(pos_id_max + 1):  # 最大 ID までの範囲を網羅
                try:
                    pos = self.positions.read(pos_id)
                    print(f"In the loss cut, pos_id in update of position value{pos_id}: {pos.numpy()}")
                    if tf.reduce_all(pos == 0.0):
                        print(f"this pos_id is all 0.0 so skipped")
                        continue
                except:
                    continue
                size, pos_type, open_price, before_unrealized_profit, margin, _ = tf.unstack(pos)
                profit = (current_price - open_price) * size  # 現在の損失計算
                #self.effective_margin.assign(self.effective_margin + profit - before_unrealized_profit) # 損失分を証拠金に反映
                update_effective_margin = self.effective_margin + profit - before_unrealized_profit
                self.effective_margin = update_effective_margin
                effective_margin_max, effective_margin_min = check_min_max_effective_margin(self.effective_margin, effective_margin_max, effective_margin_min)
                self.margin_deposit.assign(self.margin_deposit + profit)
                #self.margin_deposit += profit
                self.realized_profit.assign(self.realized_profit + profit)
                #self.realized_profit += profit
                self.required_margin -= margin

                self._remove_position(pos_id)
                print("after removing position")
                print(self.positions.stack())

                pos = [size, pos_type, open_price, 0, 0, profit]
                self.closed_positions.append(pos)
                print(f"Forced Closed at currnt_price:{current_price} with open_price:{open_price}, Effective Margin: {self.effective_margin}")
            #self.required_margin = 0.0

            # 全ポジションをクリア
            self.positions = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
            self.positions_index.assign(0)

            self.required_margin = 0

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

    for agent, action in zip(agents, actions.stack()):
        action_flat = tf.reshape(action, [-1])  # 形状 (4,) に変換
        long_open_position, short_open_position, long_close_position, short_close_position = tf.unstack(action_flat)

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


            #update final remaining unfulfilled orders
            new_unfulfilled_long_open = agent.unfulfilled_long_open + final_remaining_long_open * long_open_ratio
            agent.unfulfilled_long_open = new_unfulfilled_long_open

            new_unfulfilled_short_open = agent.unfulfilled_short_open + final_remaining_short_open * short_open_ratio
            agent.unfulfilled_short_open = new_unfulfilled_short_open

            new_unfulfilled_long_close = agent.unfulfilled_long_close + final_remaining_long_close * long_close_ratio
            agent.unfulfilled_long_close = new_unfulfilled_long_close

            new_unfulfilled_short_close = agent.unfulfilled_short_close + final_remaining_short_close * short_close_ratio
            agent.unfulfilled_short_close = new_unfulfilled_short_close



# トレーニングループ
num_agents = 5
agents = [RLAgent() for _ in range(num_agents)]
generator = MarketGenerator()

states = tf.constant([100.0, 1.0, 0.01], dtype=tf.float32)
supply_and_demand = tf.constant(0.0, dtype=tf.float32)

# 記録用の辞書
history = {
    "generated_states": [],  # 生成者の出力: [価格, 流動性, スリッページ]
    "actions": [],     # 各エージェントの行動
    "agent_assets": [],       # 各エージェントの総資産
    "liquidity": [],
    "slippage" : [],
    "gen_gradients": [],
    "disc_gradients": [],
}

# トレーニングループ
generations = 10
use_rule_based = True  # 初期段階ではルールベースで流動性・スリッページを計算
required_margin_rate=tf.Variable(0.04, name="required_margin_rate",dtype=tf.float32,trainable=True)
gamma = tf.Variable(1,name="gamma",dtype=tf.float32,trainable=True)
volume = tf.Variable(0,name="volume",dtype=tf.float32,trainable=True)
actions = tf.TensorArray(dtype=tf.float32, size=len(agents),dynamic_size=True)
disc_losses = tf.TensorArray(dtype=tf.float32, size=len(agents), dynamic_size=True)
gen_losses = tf.TensorArray(dtype=tf.float32, size=len(agents), dynamic_size=True)




with tf.GradientTape(persistent=True) as gen_tape, tf.GradientTape(persistent=True) as disc_tape:
    for generation in range(generations):
        # 市場生成用の入力データ
        input_data = tf.concat([tf.reshape(states, [-1]), [supply_and_demand]], axis=0)
        # 各要素に 1e-6 を加算して対数を取る
        log_inputs = tf.math.log(input_data + 1e-6)
        generated_states = generator.generate(log_inputs)[0]
        unlog_generated_states = tf.math.exp(generated_states) - tf.constant(1e-6, dtype=tf.float32)
        current_price, current_liquidity, current_slippage = tf.split(unlog_generated_states, num_or_size_splits=3)
        # 状態を更新
        current_price = tf.reshape(current_price, [])
        current_liquidity = tf.reshape(current_liquidity, [])
        current_slippage = tf.reshape(current_slippage, [])

        #print(f"generated_states: {generated_states}, type: {type(generated_states)}")
        #watched_vars = gen_tape.watched_variables()
        #is_watched = any(current_price is var for var in watched_vars)
        #print(f"current_price watched: {is_watched}")


        #exit()


        # ルールベースでの調整
        if use_rule_based:
            # k を TensorFlow の演算子を使用して計算
            k = tf.divide(1.0, tf.add(1.0, tf.multiply(gamma, volume)))

            # current_liquidity を TensorFlow の演算子を使用して計算
            current_liquidity = tf.divide(1.0, tf.add(1.0, tf.multiply(k, tf.abs(supply_and_demand))))

            # current_slippage を TensorFlow の演算子を使用して計算
            current_slippage = tf.divide(tf.abs(supply_and_demand), tf.add(current_liquidity, tf.constant(1e-6, dtype=tf.float32)))


        # states の更新
        states = tf.stack([current_price, current_liquidity, current_slippage])

        # 各エージェントの行動
        #actions = [agent.act([agent.effective_margin, current_price]) for agent in agents]
        i = 0
        for agent in agents:
            # 各要素に 1e-6 を加算して対数を取る
            log_inputs = tf.math.log(tf.stack([agent.effective_margin + 1e-6, current_price + 1e-6]))
            # ネットワークの出力を処理し、1e-6 を減算
            unlog_action = tf.math.exp(agent.act(log_inputs)) - tf.constant(1e-6, dtype=tf.float32)
            actions = actions.write(i,unlog_action)
            i += 1
        #print(f"actions:{actions.stack()}")

        supply_and_demand = tf.reduce_sum(actions.stack())

        volume = tf.reduce_sum([tf.abs(actions.stack())])

        # 資産更新
        #print(actions.stack().shape)
        i = 0
        match_orders(agents, actions, current_price, required_margin_rate)
        for agent, action in zip(agents, actions.stack()):
            # アクションの形状 (1, 4) を (4,) に変換
            action_flat = tf.reshape(action, [-1])  # 形状 (4,)
            # 各項目を変数に分解
            long_order_size, short_order_size, long_close_position, short_close_position = tf.unstack(action_flat)
            print("\n")
            print(f"update positions of {i}th agent")
            print(f"long_order_size:{long_order_size}, short_order_size:{short_order_size}")
            #agent.update_assets(long_order_size, short_order_size, long_close_position, short_close_position, current_price)
            #agent.process_new_order(long_order_size,short_order_size,current_price,required_margin_rate)
            #agent.process_position_closure(long_close_position,short_close_position,current_price)
            agent.process_position_update(current_price,required_margin_rate)
            i += 1
            #agent.update_effective_margin = current_price * (long_order_size + short_order_size + long_close_position + short_close_position)
            #print(f"long_order_size:{long_order_size}")
            #print(f"short_order_size:{short_order_size}")
            #print(f"long_close_position:{long_close_position}")
            #print(f"short_close_position:{short_close_position}")
            #print(f"current_price:{current_price}")
            #print(f"update_effective_margin:{agent.update_effective_margin}")

        # 識別者の評価（discriminator_performance）
        #disc_tape.watch([agent.effective_margin for agent in agents])  # 必要に応じて追跡
        #discriminator_performance = tf.stack([agent.effective_margin for agent in agents])

        # 生成者の損失計算
        #if generation < generations//2:
        #    i = 0
        #    for action in (actions.stack()):
        #        action_flat = tf.reshape(action,[-1])
        #        long_order_size, short_order_size, long_close_position, short_close_position = tf.unstack(action_flat)
#
        #        initial_loss = (tf.math.log(current_price + 1e-6) \
        #                    + tf.math.log(long_order_size + 1e-6) \
        #                    + tf.math.log(short_order_size + 1e-6) \
        #                    + tf.math.log(long_close_position + 1e-6) \
        #                    + tf.math.log(short_close_position + 1e-6))
        #        initial_losses = initial_losses.write(i,initial_loss)
        #        disc_losses = disc_losses.write(i,-initial_loss)
        #        i += 1
#
#            
#
#            gen_loss = tf.reduce_mean(initial_losses.stack())
#            print(disc_losses.stack().shape)
#            #exit()
#            stacked_disc_losses = disc_losses.stack()

        #elif generation >= generations // 2:
        if generation >= 0:
            #gen_loss = tf.reduce_mean(discriminator_performance)
            gen_loss = tf.reduce_mean(tf.stack([agent.effective_margin for agent in agents]))
            #print(f"gen_loss:{gen_loss}")
            i = 0
            for agent in agents:
                disc_losses = disc_losses.write(i,-agent.effective_margin)
                i += 1
            stacked_disc_losses = disc_losses.stack()


            """
            #this code is useful for check whether a variable holds the information of calculation graph
            i = 0
            for agent in agents:
                pos_id_max = int(agent.positions_index - 1)  # 現在の最大 ID
                for pos_id in range(pos_id_max + 1):  # 最大 ID までの範囲を網羅
                    try:
                        pos = agent.positions.read(pos_id)
                        if tf.reduce_all(pos==0.0):
                            continue
                    except:
                        continue
                    size, pos_type, open_price, before_unrealized_profit, margin, _ = tf.unstack(pos)
                    print(f"margin:{margin}")
                    if pos_id >= 1:
                        break

                #disc_losses = disc_losses.write(i,size)
                gen_losses = gen_losses.write(i,generator.log_scale_factor)
            i += 1

        print(f"disc_losses:{disc_losses.stack().numpy()}")
        print(f"gen_losses:{gen_losses.stack().numpy()}")

        gen_loss = tf.reduce_mean(gen_losses.stack())
        """

        # 勾配の計算
        # 生成者の勾配
        gen_gradients = gen_tape.gradient(gen_loss, [generator.log_scale_factor] + generator.model.trainable_variables)

        print(f"gen_gradients:{gen_gradients}")
        #print(f"gen_loss: {gen_loss}")
        #print(f"generation:{generation}")
        print(f"gen_gradients: {gen_gradients}")
        #exit()

        generator.optimizer.apply_gradients(zip(gen_gradients, [generator.log_scale_factor] + generator.model.trainable_variables))

        # 識別者の勾配
        #print(f"disc_losses: {stacked_disc_losses}, type: {type(disc_losses)}") 
        disc_gradients = []
        i = 0
        #for agent, disc_loss in zip(agents, stacked_disc_losses):
        for agent in agents:
            disc_loss = disc_losses.read(i)
            #print(f"disc_loss:{disc_loss}")
            #exit()
            #print(i)
            #print(f"disc_loss: {disc_loss}, type: {type(disc_loss)}") 
            #print(f"disc_losses: {stacked_disc_losses}, type: {type(stacked_disc_losses)}")  
            #print(type(disc_loss))
            disc_gradient = disc_tape.gradient(disc_loss, agent.model.trainable_variables)
            print(f"{i}th agents' disc_gradient:{disc_gradient}")

            #print(type(disc_gradient))
            disc_gradients.append(disc_gradient)
            #print(f"disc_gradient:{disc_gradient}")
            #exit()
            agent.optimizer.apply_gradients(zip(disc_gradient, agent.model.trainable_variables))
            i += 1

        #print(f"gen_gradients: {gen_gradients}")

        #print("gen_tape variables:", gen_tape.watched_variables())
        #print("disc_tape variables:", disc_tape.watched_variables())
        #exit()

        #for agent in agents:
        #    agent.effective_margin = agent.update_effective_margin

        #print(f"trainable_variables:{generator.trainable_variables}")
        print(f"generation:{generation}")
        tf.print(f"log_scale_factor:",generator.log_scale_factor)
        print(" ")
        print(" ")
        print(" ")
        print(" ")

        # 記録用の辞書に状態を追加
        history["disc_gradients"].append(disc_gradients)
        history["generated_states"].append(generated_states.numpy())
        history["actions"].append(actions)
        history["agent_assets"].append([agent.effective_margin.numpy() for agent in agents])
        history["liquidity"].append(current_liquidity.numpy())
        history["slippage"].append(current_slippage.numpy())
        history["gen_gradients"].append(gen_gradients)

    #print(f"Generation {generation}, Best Agent Assets: {max(float(agent.effective_margin.numpy()) for agent in agents):.2f}")
    #print(f"gen_gradients:{gen_gradients}")
    #exit()

    #exit()

    ## 進化段階でルールベースを切り替え
    #if generation == generations // 2:
    #    use_rule_based = False


    # Calculate position value
    i = 0
    for agent in agents:
        position_value = 0
        if agent.positions:
            print("🔍 Before position_value calculation, positions:")
            print(agent.positions.stack())
            # TensorArray を Python の list に変換
            #positions_tensor = agent.positions.stack()
            positions_list = agent.positions.stack().numpy().tolist()

            position_value += sum(size * (current_price - open_price) if status==1 else
                         -size * (current_price - open_price) if status==-1 else
                         0 for size, status, open_price, _, _, _ in positions_list)
            #position_value = tf.reduce_sum(
            #    positions_tensor[:, 0] * tf.math.sign(positions_tensor[:, 1]) * (current_price - positions_tensor[:, 2])
            #    )

        else:
            position_value = 0

        print(f"{i}th agent")
        print(f"預託証拠金:{agent.margin_deposit.numpy()}")
        print(f"有効証拠金:{agent.effective_margin}")
        print(f"ポジション損益:{position_value}")
        print(f"確定利益:{agent.realized_profit.numpy()}")
        print(f"証拠金維持率:{agent.margin_maintenance_rate}")
        print(f"check total:{agent.margin_deposit+position_value}")
        print("\n")
        i += 1

# ファイルへの記録
with open("./txt_dir/kabu_agent-based_metatraining.txt", "w") as f:
    f.write(str(history))
os.chmod("./txt_dir/kabu_agent-based_metatraining.txt", 0o444)
print("ファイルを読み取り専用に設定しました")
