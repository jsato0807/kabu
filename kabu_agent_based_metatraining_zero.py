import numpy as np
import math
import sign
import os
import random
from collections import OrderedDict
from common.layers import *
from common.functions import *


    

seed = 0
initial_log_scale_factor = np.log(np.exp(1))

def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)  # Python の乱数シード
    np.random.seed(seed)  # NumPy の乱数シード

class MarketGenerator:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # パラメータの初期化
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, hidden_size)
        self.params['b2'] = np.zeros(hidden_size)
        self.params['W3'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b3'] = np.zeros(output_size)

        self.params['log_scale_factor'] = initial_log_scale_factor  # 初期値は float 型

        # レイヤの生成
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Sigmoid1'] = Sigmoid()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        self.layers['Sigmoid2'] = Sigmoid()
        self.layers['Affine3'] = Affine(self.params['W3'], self.params['b3'])

        self.layers['LogScale'] = LogScale(self.params['log_scale_factor'])

        self.lastLayer = SoftplusWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

        
    # x:入力データ, t:教師データ
    def numerical_gradient(self, x, reward_func):
        def loss_fn():
            y = self.predict(x)
            return reward_func(y)
        
        grads = {}
        grads['W1'] = numerical_gradient(loss_fn, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_fn, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_fn, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_fn, self.params['b2'])
        grads['W3'] = numerical_gradient(loss_fn, self.params['W3'])
        grads['b3'] = numerical_gradient(loss_fn, self.params['b3'])

        grads['log_scale_factor'] = numerical_gradient(
                lambda lsf: loss_fn,
                self.params['log_scale_factor']
            )

        
        return grads

    def gradient(self, selected_margin):
        dout = 1
        for layer in reversed(self.layers.values()):
            dout = layer.backward(dout)

        # 勾配を格納
        grads = {}
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db
        grads['W3'] = self.layers['Affine3'].dW
        grads['b3'] = self.layers['Affine3'].db

        grads['log_scale_factor'] = self.layers['LogScale'].dscale

        return grads

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
"""
def validate_unfulfilled_orders(agents, previous_unfulfilled_long_open, previous_unfulfilled_short_open, 
                            previous_unfulfilled_long_close, previous_unfulfilled_short_close,
                            final_remaining_long_open, final_remaining_short_open, 
                            final_remaining_long_close, final_remaining_short_close):

    #各エージェントの unfulfilled (未約定) の新規注文・決済注文が整合性を持っているかをチェックする関数。


    # 🔹 3️⃣ 各エージェントの現在の未約定注文の合計を取得
    total_actual_unfulfilled_long_open = sum(agent.unfulfilled_long_open for agent in agents) - previous_unfulfilled_long_open
    total_actual_unfulfilled_short_open = sum(agent.unfulfilled_short_open for agent in agents) - previous_unfulfilled_short_open
    total_actual_unfulfilled_long_close = sum(agent.unfulfilled_long_close for agent in agents) - previous_unfulfilled_long_close
    total_actual_unfulfilled_short_close = sum(agent.unfulfilled_short_close for agent in agents) - previous_unfulfilled_short_close

    # 🔹 4️⃣ 整合性チェック
    print(f"Expected Unfulfilled Long Open: {final_remaining_long_open}, Actual: {total_actual_unfulfilled_long_open}")
    print(f"Expected Unfulfilled Short Open: {final_remaining_short_open}, Actual: {total_actual_unfulfilled_short_open}")
    print(f"Expected Unfulfilled Long Close: {final_remaining_long_close}, Actual: {total_actual_unfulfilled_long_close}")
    print(f"Expected Unfulfilled Short Close: {final_remaining_short_close}, Actual: {total_actual_unfulfilled_short_close}")

    assert abs(final_remaining_long_open - total_actual_unfulfilled_long_open) < 1e-6, \
        f"❌ Mismatch in unfulfilled long open orders! Expected: {final_remaining_long_open}, Got: {total_actual_unfulfilled_long_open}"

    assert abs(final_remaining_short_open - total_actual_unfulfilled_short_open) < 1e-6, \
        f"❌ Mismatch in unfulfilled short open orders! Expected: {final_remaining_short_open}, Got: {total_actual_unfulfilled_short_open}"

    assert abs(final_remaining_long_close - total_actual_unfulfilled_long_close) < 1e-6, \
        f"❌ Mismatch in unfulfilled long close orders! Expected: {final_remaining_long_close}, Got: {total_actual_unfulfilled_long_close}"

    assert abs(final_remaining_short_close - total_actual_unfulfilled_short_close) < 1e-6, \
        f"❌ Mismatch in unfulfilled short close orders! Expected: {final_remaining_short_close}, Got: {total_actual_unfulfilled_short_close}"

    print("✅ Unfulfilled order validation passed successfully!")
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

class RLAgent():
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01, initial_cash=100000):
        self.positions = []
        self.closed_positions = []
        self.effective_margin = initial_cash
        self.required_margin = 0
        self.margin_deposit = initial_cash
        self.realized_profit = 0.0
        self.long_position = 0.0
        self.short_position = 0.0
        self.unfulfilled_long_open = 0.0
        self.unfulfilled_short_open = 0.0
        self.unfulfilled_long_close = 0.0
        self.unfulfilled_short_close = 0.0
        self.margin_maintenance_rate = np.inf
        self.margin_maintenance_flag = False
        self.effective_margin_max = -np.inf
        self.effective_margin_min = np.inf

        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Sigmoid1'] = Sigmoid()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        self.layers['Sigmoid2'] = Sigmoid()
        self.layers['Affine3'] = Affine(self.params['W3'], self.params['b3'])


        self.lastLayer = ReluWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def numerical_gradient(self, x, reward_func):
        def loss_fn():
            y = self.predict(x)
            return reward_func(y)
        
        grads = {}
        grads['W1'] = numerical_gradient(loss_fn, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_fn, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_fn, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_fn, self.params['b2'])
        grads['W3'] = numerical_gradient(loss_fn, self.params['W3'])
        grads['b3'] = numerical_gradient(loss_fn, self.params['b3'])
        
        return grads

    def gradient(self, x, reward_func):
        # forward
        y = self.predict(x)
        loss = -reward_func(y)

        # backward
        dout = 1
        for layer in reversed(self.layers.values()):
            dout = layer.backward(dout)

        # 勾配を格納
        grads = {}
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db
        grads['W3'] = self.layers['Affine3'].dW
        grads['b3'] = self.layers['Affine3'].db

        return grads
    
    # `_remove_position()` の修正
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

# トレーニングループ
num_agents = 5
set_seed(seed)
agents = [RLAgent() for _ in range(num_agents)]
generator = MarketGenerator()

states = [100.0, 1.0, 0.01]
supply_and_demand = 0.0

# 記録用の辞書
history = {
    "generated_states": [],  # 生成者の出力: [価格, 流動性, スリッページ]
    "actions": [],     # 各エージェントの行動
    "agent_assets": [],       # 各エージェントの総資産
    "liquidity": [],
    "slippage" : [],
    "gen_gradients": [],
    "disc_gradients": [],
    "gen_loss": [],
    "disc_losses": [],
    "log_scale_factor": [],
}

# トレーニングループ
generations = 165
use_rule_based = True  # 初期段階ではルールベースで流動性・スリッページを計算
required_margin_rate=0.04
gamma = 1
volume = 0
actions = []
disc_losses = []
gen_losses = []


for generation in range(generations):
    set_seed(generation)

    ## 各要素に 1e-6 を加算して対数を取る
    # 供給需要の符号付き log 変換
    log_supply_and_demand = sign(supply_and_demand) * math.log(abs(supply_and_demand) + 1e-6)

    # generator の入力データ
    log_inputs = np.concat([
        math.log(np.reshape(states,[-1]) + 1e-6),
        [log_supply_and_demand]
    ], axis=0)

    generated_states = generator.generate(log_inputs)[0]
    unlog_generated_states = math.exp(generated_states) - 1e-6
    current_price, current_liquidity, current_slippage = np.split(unlog_generated_states, num_or_size_splits=3)


    # ルールベースでの調整
    if use_rule_based:
        # k を 計算
        k = 1/(1+gamma*volume)

        # current_liquidity を 計算
        current_liquidity = 1/(1+k*abs(supply_and_demand))

        # current_slippage を 計算
        current_slippage = abs(supply_and_demand)/(current_liquidity + 1e-6)


    # states の更新
    states = current_price, current_liquidity, current_slippage]

    # 各エージェントの行動
    #actions = [agent.act([agent.effective_margin, current_price]) for agent in agents]
    i = 0
    for agent in agents:
        # 各要素に 1e-6 を加算して対数を取る
        log_inputs = math.log([agent.effective_margin + 1e-6, current_price + 1e-6])
        # ネットワークの出力を処理し、1e-6 を減算
        unlog_action = math.exp(agent.predict(log_inputs)) - 1e-6
        actions.append(unlog_action)
        i += 1
    #print(f"actions:{actions.stack()}")

    volume = sum(abs(a) for a in actions)

    # 資産更新
    #print(actions.stack().shape)
    i = 0
    match_orders(agents, actions, current_price, required_margin_rate)

    supply_and_demand = sum([
        agent.unfulfilled_long_open
        - agent.unfulfilled_short_open
        - agent.unfulfilled_long_close
        + agent.unfulfilled_short_close
        for agent in agents
    ])
    print(f"supply_and_demand:{supply_and_demand}")

    for agent, action in zip(agents, actions):
        # アクションの形状 (1, 4) を (4,) に変換
        action_flat = np.reshape(action, [-1])  # 形状 (4,)
        # 各項目を変数に分解
        long_order_size, short_order_size, long_close_position, short_close_position = action_flat

        #reset calculation graph and update effective_margin by using outputs of models with multiplying 0 in order not to change the value of effective_margin itself. 

        print("\n")
        print(f"update positions of {i}th agent")
        print(f"long_order_size:{long_order_size}, short_order_size:{short_order_size}")
        #agent.update_assets(long_order_size, short_order_size, long_close_position, short_close_position, current_price)
        #agent.process_new_order(long_order_size,short_order_size,current_price,required_margin_rate)
        #agent.process_position_closure(long_close_position,short_close_position,current_price)
        agent.process_position_update(current_price,required_margin_rate)
        #agent.update_effective_margin = current_price * (long_order_size + short_order_size + long_close_position + short_close_position)
        print(f"{i}th long_order_size:{long_order_size}")
        print(f"{i}th short_order_size:{short_order_size}")
        print(f"{i}th long_close_position:{long_close_position}")
        print(f"{i}th short_close_position:{short_close_position}")
        i += 1
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
#         gen_loss = tf.reduce_mean(initial_losses.stack())
#         print(disc_losses.stack().shape)
#         #exit()
#         stacked_disc_losses = disc_losses.stack()

    #elif generation >= generations // 2:
    if generation >= 0:
        #gen_loss = tf.reduce_mean(discriminator_performance)
        # エージェントの effective_margin を TensorFlow のテンソルとして格納
        effective_margins = [agent.effective_margin for agent in agents]
        # ランダムなインデックスを取得
        random_index = random.randint(0, len(agents) - 1)
        # ランダムに選択した effective_margin を取得
        selected_margin = effective_margins[random_index]
        print(f"selected agent is {random_index}th agent")
        #gen_loss = tf.reduce_mean(tf.stack([agent.effective_margin for agent in agents]))
        gen_loss = selected_margin
        #print(f"gen_loss:{gen_loss}")
        i = 0
        for agent in agents:
            disc_losses.append(-agent.effective_margin)
            i += 1
        #stacked_disc_losses = disc_losses.stack()


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

    gen_gradients = generator.gradient()

    print(f"gen_gradients:{gen_gradients}")
    print(f"gen_loss: {gen_loss}")
    #print(f"generation:{generation}")
    #print(f"gen_gradients: {gen_gradients}")
    #exit()

    generator.optimizer.apply_gradients(zip(gen_gradients, [generator.log_scale_factor] + generator.model.trainable_variables))

    # 識別者の勾配
    #print(f"disc_losses: {stacked_disc_losses}, type: {type(disc_losses)}") 
    disc_gradients = []
    i = 0
    #for agent, disc_loss in zip(agents, stacked_disc_losses):
    disc_losses_list = []
    for agent in agents:
        disc_loss = disc_losses.read(i)
        disc_losses_list.append(disc_loss)
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

    print(f"trainable_variables:{generator.trainable_variables}")
    print(f"generation:{generation}")
    print(f"log_scale_factor:{generator.log_scale_factor.numpy()}")
    print(f"current_price:{current_price}")
    print(" ")
    print(" ")
    print(" ")
    print(" ")

    # 記録用の辞書に状態を追加
    history["disc_gradients"].append(disc_gradients)
    history["disc_losses"].append(disc_losses_list)
    history["generated_states"].append(generated_states.numpy())
    history["actions"].append(stacked_actions)
    history["agent_assets"].append([agent.effective_margin.numpy() for agent in agents])
    history["liquidity"].append(current_liquidity.numpy())
    history["slippage"].append(current_slippage.numpy())
    history["gen_gradients"].append(gen_gradients)
    history["gen_loss"].append(gen_loss)
    history["log_scale_factor"].append(generator.log_scale_factor.numpy())

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
    if agent.positions and agent.margin_maintenance_flag==False:
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
    print(f"ロスカットしたか:{agent.margin_maintenance_flag}")
    print("\n")
    i += 1

# ファイルへの記録
with open(f"./txt_dir/kabu_agent_based_metatraining_seed-{seed}_lsf-{log_scale_factor}_generations-{generations}.txt", "w") as f:
    f.write(str(history))

