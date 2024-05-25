import pandas as pd
import matplotlib.pyplot as plt
from kabu_makegraph import makegraph


def get_Log_Return_Diff(file_path):
    # エクセルファイルを読み込む
    df = pd.read_excel(file_path)
    return df['Log_Return_Diff']

def get_corr(data,state):

	df = pd.DataFrame(data)

	# 相関係数の計算
	correlation = df.corr()
  
	df["{}_corr".format(state)] = correlation
	return df






if __name__ == "__main__":
    print("write code, start_year,end_year, month,period")
    code, start_year,end_year, month,period = input.split("-")
    
    PRE_CORR = []
    POST_CORR = []

    for year in range(start_year,end_year+1):

      file_path = "{}-{}-{}-{}-pre_grad_output.xlsx".format(code, year, month, period)
      a = get_Log_Return_Diff(file_path)
      PRE_CORR.append(a)

      file_path = "{}-{}-{}-{}-post_grad_output.xlsx".format(code, year, month, period)
      b = get_Log_Return_Diff(file_path)
      POST_CORR.append(b)

    fig, ax = plt.subplots(figsize=(15, 8))
    
    df = get_corr(PRE_CORR,"pre_corr")
    y_label = "pre_corr"
    makegraph(df,ax,"pre_corr")
    
    df = get_corr(POST_CORR,"post_corr")
    y_label = "post_corr"
    makegraph(df,ax,"post_corr")
    
    # グラフを保存
    plt.savefig('./png_dir/{}-{}-{}-{}_corr.png'.format(code, year, month, period))
    print("Saved {}-{}-{}-{}_corr.png".format(code, year, month, period))