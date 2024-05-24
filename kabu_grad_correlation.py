import pandas as pd
import matplotlib.pyplot as plt

def get_Log_Return_Diff(file_path,):
    # エクセルファイルを読み込む
    df = pd.read_excel(file_path)
    return df['Log_Return_Diff']




def main(code,year,month,period):
  
  PRE_CORR = []
  POST_CORR = []
  for year in range(start_year,end_year+1):
    code, year, month, period = input().split("-")

    file_path = "{}-{}-{}-{}-pre_grad_output.xlsx".format(code, year, month, period)
    get_Log_Return_Diff("./xlsx_dir/{}".format(file_path), "pre_grad")

    file_path = "{}-{}-{}-{}-post_grad_output.xlsx".format(code, year, month, period)
    get_Log_Return_Diff("./xlsx_dir/{}".format(file_path), "post_grad")

    # グラフを保存
    plt.savefig('./png_dir/{}-{}-{}-{}_grad.png'.format(code, year, month, period))
    print("Saved {}-{}-{}-{}_grad.png".format(code, year, month, period))

if __name__ == "__main__":
    print("write code, start_year,end_year, month,period")
    code, start_year,end_year, month,period = input.split("-")
    main(code,start_year,end_year,month,period)
