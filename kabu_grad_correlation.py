import pandas as pd
import matplotlib.pyplot as plt
from kabu_makegraph import makegraph

def get_Log_Return_Diff(file_path,):
    # エクセルファイルを読み込む
    df = pd.read_excel(file_path)
    return df['Log_Return_Diff']




def main(code,year,month,period):
  
  PRE_CORR = []
  POST_CORR = []
  for year in range(int(start_year),int(end_year)+1):
    code, year, month, period = input().split("-")

    file_path = "{}-{}-{}-{}-pre_grad_output.xlsx".format(code, year, month, period)
    a = get_Log_Return_Diff("./xlsx_dir/{}".format(file_path), "pre_grad")
    PRE_CORR.append(a)


    file_path = "{}-{}-{}-{}-post_grad_output.xlsx".format(code, year, month, period)
    b = get_Log_Return_Diff("./xlsx_dir/{}".format(file_path), "post_grad")
    POST_CORR.append(b)

  for year in range(int(start_year),int(end_year)+1):
    pre_corr = PRE_CORR[year-int(start_year)]
    post_corr = POST_CORR(year-int(start_year))

    fig, ax = plt.subplots(figsize=(15, 8))
 
    file_path = "./xlsx_dir/{}-{}-{}-{}-pre_grad_output.xlsx".format(code, year, month, period)
        # エクセルファイルを読み込む
    df = pd.read_excel(file_path)
    makegraph(df, ax, "pre_grad_corr")

    file_path = "./xlsx_dir/{}-{}-{}-{}-post_grad_output.xlsx".format(code, year, month, period)
        # エクセルファイルを読み込む
    df = pd.read_excel(file_path)
    makegraph(df, ax, "post_grad_corr")

    plt.savefig('./png_dir/{}-{}-{}-{}_grad_corr.png'.format(code, year, month, period))
    print("Saved {}-{}-{}-{}_grad_corr.png".format(code, year, month, period))

if __name__ == "__main__":
    print("write code, start_year,end_year, month,period")
    code, start_year,end_year, month,period = input.split("-")
    main(code,start_year,end_year,month,period)
