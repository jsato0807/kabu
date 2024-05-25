import pandas as pd
import matplotlib.pyplot as plt

def makegraph(df, ax, label):

    # グラフを作成
    ax.plot(df["Date"], df["Log_Return_Diff"], label=label)

    # グラフの装飾
    ax.set_title('Stock Prices Over Time')
    ax.set_xlabel('Date')
    ax.set_ylabel('Log_Return_Diff')
    ax.legend()
    ax.grid(True)

def main():
    fig, ax = plt.subplots(figsize=(15, 8))

    code, year, month, period = input().split("-")

    file_path = "./xlsx_dir/{}-{}-{}-{}-pre_grad_output.xlsx".format(code, year, month, period)
        # エクセルファイルを読み込む
    df = pd.read_excel(file_path)
    makegraph(df, ax, "pre_grad")

    file_path = "./xlsx_dir/{}-{}-{}-{}-post_grad_output.xlsx".format(code, year, month, period)
        # エクセルファイルを読み込む
    df = pd.read_excel(file_path)
    makegraph(df, ax, "post_grad")

    # グラフを保存
    plt.savefig('./png_dir/{}-{}-{}-{}_grad.png'.format(code, year, month, period))
    print("Saved {}-{}-{}-{}_grad.png".format(code, year, month, period))

if __name__ == "__main__":
    main()
