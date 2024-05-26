from glob import glob
from kabu_event_study import get_month_of_devidend
from PIL import Image
import matplotlib.pyplot as plt

code, start_year, end_year, period = input().split()

# get_month_of_devidend関数を使ってmonthsを取得
months = get_month_of_devidend(code)

# ファイルの数だけ行と列を計算
num_files = len(months)
num_cols = 4  # 1行あたりの最大の列数
num_rows = (num_files + num_cols - 1) // num_cols  # 切り上げ

fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5*num_rows))

# 1次元のaxesを2次元の配列に変換
axes = axes.reshape((num_rows, num_cols))

for i, month in enumerate(months):
    row = i // num_cols
    col = i % num_cols

    ax = axes[row, col] if num_files > 1 else axes[col]
    ax.axis('off')  # 初期状態ではすべてのサブプロットを非表示にする
    for year in range(int(start_year), int(end_year) + 1):
        filenames = glob(f"./png_dir/{code}-{year}-{month}-{period}*.png")
        if filenames:  # ファイルが見つかった場合のみ処理する
            for filename in filenames:
                # 画像の読み込み 
                img = Image.open(filename)
                ax.imshow(img)
                ax.axis('on')  # 軸を表示する
                break  # 最初の画像のみ表示する
            break  # 最初の年のみ処理する

# タイトルを追加
fig.suptitle('Images', fontsize=8)
fig.savefig("./png_dir/kabu_makegraph_generalize.png")
plt.show()


