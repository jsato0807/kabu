import difflib

def compare_files_and_save(file1_path, file2_path, output_path):
    # ファイルを読み込む
    with open(file1_path, 'r', encoding='utf-8') as file1:
        file1_lines = file1.readlines()
    with open(file2_path, 'r', encoding='utf-8') as file2:
        file2_lines = file2.readlines()

    # 差分を比較
    diff = difflib.unified_diff(
        file1_lines, file2_lines,
        fromfile='File1', tofile='File2',
        lineterm=''
    )

    # 差分をファイルに書き込む
    with open(output_path, 'w', encoding='utf-8') as output_file:
        output_file.write("\n".join(diff))
    
    print(f"差分の内容を {output_path} に保存しました。")

# ファイルパスを指定
file1_path = './kabu_backtest_debug.py'
file2_path = './kabu_backtest.py'
output_path = './txt_dir/check_filesdiff.txt'

# 比較して結果を保存
compare_files_and_save(file1_path, file2_path, output_path)
