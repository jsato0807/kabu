import pandas as pd

def open_file(file_path,i):
						# エクセルファイルを読み込む
	df = pd.read_excel(file_path, sheet_name='Sheet1')  # 'Sheet1'は読み込みたいシート名
	
	code_str = df["code"].astype(str)
	month_str = df["month"].astype(str)
	code = code_str[i]
	year = 2023
	month = month_str[i]

	return code_str, month_str, code, year, month

# エクセルファイルのパスを指定
def make_sh(file_path):

		#print(month_str)

	sh_filename = "kabu_makegraph.sh"

		###kabu.shを初期化
	f = open("{}".format(sh_filename),"w")
	f.close()

	print("input period")
	period = input()

	for i in range(len(code_str)):	


			if month == "nan":
				print("there is no data in month")
				continue

			if "," in month:
				month = month.split(",")
				for j in range(len(month)):
					file_path = "./txt_dir/{}-{}-{}-{}.txt".format(code,year,month[j],period)

					code_str, month_str, code, year, month = open_file(file_path,i)
		
					f = open("{}".format(sh_filename),"a")
					f.write("python kabu_makegraph.py < {}\n".format(file_path))
					f.close()
					print("write {}-{}-{}-{} in {}".format(code,year,month[j],period,sh_filename))

			else:
				file_path = "./txt_dir/{}-{}-{}-{}.txt".format(code,year,month,period)

				code_str, month_str, code, year, month = open_file(file_path,i)

				f = open("{}".format(sh_filename),"a")
				f.write("python kabu_makegraph.py < {}".format(file_path))
				f.close()
				print("write {}-{}-{}-{} in {}".format(code,year,month,period,sh_filename))


if __name__ == '__main__':

