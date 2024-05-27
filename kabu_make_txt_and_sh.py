import pandas as pd
from glob import glob
from kabu_event_study import extract_year_month

def write_txt(file_path,code,year,month,period):

	with open(file_path, 'w') as f:
		f.write("{}-{}-{}-{}".format(code,year,month,period))	

	print(f"DataFrame has been written to {file_path}")

def make_sh(filename,code_str,month_str,start_year,end_year,period,FLAG=False):
	###kabu.shを初期化
	f = open("{}".format("{}.sh".format(filename)),"w")
	f.close()

	for year in range(int(start_year),int(end_year)+1):
		for i in range(len(code_str)):	
			code = code_str[i]
			month = month_str[i]

			if month == "nan":
				print("there is no data in month")
				continue

			if "," in month:
				month = month.split(",")
				for j in range(len(month)):
					file_path = "./txt_dir/{}-{}-{}-{}.txt".format(code,year,month[j],period)
					write_txt(file_path,code,year,month[j],period)

					filenames = glob("./txt_dir/{}*{}.txt".format(code,period))
					min_year = sorted(filenames, key=lambda x: extract_year_month(x))[0].split("-")[1]
					min_month = sorted(filenames, key=lambda x: extract_year_month(x))[0].split("-")[2]

					f = open("{}.sh".format(filename),"a")

					

					if FLAG:
						if year == int(min_year) and month[j] == min_month:
							f.write("python {}.py < ./txt_dir/{}-{}-{}-{}.txt\n".format(filename,code,year,month[j],period))

						else:
							pass

					else:
						f.write("python {}.py < ./txt_dir/{}-{}-{}-{}.txt\n".format(filename,code,year,month[j],period))
					f.close()
					print("write {}-{}-{}-{} in {}.sh".format(code,year,month[j],period,filename))

			else:
				file_path = "./txt_dir/{}-{}-{}-{}.txt".format(code,year,month,period)
				j = 0
				write_txt(file_path,code,year,month,period)

				f = open("{}".format(filename),"a")
				f.write("python {}.py < ./txt_dir/{}-{}-{}-{}.txt\n".format(filename,code,year,month[j],period))
				f.close()
				print("write {}-{}-{}-{}in {}.sh".format(code,year,month,period,filename))



if __name__ == '__main__':
# エクセルファイルのパスを指定
	file_path = 'kabu.xlsx'

	# エクセルファイルを読み込む
	df = pd.read_excel(file_path, sheet_name='Sheet1')  # 'Sheet1'は読み込みたいシート名

	# データフレームの内容を確認
	#print(df)
	#print(type(df["code"]))

	code_str = df["code"].astype(str)
	month_str = df["month"].astype(str)

	#print(month_str)


	print("input start_year,end_year and period")
	start_year,end_year,period = input().split()


	filename = "kabu"
	#make_sh(filename,code_str,month_str,start_year,end_year,period)

	filename = "kabu_makegraph"
	#make_sh(filename,code_str,month_str,start_year,end_year,period)

	filename = "kabu_event_study"
	make_sh(filename,code_str,month_str,start_year,end_year,period,FLAG=True)