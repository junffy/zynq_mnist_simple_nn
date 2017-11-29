
# MNISTのデータをCの配列に出力し、ファイルに書き込み

# coding: utf-8
import sys, os
sys.path.append(os.pardir)

import numpy as np
from dataset.mnist import load_mnist
import datetime

OUTPUT_DATA_NUM = 100 # 出力するMNISTのテストデータ数 10000までの数

# データの読み込み
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

f = open("mnist_data.h", 'w')
todaytime = datetime.datetime.today()
f.write('// mnist_data.h\n')
strdtime = todaytime.strftime("%Y/%m/%d %H:%M:%S")
f.write('// {0} by marsee\n'.format(strdtime))
f.write("\n")

f.write('ap_ufixed<8, 0, AP_TRN_ZERO, AP_SAT> t_train['+str(OUTPUT_DATA_NUM)+'][784] = {\n')
for i in range(OUTPUT_DATA_NUM):
    f.write("\t{")
    for j in range(x_test.shape[1]):
        f.write(str(x_test[i][j]))
        if (j==x_test.shape[1]-1):
            if (i==OUTPUT_DATA_NUM-1):
                f.write("}\n")
            else:
                f.write("},\n")
        else:
            f.write(", ")
f.write("};\n")

f.write('int t_train_256['+str(OUTPUT_DATA_NUM)+'][784] = {\n')
for i in range(OUTPUT_DATA_NUM):
    f.write("\t{")
    for j in range(x_test.shape[1]):
        f.write(str(int(x_test[i][j]*256)))
        if (j==x_test.shape[1]-1):
            if (i==OUTPUT_DATA_NUM-1):
                f.write("}\n")
            else:
                f.write("},\n")
        else:
            f.write(", ")
f.write("};\n")

f.write("\n")
f.write('float t_test['+str(OUTPUT_DATA_NUM)+'][784] = {\n')
for i in range(OUTPUT_DATA_NUM):
    f.write("\t{")
    for j in range(t_test.shape[1]):
        f.write(str(t_test[i][j]))
        if (j==t_test.shape[1]-1):
            if (i==OUTPUT_DATA_NUM-1):
                f.write("}\n")
            else:
                f.write("},\n")
        else:
            f.write(", ")
f.write("};\n")
f.close() 
