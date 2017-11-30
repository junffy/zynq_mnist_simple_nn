def fwrite_weight(weight, wfile_name, float_wt_name, fixed_wt_name, MAGNIFICATION, row_size, column_size):
  import datetime
  import numpy as np
  
  f = open(wfile_name, 'w')
  todaytime = datetime.datetime.today()
  f.write('// '+wfile_name+'\n')
  strdtime = todaytime.strftime("%Y/%m/%d %H:%M:%S")
  f.write('// {0} by jun-i\n'.format(strdtime))
  f.write("\n")
  
  f.write('const float '+float_wt_name+'['+str(row_size)+']['+str(column_size)+'] = {\n')
  for i in range(weight.shape[0]):
    f.write("\t{")
    for j in range(weight.shape[1]):
      f.write(str(weight[i][j]))
      if (j==weight.shape[1]-1):
        if (i==weight.shape[0]-1):
          f.write("}\n")
        else:
          f.write("},\n")
      else:
        f.write(", ")
  f.write("};\n")

  f.write("\n")
  f.write('const ap_fixed<'+str(int(np.log2(MAGNIFICATION))+1)+', 1, AP_TRN_ZERO, AP_SAT> '+fixed_wt_name+'['+str(row_size)+']['+str(column_size)+'] = {\n')
  for i in range(weight.shape[0]):
    f.write("\t{")
    for j in range(weight.shape[1]):
      w_int = int(weight[i][j]*MAGNIFICATION+0.5)
      if (w_int > MAGNIFICATION-1):
        w_int = MAGNIFICATION-1
      elif (w_int < -MAGNIFICATION):
        w_int = -MAGNIFICATION
      f.write(str(w_int/MAGNIFICATION))
      if (j==weight.shape[1]-1):
        if(i==weight.shape[0]-1):
          f.write("}\n")
        else:
          f.write("},\n")
      else:
        f.write(", ")
  f.write("};\n")

  f.close()

# how to use
"""
MAGNIFICATION = 2 ** (9-1)
fwrite_weight(network.params['W1'], 'af1_weight.h', 'af1_fweight', 'af1_weight', MAGNIFICATION, 784, 50)
fwrite_weight(network.params['W2'], 'af2_weight.h', 'af2_fweight', 'af2_weight', MAGNIFICATION, 50, 10)
"""
