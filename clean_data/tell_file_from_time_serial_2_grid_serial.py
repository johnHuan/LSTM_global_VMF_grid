import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from time_util import *


def multi_thread_tell(path_with_year, each_year_files):
    for file in each_year_files:  # each year VMF file
        current_file = path_with_year + file
        for sample in open(current_file):
            if (sample.startswith('!')):
                continue
            else:
                lon = sample[0:6].rstrip().lstrip()
                lat = sample[7:13].rstrip().lstrip()
                ztd = sample[38:45]
                target_file = target_base_dir + year + '/' + lon + '_' + lat + '.txt'
                fobj = open(target_file, mode='a')
                current_year = int(file[5:9])  # 2020
                current_month = int(file[9:11])  # 9
                current_day = int(file[11:13])  # 13
                current_doy = get_day_of_year(current_year, current_month, current_day)  # 257
                current_epoch = int(file[15:17]) / 6 * 0.25
                element_x = current_doy + current_epoch
                content = "{:6.2f}".format(element_x) + ' ' + ztd
                fobj.write(content)
                fobj.write('\n')
                fobj.close()
    print('finished ' + year + ' year...')


if __name__ == '__main__':
    source_base_dir = 'E:/shell/VMF3_OP/source_data/'
    target_base_dir = 'E:/shell/VMF3_OP/grid/'
    source_files = os.listdir(source_base_dir)
    source_files.sort()
    with ThreadPoolExecutor(max_workers=8) as t:
        for year in source_files:  # year
            path_with_year = source_base_dir + year + '/'
            each_year_files = os.listdir(path_with_year)
            each_year_files.sort()
            multi_thread_tell(path_with_year, each_year_files)
