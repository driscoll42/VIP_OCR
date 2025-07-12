import multiprocessing as mp
import time
import pytesseract
from pdf2image import convert_from_path
import pandas as pd
from multiprocessing import cpu_count
from functools import partial


def ocr_core(ocr_df, directory, row_num):
    """
    This function will handle the core OCR processing of images.
    """
    images = convert_from_path(f'''{directory}/{ocr_df.iloc[row_num]['item_filename']}.pdf''', thread_count=cpu_count())
    pdf_text = ''
    for i in range(len(images)):
        img = images[i]
        if i == 0:
            pdf_text = f'{pytesseract.image_to_string(img)}'
        else:
            pdf_text = f'{pdf_text} {pytesseract.image_to_string(img)}'  # Should this be a new character line?

    ocr_df.at[row_num, 'tesseract_multi'] = pdf_text

    return pd.DataFrame(ocr_df.iloc[row_num]).T


if __name__ == '__main__':
    ocr_df = pd.read_excel('ocr_df.xlsx', index_col=0)
    directory = 'test_files'
    cpu_cnts = [1, 2, 4, 8, 16, 32]
    for num_cpu in cpu_cnts:
        print("Number of CPUs:", num_cpu)
        start_time = time.time()
        # num_cpu = max(mp.cpu_count() - 2, 1)
        num_runs = [x for x in range(len(ocr_df))]

        with mp.Pool(num_cpu) as pool:
            func = partial(ocr_core, ocr_df, directory)
            df = pd.concat(pool.map(func, num_runs))
        # print(df)
        df.to_excel('ocr_df_multitest.xlsx')
        pool.close()
        print('Execution Time:', round(time.time() - start_time, 2), 'Num Cores', num_cpu)
        # df = df.sort_values(by='Score', ascending=False).reset_index(drop=True)
        # print(df)
        # print(df[df['Score'] >= 90].count().tolist()[0])
        # print(100 * round(df[df['Score'] >= 90].count().tolist()[0] / total_to_run, 4))
