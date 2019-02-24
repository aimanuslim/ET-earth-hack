import lasio, os

def convert(file):
    csvname = os.path.splitext(file)[0]+'.csv'
    las = lasio.read(file)
    df = las.df()
    df.drop_duplicates(inplace=True)
    df.to_csv(csvname)

#convert(r'data\Cheal-C4\Wireline\LAS\PEX-HRLA-BHC-Rollers_Cheal C-4_Main Pass__05Jul12_001412.las')
