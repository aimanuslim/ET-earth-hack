import lasio, os

def convert(file):
    csvname = os.path.splitext(file)[0]+'.csv'
    las = lasio.read(file)
    df = las.df()
    df.to_csv(csvname)

# convert('Cheal-A10/LWD/LAS/Cheal_A-10_MLWD_MD_250m-1576m.las')
