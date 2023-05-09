# Meta-NA

## Usage

1. Dataset Download.
For MOSI and MOSEI dataset, you need to download them Using the following link. (aligned_50.pkl).

- [BaiduYun Disk](https://pan.baidu.com/s/1XmobKHUqnXciAm7hfnj2gg) `code: mfet`
- [Google Drive](https://drive.google.com/drive/folders/1A2S4pqCHryGmiqnNSPLv7rEg63WvjCSk?usp=sharing)

And move them into the src/data dir, and rename them using `<dataset>_unaligned.pkl` format. e.g. `mosi_unaligned.pkl`.

For reproduce experiments in our paper, Please run the .sh file in src/ dir.

```
cd src && sh train_<dataset_name>.sh
```