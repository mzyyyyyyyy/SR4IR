import os


if __name__ == '__main__':
    
    file_path = '/mnt/data/dataset/Pascal-VOC/VOCtrainval_11-May-2012.tar'
    if not os.path.exists(file_path):
        print("FAILED: The data path datasets/VOCtrainval_11-May-2012.tar does NOT EXIST")
    else:
        # make directories
        os.makedirs('/mnt/data/dataset/VOC', exist_ok=True)

        # untar VOC
        print('Untar VOCtrainval_11-May-2012.tar ...')
        os.system('tar -xf file_path -C /mnt/data/dataset/VOC')

        # # remove .tar file
        # if os.path.exists('datasets/VOCtrainval_11-May-2012.tar'):
        #     os.remove('datasets/VOCtrainval_11-May-2012.tar')

        print('Done!')
