import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='GAN NetWork')
    parser.add_argument('--batch_size', type=int, default=8, help='input batch size')
    parser.add_argument('--drop_ratio', default=0,
                        help='Probability to drop a cropped area if the label is empty. All empty patches will be dropped for 0 and accept all cropped patches if set to 1')
    parser.add_argument('--gpu_ids',type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--name', type=str, default='experiment_name',
                        help='name of the experiment. It decides where to store samples and models')
    parser.add_argument('--cli_dir', type=str, default='./ADNI_Clinical.csv', help='pet input path')
    parser.add_argument('----model', type=str, default='Triple+KAN+self')
    parser.add_argument("--seed", default=42, type=int, help="seed given by LinkStart.py on cross Val")

    parser.add_argument("--fold", default=0, type=int, help="0~4")
    parser.add_argument("--data_parallel", default=0, type=int, help="test on server")

    ## ADNI1_experiment_arg:
    # parser.add_argument('--train_log_path', type=str, default='ADNI1_train.txt', help='file path to record the train result')
    # parser.add_argument('--test_log_path', type=str, default='ADNI1_test.txt', help='file path to record the test result')

    ## data_dir
    parser.add_argument('--mri_dir', type=str, default='/mntcephfs/med_dataset/huxiangyang/ADNI/freesurfer/ADNI1/MRI', help='mri input path')
    parser.add_argument('--pet_dir', type=str, default='/mntcephfs/med_dataset/huxiangyang/ADNI/freesurfer/ADNI1/PET', help='pet input path')

    ## match_data
    parser.add_argument('--csv_train_file_path', type=str, default='/mntcephfs/lab_data/wangcm/hxy/ADNI_Label/match/ADNI1_train.csv', help='csv file path to train')
    parser.add_argument('--csv_test_file_path', type=str, default='/mntcephfs/lab_data/wangcm/hxy/ADNI_Label/match/ADNI1_test.csv', help='csv file path to test')
    parser.add_argument('--csv_eval_file_path', type=str, default='/mntcephfs/lab_data/wangcm/hxy/ADNI_Label/match/ADNI1_validation.csv', help='csv file path to eval')

    ## all_data
    # parser.add_argument('--csv_train_file_path', type=str, default='/mntcephfs/lab_data/wangcm/hxy/ADNI_Label/all/ADNI1_train.csv', help='csv file path to train')
    # parser.add_argument('--csv_test_file_path', type=str, default='/mntcephfs/lab_data/wangcm/hxy/ADNI_Label/all/ADNI1_test.csv', help='csv file path to test')
    # parser.add_argument('--csv_eval_file_path', type=str, default='/mntcephfs/lab_data/wangcm/hxy/ADNI_Label/all/ADNI1_validation.csv',help='csv file path to eval')

    parser.add_argument('--best_result_model_path', type=str,default='best_result.pth', help='the best result model path')
    parser.add_argument('--device', type=str, default='cuda:0', help='device to run')
    parser.add_argument('--workers', default=0, type=int, help='number of data loading workers')
    parser.add_argument('--init_type', type=str, default='kaiming', help='network initialization [normal|xavier|kaiming|orthogonal]')
    parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
    parser.add_argument('--class_num', type=int, default=2, help='the number of class')
    parser.add_argument('--m', type=float, default=0.999, help='ema momentum decay for prototype update scheme')
    parser.add_argument('--checkpoints_dir', type=str, default='./result', help='models are saved here')
    parser.add_argument("--epochs", type=int, default=150, help="number of epochs to train")
    parser.add_argument('--print_freq', type=int, default=1,help='frequency of showing training results on console')
    parser.add_argument('--save_epoch_freq', type=int, default=20,help='frequency of saving checkpoints at the end of epochs')
    parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
    parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate for adam')
    parser.add_argument('--lr_decay', type=float, default=0.95, help='initial lambda decay value')
    parser.add_argument('--lr_policy', type=str, default='exp', help='learning rate policy: lambda|step|plateau|cosine')
    parser.add_argument('--interpolation_lambda', type=float, default=20.0, help='interpolation strength')
    args = parser.parse_args()

    return args