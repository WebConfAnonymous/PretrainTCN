import torch
import argparse
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import sys,time
sys.path.append("../../")
from Methods.TCN_num_pretrain.model import TCN
from Methods.utils import data_generator_num


parser = argparse.ArgumentParser(description='Sequence Modeling - The Popularity Prediction Problem')
parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help='batch size (default: 32)')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA (default: True)')
parser.add_argument('--cuda_index', type=int, default=0,
                    help='index of cuda (default: 0)')
parser.add_argument('--dropout', type=float, default=0.0,
                    help='dropout applied to layers (default: 0.0)')
parser.add_argument('--clip', type=float, default=-1,
                    help='gradient clip, -1 means no clip (default: -1)')
parser.add_argument('--epochs', type=int, default=20,
                    help='upper epoch limit (default: 20)')
parser.add_argument('--ksize', type=int, default=8,
                    help='kernel size (default: 8)')
parser.add_argument('--levels', type=int, default=9,
                    help='# of levels (default: 9)')
parser.add_argument('--log_interval', type=int, default=100, metavar='N',
                    help='report interval (default: 100')

parser.add_argument('--lr', type=float, default=4e-3,
                    help='initial learning rate (default: 4e-3)')
parser.add_argument('--weight_decay', type=float, default=1e-5,
                    help='weight of l2 regularizations(default: 1e-5)')
parser.add_argument('--mlp_hid_size', type=int, default=8,
                    help='hidden size in mlp (default: 8)')
parser.add_argument('--if_fix', type=int, default=0,
                    help='if fix tcn.0 means no, 1 means yes (default: 0)')
parser.add_argument('--max_label', type=int, default=12,
                    help='max number of labels (default: 12)')
parser.add_argument('--time_slices', type=int, default=1800,
                    help='time slices of each cascades parts (default: 1800)')

parser.add_argument('--optim', type=str, default='Adam',
                    help='optimizer to use (default: Adam)')
parser.add_argument('--nhid', type=int, default=8,
                    help='number of hidden units per layer (default: 8)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed (default: 1111)')
parser.add_argument('--max_try', type=int, default=20,
                    help='number of max try (default: 20')

parser.add_argument('--ob_time', type=int, default=3600,
                    help='observation time (default: 3600')
parser.add_argument('--interval_time', type=int, default=5,
                    help='time interval of inputs (default: 5')

parser.add_argument('--file_dir', type=str, default='../../data/weibo_large/',
                    help='the directory of the data file (default: ../../data/weibo_large/)')

parser.add_argument('--filename_train', type=str, default='cascades_train_20160601to0609_10_context.txt',
                    help='filename of data (default:cascades_train_20160601to0609_10_context.txt)')

parser.add_argument('--filename_val', type=str, default='cascades_val_20160601to0609_10_context.txt',
                    help='filename of data (default:cascades_val_20160601to0609_10_context.txt)')

parser.add_argument('--filename_test', type=str, default='cascades_test_20160610_10_context.txt',
                    help='filename of data (default:cascades_test_20160610_10_context.txt)')
parser.add_argument('--filename_pretrain_model', type=str, default='./save_models/pretrain_model_max_label12_time_slices1800_lr0.005_wd1e-05.pkl',
                    help='filename of pretrain model (default:./save_models/pretrain_model_max_label12_time_slices1800_lr0.005_wd1e-05.pkl)')

args = parser.parse_args()

torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")


input_channels = 1
n_classes = 1
batch_size = args.batch_size
epochs = args.epochs
max_try = args.max_try

print(args)
print("Producing data...")
X_train, Y_train = data_generator_num(filename=args.file_dir+args.filename_train,ob_time=args.ob_time, gap_time=args.interval_time)
X_val,Y_val = data_generator_num(filename=args.file_dir+args.filename_val,ob_time=args.ob_time, gap_time=args.interval_time) 
X_test, Y_test = data_generator_num(filename=args.file_dir+args.filename_test,ob_time=args.ob_time, gap_time=args.interval_time)
print("Finish Producing data!")

# Note: We use a very simple setting here (assuming all levels have the same # of channels.
channel_sizes = [args.nhid]*args.levels
kernel_size = args.ksize
dropout = args.dropout
device = torch.device("cpu")
if args.cuda:
    device = torch.device("cuda:" + str(args.cuda_index))
print("device:",device)
if args.if_fix == 0:
    model = TCN(input_channels, n_classes, channel_sizes,kernel_size=kernel_size,mlp_hid_size = args.mlp_hid_size, dropout=dropout,if_fix=False)
elif args.if_fix == 1:
    model = TCN(input_channels, n_classes, channel_sizes,kernel_size=kernel_size,mlp_hid_size = args.mlp_hid_size, dropout=dropout,if_fix=True)
else:
    print("error! if_fix can only equals to 0 or 1")

print("before load pretrain model!")
print(model.linear_a.weight)
print(model.tcn.network[11].conv2.bias)
pretrain_model = torch.load(args.filename_pretrain_model)
model.load_state_dict(pretrain_model.state_dict(), strict=False)
print("after load pretrain model!")
print(model.linear_a.weight)
print(model.tcn.network[11].conv2.bias)


if args.cuda:
    model.to(device)

lr = args.lr
optimizer = getattr(optim, args.optim)(model.parameters(), lr=lr, weight_decay=args.weight_decay)
print("model paramerters",[x.numel() for x in model.parameters()])
print("model has {} paramerters in total".format(sum(x.numel() for x in model.parameters())))

print("trainable model paramerters",[x.numel() for x in model.parameters() if x.requires_grad])
print("model has {} trainable paramerters in total".format(sum(x.numel() for x in model.parameters() if x.requires_grad)))

def get_batch(i,batch_size,X,Y):
    if i + batch_size > len(X):
        x, y = X[i:], Y[i:]
    else:
        x, y = X[i:(i + batch_size)], Y[i:(i + batch_size)]

    # print(x)
    # print(y)
    x_input = np.zeros([len(x),1,int(args.ob_time/args.interval_time)])
    for n_batch in range(len(x)):
        for t_index in x[n_batch]:
            x_input[n_batch,0,t_index] += 1


    torch_X = torch.tensor(x_input,dtype = torch.float32,requires_grad=False)
    torch_Y = torch.tensor(y,dtype=torch.float32, requires_grad=False)

    if args.cuda:
        device = torch.device("cuda:" + str(args.cuda_index))
        torch_X = torch_X.to(device)
        torch_Y = torch_Y.to(device)
    return torch_X, torch_Y

def evaluate(X,Y,model):
    model.eval()
    with torch.no_grad():
        output = model(X)
        #test_loss = F.mse_loss(output, Y)
        test_loss = model.loss(Y,output)
        output_array = output.cpu().numpy()
        true_array = Y.cpu().numpy()
        MRSE = np.mean(np.power((output_array-true_array)/true_array,2))
        mRSE = np.median(np.power((output_array-true_array)/true_array,2))
        APE = np.abs((output_array-true_array)/true_array)
        MAPE = np.mean(APE)
        Acc_1 = 0
        Acc_3 = 0
        Acc_5 = 0
        for i in range(len(APE)):
            if APE[i] <=0.1:
                Acc_1 +=1
            if APE[i] <=0.3:
                Acc_3 +=1 
            if APE[i] <=0.5:
                Acc_5 +=1
        Acc_1 = Acc_1*1.0/len(APE)
        Acc_3 = Acc_3*1.0/len(APE)
        Acc_5 = Acc_5*1.0/len(APE)
        return test_loss.item(),MRSE,mRSE,MAPE,Acc_1,Acc_3,Acc_5


#remove function
start_time = time.time()
best_model = torch.load('./save_models/model_obtime'+str(args.ob_time)+"_maxlabel"+str(args.max_label)+"_time_slices"+str(args.time_slices)+"_fictcn"+str(args.if_fix)+"_lr"+str(args.lr)+"_wd"+str(args.weight_decay)+"_"+str(args.filename_train)+'.pkl')
test_loss_list = []
MRSE_list = []
mRSE_list = []
MAPE_list = []
Acc_1_list = []
Acc_3_list = []
Acc_5_list = []
for j in range(0, len(X_test), batch_size):
    x_test, y_test = get_batch(j, batch_size, X_test, Y_test)
    loss,MRSE,mRSE,MAPE,Acc1,Acc3,Acc5 = evaluate(x_test,y_test,best_model)
    test_loss_list.append(loss)
    MRSE_list.append(MRSE)
    mRSE_list.append(mRSE)
    MAPE_list.append(MAPE)
    Acc_1_list.append(Acc1)
    Acc_3_list.append(Acc3)
    Acc_5_list.append(Acc5)


print('Best test loss: {:.6f}'.format(np.mean(test_loss_list)))
print('Best test MRSE: {:.6f}'.format(np.mean(MRSE_list)))
print('Best test mRSE: {:.6f}'.format(np.mean(mRSE_list)))
print('Best test MAPE: {:.6f}'.format(np.mean(MAPE_list)))
print('Best test ACC1: {:.6f}'.format(np.mean(Acc_1_list)))
print('Best test ACC3: {:.6f}'.format(np.mean(Acc_3_list)))
print('Best test ACC5: {:.6f}'.format(np.mean(Acc_5_list)))

