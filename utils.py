import torch, random
import numpy as np
import pickle as pkl
import datetime

def data_get_retweets(filename,min_time):
    all_retweets = []
    file = open(filename,"r",encoding="utf8")
    for line in file:
        retweets = []
        parts = line.split("\t")
        records = parts[6].split(" ")
        for r in records:
            time = int(r.split(":")[1])
            retweets.append(time)
        retweets.sort()
        #print(retweets)
        #print(line)
        #break
        temp_idx = int(len(retweets) * 0.9)
        if retweets[temp_idx] < min_time:
            continue
        all_retweets.append(retweets)
    print("data size:",len(all_retweets)) 
    return all_retweets

def feature_generator_num(filename, ob_time, gap_time,drop_prob=0.0):
    """
    Args:
        filename: the name of the data file, obeys the format
        ob_time: the time of observations
        gap_time: the time gap of each tcn input
    """
    random.seed(0)
    data_X = []
    data_Y = []

    file = open(filename,"r",encoding="utf8")
    n_line = 0
    n_ok_line = 0
    for line in file:
        n_line +=1
        parts = line.split("\t")
        if len(parts) != 7:
            continue

        x = np.zeros(int(ob_time/gap_time))

        retweets = parts[6].split(" ")
        for r in retweets:
            t = int(r.split(":")[1])
            if t < ob_time:
                index = int(t*1.0/gap_time)
                x[index] +=1
                #x.append(index)
        
        pubtime = int(parts[3])
        date_time = datetime.datetime.fromtimestamp(pubtime)
        hour = date_time.hour

        #y = np.log(int(parts[5])-len(x)+1.0) / np.log(2.0)
        # Mean Log-transformed Square Error Label
        #y = np.log(int(parts[5])+1.0) / np.log(2.0)
        # Mean Relative Square Error Label
        y = int(parts[5])
        if int(parts[5])-np.sum(x)+1.0 <0:
            print("error!",int(parts[5])-len(x)+1.0)
        # np.log(y + 1.0) / np.log(2.0)
        #y = np.log(int(parts[5]))
        #if len(x) < 10 or hour <8 or hour >18 or len(x) >1000 :
        #    continue
        r = random.random()
        if r < drop_prob:
            continue
        data_X.append(x)
        data_Y.append([y])
        n_ok_line +=1
    print("number of lines:",n_line," number of ok lines",n_ok_line)
    print("data size:",len(data_X))

    return data_X,data_Y

def data_generator_num(filename, ob_time, gap_time,drop_prob=0.0):
    """
    Args:
        filename: the name of the data file, obeys the format
        ob_time: the time of observations
        gap_time: the time gap of each tcn input
    """
    random.seed(0)
    data_X = []
    data_Y = []

    file = open(filename,"r",encoding="utf8")
    n_line = 0
    n_ok_line = 0
    for line in file:
        n_line +=1
        parts = line.split("\t")
        if len(parts) != 7:
            continue

        # x = np.zeros([1,int(ob_time/gap_time)])
        x = []

        retweets = parts[6].split(" ")
        for r in retweets:
            t = int(r.split(":")[1])
            if t < ob_time:
                index = int(t*1.0/gap_time)
                # x[0,index] +=1
                x.append(index)
        
        #pubtime = int(parts[3])
        #date_time = datetime.datetime.fromtimestamp(pubtime)
        #hour = date_time.hour

        #y = np.log(int(parts[5])-len(x)+1.0) / np.log(2.0)
        # MLSE Label
        #y = np.log(int(parts[5])+1.0) / np.log(2.0)
        # MRSE Label
        y = int(parts[5])
        if int(parts[5])-len(x)+1.0 <0:
            print("error!",int(parts[5])-len(x)+1.0)
        # np.log(y + 1.0) / np.log(2.0)
        #y = np.log(int(parts[5]))
        #if len(x) < 10 or hour <8 or hour >18 or len(x) >1000 :
        #    continue
        r = random.random()
        if r < drop_prob:
            continue
        data_X.append(x)
        data_Y.append([y])
        n_ok_line +=1
    print("number of lines:",n_line," number of ok lines",n_ok_line)
    print("data size:",len(data_X))

    return data_X,data_Y

def data_loader(filein_dir,ob_time,gap_time,pop_threshold,
                       split="random"):
    """
    Args:
        filename: the name of the data file, obeys the format
        ob_time: the time of observations
        split: the way of split, choose from "random" and "pubtime"

    """
    print("start data load!")
    [train_X, train_Y, val_X, val_Y, test_X, test_Y] = \
        pkl.load(open(filein_dir+"dataset_obtime"+str(ob_time)
                  +"_gaptime"+str(gap_time)+"_threshol"+str(pop_threshold)
                  +"_split"+split,"rb"))

if __name__ == "__main__":
    data_generator_num(filename = "../../data/weibo/cascades_2016.6.1_10_context.txt",
                       fileout_dir = "../../data/weibo/",
                       ob_time = 3600,gap_time = 5,pop_threshold = 10,
                       split="deephawkes")
    # data_generator_num_deephawkes(filename = "../../data/weibo/cascades_2016.6.1_10_context.txt",
    #                    fileout_dir = "../../data/weibo/",
    #                    ob_time = 3600,gap_time = 5,pop_threshold = 10)
    # data_loader(filein_dir = "../../../data/weibo/",
    #                    ob_time = 3600,gap_time = 5,pop_threshold = 10,
    #                    split="random")
