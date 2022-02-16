import random
import argparse
def get_item_size(data_file):
    lines = open(data_file).readlines()
    lines = lines[1:]
    user_seq = []
    item_set = set()
    for line in lines:
        user, items = line.strip().split('	', 1)
        items = items.split()
        items = [int(item) for item in items]
        user_seq.append(items)
        item_set = item_set | set(items)
    max_item = max(item_set)
    item_size = max_item + 1
    return item_size

def get_user_seqs_and_gene_sample(data_file,item_size):
    lines = open(data_file).readlines()
    lines = lines[1:]
    user_seq = []
    item_set = set()
    for line in lines:
        user, items = line.strip().split('	', 1)
        items = items.split()
        items = [int(item) for item in items]
        user_seq.append(items)
        item_set = item_set | set(items)

    sample_seq = []
    for i in range(len(lines)):
        sample_list = neg_sample(set(user_seq[i]), item_size)
        sample_seq.append(sample_list)

    return sample_seq

def neg_sample(item_set, item_size):  # 前闭后闭
    sample_list = []
    for _ in range(99):
        item = random.randint(1, item_size - 1)
        while (item in item_set) or (item in sample_list):
            item = random.randint(1, item_size - 1)
        sample_list.append(item)
    return sample_list

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', default='./', type=str)
    parser.add_argument('--data_name', default='nowplaying', type=str)
    args = parser.parse_args()
    args.data_file = args.data_dir + args.data_name +'/'+ args.data_name + '.train.inter'
    args.data_file_eval = args.data_dir + args.data_name +'/'+ args.data_name + '.valid.inter'
    args.data_file_test = args.data_dir + args.data_name +'/'+ args.data_name + '.test.inter'

    args.sample_file_eval = args.data_dir + args.data_name +'/'+ args.data_name + '_valid_sample.txt'
    args.sample_file_test = args.data_dir + args.data_name +'/'+ args.data_name + '_test_sample.txt'

    item_size = get_item_size(args.data_file)
    neg_sample_eval = get_user_seqs_and_gene_sample(args.data_file_eval,item_size)
    output = open(args.sample_file_eval,'w')
    for i in range(len(neg_sample_eval)):
        output.write(str(i))
        for k in neg_sample_eval[i]:
            output.write(' '+str(k))
        output.write('\n')
    output.close()
    neg_sample_test = get_user_seqs_and_gene_sample(args.data_file_test,item_size)
    output = open(args.sample_file_test,'w')
    for i in range(len(neg_sample_test)):
        output.write(str(i))
        for k in neg_sample_test[i]:
            output.write(' '+str(k))
        output.write('\n')
    output.close()

main()
