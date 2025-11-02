label_dict = {}
tweet_dict = {}
with open('data/test_a_labels_all.csv', 'r') as f:
    lines = f.readlines()
    for i in range(len(lines)):
        line = lines[i].strip().split(',')
        label_dict[line[0]] = line[1]
    print(label_dict)
        

with open('data/test_a_tweets_all.tsv', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    lines_split = [line.strip().split('\t') for line in lines]
    for i in range(len(lines_split)):
        tweet_dict[lines_split[i][0]] = lines_split[i][1]
    with open('data/test_a_combined.tsv', 'w', encoding='utf-8') as out_f:
        for key in label_dict.keys():
            if key != "id":
                if label_dict[key] == "OFF":
                    line = tweet_dict[key] + '\t' + label_dict[key] + '\n'
                    out_f.write(line)

        