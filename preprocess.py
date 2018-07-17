import json
import random
import pickle


def init_train():

    with open('origin_data/training.txt','r',encoding='utf-8') as input:
        lines = input.readlines()

    dict_nums = {}
    dict_content = {}#{index:[content]}
    print (len(lines))
    num = 0

    for line in lines:
        try:
            line = line.strip()
            items = json.loads(line)
            index = items['标签']
            content = items['内容']

            if index not in dict_nums.keys():
                dict_nums[index] = 0
                dict_nums[index] +=1
            else:
                dict_nums[index] += 1

            if index not in dict_content.keys():
                dict_content[index] = []
                dict_content[index].append(content)
            else:
                dict_content[index].append(content)

        except:
            print (line)

        num +=1
        if num %1000 == 0:
            print(num)

    print (dict_nums,len(dict_content))


    #{'自动摘要': 31034, '机器翻译': 36206, '人类作者': 48018, '机器作者': 31163}
    '''
    with open('origin_data/human','r',encoding='utf-8') as input:
        human = input.readlines()
    
    with open('origin_data/machine','r',encoding='utf-8') as input:
        machine = input.readlines()
    
    with open('origin_data/summary','r',encoding='utf-8') as input:
        summary = input.readlines()
    
    with open('origin_data/translate','r',encoding='utf-8') as input:
        translate = input.readlines()
    
    print (len(summary),len(translate),len(human),len(machine))
    '''
    dict_index = {'自动摘要': 0, '机器翻译': 1, '人类作者': 2, '机器作者': 3}

    train_total = []
    test_total = []
    for key in dict_content.keys():
        contents = dict_content[key]
        contents = list(set(contents))

        tmp_len  = len(contents)
        len_test = 5000

        contents_test = random.sample(contents,len_test)
        contents_train = list((set(contents) - set(contents_test)))
        print (len(contents))
        print (len(contents_test))
        print (len(contents_train))

        train_total += [(x, dict_index[key]) for x in contents_train]
        test_total  += [(x, dict_index[key]) for x in contents_test]

    print ('\n')
    print (len(train_total))
    print (len(test_total))

    '''for i in train_total:
        if train_total.count(i) >1:
            print (i)
    
    for i in test_total:
        if test_total.count(i) >1:
            print (i)
    '''

    train_total = list(set(train_total))
    test_total= list(set(test_total))

    print ('\n')
    print (len(train_total))
    print (len(test_total))


    xtrain = [x[0] for x in train_total]
    ytrain = [x[1] for x in train_total]
    xtest = [x[0] for x in test_total]
    ytest = [x[1] for x in test_total]

    print (len(xtrain),len(ytrain),len(xtest),len(ytest))


    with open('data/xtrain.pkl','wb') as output:
        pickle.dump(xtrain,output)

    with open('data/ytrain.pkl','wb') as output:
        pickle.dump(ytrain,output)

    with open('data/xtest.pkl','wb') as output:
        pickle.dump(xtest,output)

    with open('data/ytest.pkl','wb') as output:
        pickle.dump(ytest,output)


def init_validation():
    with open('origin_data/validation.txt','r',encoding='utf-8') as input:
        lines = input.readlines()

    x_validation = []
    id_validation = []

    for line in lines:
        try:
            line = line.strip()
            items = json.loads(line)
            id = items['id']
            content = items['内容']
            x_validation.append(content)
            id_validation.append(id)
        except:
            print (line)


    with open('data/x_validation.pkl','wb') as output:
        pickle.dump(x_validation,output)

    with open('data/id_validation.pkl','wb') as output:
        pickle.dump(id_validation,output)

init_train()
#init_validation()



