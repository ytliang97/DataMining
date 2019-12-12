import argparse
from math import log

from utility import load_data

parser = argparse.ArgumentParser()
parser.add_argument('--xlsx', help='need a xlsx file. \
                    e.g. --xlsx example.xlsx', dest='XLSX')
args = parser.parse_args()

def impurity(groups, classes, mode='gini'):
    n_instance = float(sum([len(group) for group in groups]))

    gini = 0
    parent = [0] * len(classes)
    entropy = 0
    gainRatio = 0
    splitInfo = 0
    
    for group in groups:
        size = float(len(group))
        if size == 0: continue
        score = 0
        score_e = 0
        for class_val in classes:
            p = [row[-1] for row in group].count(class_val) / size
            
            if mode == 'gini':
                score += p*p
            elif mode == 'entropy':
                if p != 0:
                    score_e -= p*log(p,2)
            elif mode == 'gain_ratio':
                parent[int(class_val)-1] += [row[-1] for row in group].count(class_val)

        if mode == 'gini':
            gini += (size/n_instance) * (1 - score)
        elif mode == 'entropy':
            entropy += (size/n_instance) * score_e
        elif mode == 'gain_ratio':
            p = size/n_instance
            if p != 0:
                splitInfo -= p*log(p,2)
    if mode == 'gini':
        return gini
    if mode == 'entropy':
        return entropy
    if mode == 'gain_ratio':
        entropy_p = 0
        for pt in parent:
            p = pt/n_instance
            if p != 0:
                entropy_p -= p*log(p,2)
        info_gain = entropy_p - entropy
    
        if splitInfo != 0:
            gainRatio = info_gain/splitInfo
        else:
            gainRatio = info_gain
        
        return gainRatio


def _continuous_attribute_split_position(featList):
    """
    input 7 values, output 8 values
     1 2 3 4 5 6 7
    ^ ^ ^ ^ ^ ^ ^ ^
    """
    featList.sort()

    s = []
    s.append(featList[0] - (featList[0]+featList[1])/2)
    for i in range(0, len(featList)-1):
        s.append((featList[i]+featList[i+1])/2)
    s.append(featList[-1] + (featList[0]+featList[1])/2)
    return s

def test_split(dataset, index, value):     
    left = []
    right = []                                     
    for row in dataset:
        if row[index] < value:
            left.append(row)
        elif row[index] >= value:
            right.append(row)
    return left, right

def get_best_feature_to_split(dataset, mode):
    class_values = list(set(row[-1] for row in dataset))

    b_f_index, b_value, b_score, groups = None, None, None, None
    if mode == 'gini' or mode == 'entropy': b_score = 10000
    elif mode == 'gain_ratio': b_score = -1

    for f_index in range(len(dataset[0])-1):
        
        column_value = [example[f_index] for example in dataset]
        split_values = _continuous_attribute_split_position(column_value)

        
        for value in split_values:
            groups = test_split(dataset, f_index, value)
            score = impurity(groups, class_values, mode)
            if mode == 'gini' or mode == 'entropy':
                if score < b_score:
                    b_f_index, b_value, b_score, b_groups = f_index, value, score, groups
            if mode == 'gain_ratio':
                if score > b_score:
                    b_f_index, b_value, b_score, b_groups = f_index, value, score, groups
            #print('feature %d < %f score = %.3f / best: feature %d < %f score = %.3f' % \
            #    (f_index, value, score, b_f_index, b_value, b_score))
    print('best: feature %d < %f, score = %f, {left: %d, right: %d}' % \
                (b_f_index, b_value, b_score, len(b_groups[0]), len(b_groups[1])))
    return {'feature_index':b_f_index, 'split_value':format(b_value, '.8f'), 'groups':b_groups}

def set_class_value_to_terminal(group, class_values):
    outcomes = [row[-1] for row in group]
    return {'class': max(set(outcomes), key=outcomes.count),
            'c1': outcomes.count(class_values[0]),
            'c2': outcomes.count(class_values[1])}

def per_class_number(group, class_values):
    outcomes = [row[-1] for row in group]
    return {'c1': outcomes.count(class_values[0]),
            'c2': outcomes.count(class_values[1])}

def split(node, class_values, max_depth=3, min_size=1, depth=1, mode='gini'):
    
    left, right = node['groups']
    del(node['groups'])
    node['data分佈'] = per_class_number(left+right, class_values)
    if not left:
        node['right'] = set_class_value_to_terminal(right, class_values)
        return
    elif not right:
        node['left'] = set_class_value_to_terminal(left, class_values)
        return

    if depth >= max_depth:
        node['left'], node['right'] = set_class_value_to_terminal(left, class_values), set_class_value_to_terminal(right, class_values)
        return
    if len(left) <= min_size:
        node['left'] = set_class_value_to_terminal(left, class_values)
    else:
        node['left'] = get_best_feature_to_split(left, mode)
        split(node['left'], class_values, max_depth, min_size, depth+1, mode)

    if len(right) <= min_size:
        node['right'] = set_class_value_to_terminal(right, class_values)
    else:
        node['right'] = get_best_feature_to_split(right, mode)
        split(node['right'], class_values, max_depth, min_size, depth+1, mode) 

def main():
    dataset, labels  = load_data(args.XLSX)

    class_values = list(set(row[-1] for row in dataset))
    
    root = get_best_feature_to_split(dataset, 'gain_ratio')

    split(root, class_values, max_depth=3, min_size=5, depth=1, mode='gain_ratio')

    print(root)


if __name__ == '__main__':
    main()
