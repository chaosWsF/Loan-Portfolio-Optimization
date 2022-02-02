from sklearn.tree import plot_tree # Used to plot tree visual, xgboost has plot_tree method too
from sklearn.tree import export_text # Use to extract SQL/ logic


'''
Proper distance metric for https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance_multicollinear.html

is sqrt(0.5*(correlation))
'''


'''
Example plotting code for plot_tree function

Variables
DT: Decision tree classifier from sklearn
non_acct_features: names of the input features which do not use account id/ names
Y: target variable

Parameters for plot_tree
feature_names parameter: columns for input numpy array
class_names: Names for target classes if available, e.g. fraud/ non-fraud or defaulted/ non-defaulted
filled: True makes it look nicer, not functionally any different

plt.figure(figsize = (16,16))
plot_tree(DT,feature_names = non_acct_features, class_names = [str(x) for x in list(np.unique(Y))], filled = True)
plt.title('<TITLE>')
plt.savefig('Save model')
plt.show()


'''

def tree_to_sql(input_str):
    '''
    Takes input string from export_text method
    '''
    predicates = []
    rules = {}
    sql = {}
    
    # Join conjunctions
    for row in input_str.split('\n'):
        if row.count('|') < 1:
            continue
        level_str, text = row.split('---')

        new_depth = level_str.count('|')
        terminal_leaf = text.count('class:')

        # Check if terminal leaf
        if terminal_leaf == 1:
            key = text.split(' ')[-1]
            

            if key not in rules:
                rules[key] = []
                
            rules[key].append( ' AND '.join(predicates) )
        else:

            # Remove predicates for separate path    
            predicates = predicates[0:(new_depth-1)]

            # Append predicate for current rule
            predicates.append(text)
            
    
    # Join disjunctions of conjuctions
    for k,v in rules.items():
        v = [ '(' + x + ')' for x in v] # wrap conjunctions in brackets
        sql_rule = ' OR '.join(v)
        sql[k] = sql_rule
        
    
    return sql