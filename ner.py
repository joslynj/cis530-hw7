from nltk.corpus import conll2002
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Perceptron, PassiveAggressiveClassifier, RidgeClassifier, LogisticRegression, \
    SGDClassifier, LogisticRegressionCV, RidgeClassifierCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.neural_network.multilayer_perceptron import MLPClassifier
from sklearn_crfsuite import CRF
import pickle
# import sklearn_crfsuite
# from sklearn.metrics import precision_recall_fscore_support

# Assignment 7: NER
# This is just to help you get going. Feel free to
# add to or modify any part of it.


def getfeats(word, o, pos):
    """ This takes the word in question and
    the offset with respect to the instance
    word """
    o = str(o)
    features = [
        (o + 'word', word),
        ('POS', pos),
        # ('word.isupper()', word.isupper()),
        # ('word.istitle()', word.istitle()),
        # ('word.isdigit()', word.isdigit()),
        # ('isApostrophePresent()', "'" in word),
        # ('isHyphenPresent()', '-' in word),
        # ('prefix-1', word[0]),
        # ('prefix-2', word[:2]),
        # ('prefix-3', word[:3]),
        # # ('prefix-3', word[:4]),
        # ('suffix-1', word[-1]),
        # ('suffix-2', word[-2:]),
        # ('suffix-3', word[-3:])
        # ('suffix-3', word[-4:])
    ]
    if int(o) == 0:
        feats = [
            ('word.isupper()', word.isupper()),
            ('word.istitle()', word.istitle()),
            ('word.isdigit()', word.isdigit()),
            ('isApostrophePresent()', "'" in word),
            ('isHyphenPresent()', '-' in word),
            ('prefix-1', word[0]),
            ('prefix-2', word[:2]),
            ('prefix-3', word[:3]),
            ('prefix-4', word[:4]),
            ('prefix-5', word[:5]),
            ('suffix-1', word[-1]),
            ('suffix-2', word[-2:]),
            ('suffix-3', word[-3:]),
            ('suffix-4', word[-4:]),
            ('suffix-5', word[-5:])
            ]
        features.extend(feats)
    return features
    

def word2features(sent, i):
    """ The function generates all features
    for the word at position i in the
    sentence."""
    features = []
    # the window around the token
    for o in [-3,-2,-1,0,1,2,3,4]:
        if i+o >= 0 and i+o < len(sent):
            word = sent[i+o][0]
            pos = sent[i + o][1]
            featlist = getfeats(word, o, pos)
            features.extend(featlist)
    
    return dict(features)

def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, postag, label in sent]

def sent2tokens(sent):
    return [token for token, postag, label in sent]

if __name__ == "__main__":
    # Load the training data
    train_sents = list(conll2002.iob_sents('esp.train')) # 8323
    dev_sents = list(conll2002.iob_sents('esp.testa')) # 1915
    test_sents = list(conll2002.iob_sents('esp.testb'))

    ### Use the following code when experimenting with sklearn models ###
    '''
    train_feats = []
    train_labels = []
    for sent in train_sents:
        for i in range(len(sent)):
            feats = word2features(sent,i)
            train_feats.append(feats)
            train_labels.append(sent[i][-1])

    vectorizer = DictVectorizer()
    X_train = vectorizer.fit_transform(train_feats)

    model = LinearSVC()
    model.fit(X_train, train_labels)

    test_feats = []
    test_labels = []

    # switch to test_sents for your final results
    for sent in train_sents:                              # !!!
        for i in range(len(sent)):
            feats = word2features(sent,i)
            test_feats.append(feats)
            test_labels.append(sent[i][-1])

    X_test = vectorizer.transform(test_feats)
    y_pred = model.predict(X_test)

    # Save model
    pickle.dump(model, open('model', 'wb'))
    # loaded_model = pickle.load(open('model', 'rb'))

    j = 0
    print("Writing to results.txt")
    # format is: word gold pred
    with open("train_results.txt", "w") as out:          # !!!
        for sent in train_sents:                         # !!!
            for i in range(len(sent)):
                word = sent[i][0]
                gold = sent[i][-1]
                pred = y_pred[j]
                j += 1
                out.write("{}\t{}\t{}\n".format(word,gold,pred))
        out.write("\n")
    '''

    ###### Use the following when experimenting with CRF ######
    
    X_train = [sent2features(s) for s in train_sents]
    train_labels = [sent2labels(s) for s in train_sents]
    X_dev = [sent2features(s) for s in dev_sents]
    dev_labels = [sent2labels(s) for s in dev_sents]
    X_test = [sent2features(s) for s in test_sents]
    test_labels = [sent2labels(s) for s in test_sents]

    model = CRF(algorithm='lbfgs', 
    c1=0.1, 
    c2=0.1, 
    max_iterations=100, 
    all_possible_transitions=True)
    model.fit(X_train, train_labels)
    # Save trained model in current directory
    pickle.dump(model, open('model', 'wb')) 
    # model = pickle.load(open('model', 'rb'))
    y_pred = model.predict(X_test)                  ### Change to X_test ###

    j = 0
    print("Writing to results.txt")
    # format is: word gold pred
    with open("test_results.txt", "w") as out:
        for sent in test_sents:                     ### Change to test_sents ###
            for i in range(len(sent)):
                word = sent[i][0]
                gold = sent[i][-1]
                pred = y_pred[j][i]
                out.write("{}\t{}\t{}\n".format(word,gold,pred))
            j += 1
        out.write("\n")
        
        
    ##########################################################

    print("Now run: python conlleval.py dev_results.txt")