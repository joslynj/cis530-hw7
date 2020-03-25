from nltk.corpus import conll2002
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Perceptron, PassiveAggressiveClassifier, RidgeClassifier, LogisticRegression, \
    SGDClassifier, LogisticRegressionCV, RidgeClassifierCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.neural_network.multilayer_perceptron import MLPClassifier
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
        # TODO: add more features here.
        ('word.isupper()', word.isupper()),
        ('word.istitle()', word.istitle()),
        # ('word.isdigit()', word.isdigit()),
        # ('word.lower()', word.lower()),
        # ('isApostrophePresent()', (word.find("'") != -1)),
        # ('isHyphenPresent()', (word.find("-") != -1)),
        # ('POS', pos)
    ]
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

if __name__ == "__main__":
    # Load the training data
    train_sents = list(conll2002.iob_sents('esp.train'))
    dev_sents = list(conll2002.iob_sents('esp.testa'))
    test_sents = list(conll2002.iob_sents('esp.testb'))

    train_feats = []
    train_labels = []
    for sent in train_sents:
        for i in range(len(sent)):
            feats = word2features(sent,i)
            train_feats.append(feats)
            train_labels.append(sent[i][-1])

    vectorizer = DictVectorizer()
    X_train = vectorizer.fit_transform(train_feats)

    # TODO: play with other models
    # model = Perceptron(verbose=1) # Test f1: 57.37
    # model = SGDClassifier()  # Test f1: 36.48
    # model = PassiveAggressiveClassifier() # Test f1:60.89
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

    print("Now run: python conlleval.py dev_results.txt")