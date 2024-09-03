# models.py

from sentiment_data import *
from utils import *
import random
import math
from collections import Counter
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')


class FeatureExtractor(object):
    """
    Feature extraction base type. Takes a sentence and returns an indexed list of features.
    """
    def get_indexer(self):
        raise Exception("Don't call me, call my subclasses")

    def extract_features(self, sentence: List[str], add_to_indexer: bool=False) -> Counter:
        """
        Extract features from a sentence represented as a list of words. Includes a flag add_to_indexer to
        :param sentence: words in the example to featurize
        :param add_to_indexer: True if we should grow the dimensionality of the featurizer if new features are encountered.
        At test time, any unseen features should be discarded, but at train time, we probably want to keep growing it.
        :return: A feature vector. We suggest using a Counter[int], which can encode a sparse feature vector (only
        a few indices have nonzero value) in essentially the same way as a map. However, you can use whatever data
        structure you prefer, since this does not interact with the framework code.
        """
        raise Exception("Don't call me, call my subclasses")


class UnigramFeatureExtractor(FeatureExtractor):
    """
    Extracts unigram bag-of-words features from a sentence. It's up to you to decide how you want to handle counts
    and any additional preprocessing you want to do.
    """
    def __init__(self, indexer: Indexer):
        self.indexer = indexer
        self.stopwords = set(stopwords.words('english'))
        self.words_to_remove = {'the', 'and', 'a', 'it', 'if'}
        self.negation_words = {'not', 'never', 'no', 'none', 'nobody', 'nothing', 'neither', 'nowhere'}

    def handle_negations(self, sentence):
        negated_sentence = []
        negate = False
        for word in sentence:
            if word in self.negation_words:
                negate = True
                continue
            if negate:
                negated_sentence.append(f'not_{word}')
                negate = False
            else:
                negated_sentence.append(word)
        return negated_sentence

    def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> Counter:

        lowercase_sentence = [word.lower() for word in sentence]
        filtered_sentence = [word for word in lowercase_sentence if word not in self.stopwords]
        filtered_sentence = [word for word in filtered_sentence if word.isalpha()]
        filtered_sentence = [word for word in filtered_sentence if word not in self.words_to_remove]

        filtered_sentence = self.handle_negations(filtered_sentence)

        features = Counter()

        for word in filtered_sentence:
            if add_to_indexer:
                index = self.indexer.add_and_get_index(word)
            else:
                index = self.indexer.index_of(word)
            if index != -1:
                features[index] += 1

        return features

    def update_word_counts(self, sentences: List[List[str]]):
        pass


class BigramFeatureExtractor(FeatureExtractor):
    """
    Bigram feature extractor analogous to the unigram one.
    """

    def __init__(self, indexer: Indexer):
        self.indexer = indexer
        self.stopwords = set(stopwords.words('english'))
        self.words_to_remove = {'the', 'and', 'a', 'it', 'if'}
        self.negation_words = {'not', 'never', 'no', 'none', 'nobody', 'nothing', 'neither', 'nowhere'}

    def handle_negations(self, sentence):
        negated_sentence = []
        negate = False
        for word in sentence:
            if word in self.negation_words:
                negate = True
                continue
            if negate:
                negated_sentence.append(f'not_{word}')
                negate = False
            else:
                negated_sentence.append(word)
        return negated_sentence

    def extract_features(self, sentence: List[str], add_to_indexer: bool=False) -> Counter:
        lowercase_sentence = [word.lower() for word in sentence]
        filtered_sentence = lowercase_sentence

        # REMOVING BECAUSE BIGRAMS COVERS MOST OF THIS STUFF
        #filtered_sentence = [word for word in lowercase_sentence if word not in self.stopwords]
        #filtered_sentence = [word for word in filtered_sentence if word.isalpha()]
        #filtered_sentence = [word for word in filtered_sentence if word not in self.words_to_remove]

        # filtered_sentence = self.handle_negations(filtered_sentence)

        features = Counter()

        for i in range(len(filtered_sentence) - 1):
            bigram = (filtered_sentence[i], filtered_sentence[i + 1])
            if add_to_indexer:
                index = self.indexer.add_and_get_index(bigram)
            else:
                index = self.indexer.index_of(bigram)
            if index != -1:
                features[index] += 1

        return features

    def update_word_counts(self, sentences: List[List[str]]):
        pass


class BetterFeatureExtractor(FeatureExtractor):
    """
    Better feature extractor...try whatever you can think of!
    """
    def __init__(self, indexer: Indexer):
        self.indexer = indexer
        self.min_freq = 2
        self.stopwords = set(stopwords.words('english'))
        self.word_counts = Counter()

    def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> Counter:
        lowercase_sentence = [word.lower() for word in sentence if word.isalpha()]
        filtered_sentence = [word for word in lowercase_sentence if word not in self.stopwords]

        features = Counter()

        tf_counter = Counter(filtered_sentence)

        for word, count in tf_counter.items():
            if self.word_counts[word] >= self.min_freq:
                if add_to_indexer:
                    index = self.indexer.add_and_get_index(word)
                else:
                    index = self.indexer.index_of(word)
                if index != -1:
                    features[index] = count

        return features

    def update_word_counts(self, sentences: List[List[str]]):
        for sentence in sentences:
            lowercase_sentence = [word.lower() for word in sentence if word.isalpha()]
            filtered_sentence = [word for word in lowercase_sentence if word not in self.stopwords]
            self.word_counts.update(filtered_sentence)

class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """
    def predict(self, sentence: List[str]) -> int:
        """
        :param sentence: words (List[str]) in the sentence to classify
        :return: Either 0 for negative class or 1 for positive class
        """
        raise Exception("Don't call me, call my subclasses")


class TrivialSentimentClassifier(SentimentClassifier):
    """
    Sentiment classifier that always predicts the positive class.
    """
    def predict(self, sentence: List[str]) -> int:
        return 1


class PerceptronClassifier(SentimentClassifier):
    """
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.
    """
    def __init__(self, weights, feat_extractor: FeatureExtractor):
        self.weights = weights
        self.feat_extractor = feat_extractor

    def predict(self, sentence: List[str]) -> int:
        features = self.feat_extractor.extract_features(sentence, add_to_indexer=False)
        score = sum(self.weights[feature] * count for feature, count in features.items())
        if score > 0.5:
            return 1
        else:
            return 0


class LogisticRegressionClassifier(SentimentClassifier):
    """
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.
    """
    def __init__(self, weights, feat_extractor, beta):
        self.weights = weights
        self.feat_extractor = feat_extractor
        self.beta = beta

    def predict(self, sentence):
        features = self.feat_extractor.extract_features(sentence, add_to_indexer=False)
        score = sum(self.weights[feature] * count for feature, count in features.items()) + self.beta
        probability = 1 / (1 + math.exp(-score))

        if probability > 0.5:
            return 1
        else:
            return 0


def train_perceptron(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor) -> PerceptronClassifier:
    """
    Train a classifier with the perceptron.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained PerceptronClassifier model
    """

    random.seed(42)
    num_epochs = 10

    weights = Counter()
    feat_extractor.update_word_counts([ex.words for ex in train_exs])

    for epoch in range(num_epochs):
        random.shuffle(train_exs)

        for ex in train_exs:
            features = feat_extractor.extract_features(ex.words, add_to_indexer=True)

            score = sum(weights[feature] * count for feature, count in features.items())
            prediction = 0
            if score > 0:
                prediction = 1

            if prediction != ex.label:
                for feature, count in features.items():
                    weights[feature] += count if ex.label == 1 else -count

    return PerceptronClassifier(weights, feat_extractor)


def train_logistic_regression(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor) -> LogisticRegressionClassifier:
    """
    Train a logistic regression model.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained LogisticRegressionClassifier model
    """
    random.seed(42)
    num_epochs = 10
    learning_rate = 0.1

    weights = Counter()
    feat_extractor.update_word_counts([ex.words for ex in train_exs])
    beta = 0.0

    for epoch in range(num_epochs):
        random.shuffle(train_exs)

        for sentence in train_exs:
            features = feat_extractor.extract_features(sentence.words, add_to_indexer=True)
            score = sum(weights[feature] * count for feature, count in features.items()) + beta
            probability = 1 / (1 + math.exp(-score))

            error = sentence.label - probability

            for feature, count in features.items():
                weights[feature] += learning_rate * error * count
            beta += learning_rate * error

    return LogisticRegressionClassifier(weights, feat_extractor, beta)


def train_model(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample]) -> SentimentClassifier:
    """
    Main entry point for your modifications. Trains and returns one of several models depending on the args
    passed in from the main method. You may modify this function, but probably will not need to.
    :param args: args bundle from sentiment_classifier.py
    :param train_exs: training set, List of SentimentExample objects
    :param dev_exs: dev set, List of SentimentExample objects. You can use this for validation throughout the training
    process, but you should *not* directly train on this data.
    :return: trained SentimentClassifier model, of whichever type is specified
    """
    # Initialize feature extractor
    if args.model == "TRIVIAL":
        feat_extractor = None
    elif args.feats == "UNIGRAM":
        # Add additional preprocessing code here
        feat_extractor = UnigramFeatureExtractor(Indexer())
    elif args.feats == "BIGRAM":
        # Add additional preprocessing code here
        feat_extractor = BigramFeatureExtractor(Indexer())
    elif args.feats == "BETTER":
        # Add additional preprocessing code here
        feat_extractor = BetterFeatureExtractor(Indexer())
    else:
        raise Exception("Pass in UNIGRAM, BIGRAM, or BETTER to run the appropriate system")

    # Train the model
    if args.model == "TRIVIAL":
        model = TrivialSentimentClassifier()
    elif args.model == "PERCEPTRON":
        model = train_perceptron(train_exs, feat_extractor)
    elif args.model == "LR":
        model = train_logistic_regression(train_exs, feat_extractor)
    else:
        raise Exception("Pass in TRIVIAL, PERCEPTRON, or LR to run the appropriate system")
    return model
