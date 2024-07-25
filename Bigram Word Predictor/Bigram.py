import numpy as np
import random
class WordTree:
    def __init__(self, min_leaf_size=1, max_depth=5):
        self.root_node = None
        self.word_list = None
        self.min_leaf_size = min_leaf_size
        self.max_depth = max_depth
    def fit(self, word_list, verbose=False):
        self.word_list = word_list
        self.root_node = WordNode(depth=0, parent_node=None)
        self.root_node.fit(all_words=self.word_list, word_indices=np.arange(len(self.word_list)),
                           min_leaf_size=self.min_leaf_size, max_depth=self.max_depth, verbose=verbose)
    def predict(self, bigrams, max_words=5):
        return self.root_node.predict(bigrams, max_words)
class WordNode:
    def __init__(self, depth, parent_node):
        self.depth = depth
        self.parent_node = parent_node
        self.all_words = None
        self.word_indices = None
        self.child_nodes = {}
        self.is_leaf_node = True
        self.split_query = None
        self.history = []
    def get_query(self):
        return self.split_query
    def get_child_node(self, response):
        if self.is_leaf_node:
            return self
        if response not in self.child_nodes:
            response = list(self.child_nodes.keys())[0]
        return self.child_nodes[response]
    def extract_bigrams(self, word, limit=5):
        bigrams = [''.join(bg) for bg in zip(word, word[1:])]
        bigrams = sorted(set(bigrams))
        return tuple(bigrams)[:limit]
    def generate_random_bigram(self):
        return chr(ord('a') + random.randint(0, 25)) + chr(ord('a') + random.randint(0, 25))
    def handle_leaf_node(self, all_words, word_indices, history, verbose):
        self.word_indices = word_indices
    def handle_split_node(self, all_words, word_indices, history, verbose):
        split_query = self.generate_random_bigram()
        split_dict = {True: [], False: []}
        for idx in word_indices:
            bigram_list = self.extract_bigrams(all_words[idx])
            split_dict[split_query in bigram_list].append(idx)
        return split_query, split_dict
    def fit(self, all_words, word_indices, min_leaf_size, max_depth, fmt_str="    ", verbose=False):
        self.all_words = all_words
        self.word_indices = word_indices
        if len(word_indices) <= min_leaf_size or self.depth >= max_depth:
            self.is_leaf_node = True
            self.handle_leaf_node(self.all_words, self.word_indices, self.history, verbose)
        else:
            self.is_leaf_node = False
            self.split_query, split_dict = self.handle_split_node(self.all_words, self.word_indices, self.history, verbose)
            for response, split in split_dict.items():
                self.child_nodes[response] = WordNode(depth=self.depth + 1, parent_node=self)
                history_copy = self.history.copy()
                history_copy.append(self.split_query)
                self.child_nodes[response].history = history_copy
                self.child_nodes[response].fit(self.all_words, split, min_leaf_size, max_depth, fmt_str, verbose)
    def predict(self, bigrams, max_words=5):
        node = self
        valid_words = []
        def contains_all_bigrams(word, bigrams):
            word_bigrams = self.extract_bigrams(word)
            return all(bg in word_bigrams for bg in bigrams)
        while len(valid_words) < max_words and not node.is_leaf_node:
            node = node.get_child_node(any(bg in bigrams for bg in node.extract_bigrams(self.all_words[node.word_indices[0]])))
        for idx in node.word_indices:
            word = self.all_words[idx]
            if contains_all_bigrams(word, bigrams):
                valid_words.append(word)
                if len(valid_words) == max_words:
                    break
        return valid_words

def my_fit(word_list):
    tree = WordTree(min_leaf_size=1, max_depth=4)
    tree.fit(word_list)
    return tree

def my_predict(model, bigram_list):
    return model.predict(bigram_list)
