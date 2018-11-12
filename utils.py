# utils.py
import numpy as np
import torch

# Bijection between objects and integers starting at 0. Useful for mapping
# labels, features, etc. into coordinates of a vector space.
class Indexer(object):
    def __init__(self):
        self.objs_to_ints = {}
        self.ints_to_objs = {}

    def __repr__(self):
        return str([str(self.get_object(i)) for i in range(0, len(self))])

    def __len__(self):
        return len(self.objs_to_ints)

    def get_object(self, index):
        if (index not in self.ints_to_objs):
            return None
        else:
            return self.ints_to_objs[index]

    def contains(self, object):
        return self.index_of(object) != -1

    # Returns -1 if the object isn't present, index otherwise
    def index_of(self, object):
        if (object not in self.objs_to_ints):
            return -1
        else:
            return self.objs_to_ints[object]

    # Adds the object to the index if it isn't present, always returns a nonnegative index
    def get_index(self, object, add=True):
        if not add:
            return self.index_of(object)
        if (object not in self.objs_to_ints):
            new_idx = len(self.objs_to_ints)
            self.objs_to_ints[object] = new_idx
            self.ints_to_objs[new_idx] = object
        return self.objs_to_ints[object]

    def shift(self, amt):
        self.objs_to_ints = { k:v + amt for k, v in self.objs_to_ints.items() }
        self.ints_to_objs = { k + amt:v for k, v in self.ints_to_objs.items() }

# Map from objects to doubles that has a default value of 0 for all elements
# Relatively inefficient (dictionary-backed); shouldn't be used for anything very large-scale,
# instead use an Indexer over the objects and use a numpy array to store the values
class Counter(object):
    def __init__(self):
        self.counter = {}

    def __repr__(self):
        return str([str(key) + ": " + str(self.get_count(key)) for key in self.counter.keys()])

    def __len__(self):
        return len(self.counter)

    def keys(self):
        return self.counter.keys()

    def get_count(self, key):
        if key in self.counter:
            return self.counter[key]
        else:
            return 0

    def increment_count(self, obj, count):
        if obj in self.counter:
            self.counter[obj] = self.counter[obj] + count
        else:
            self.counter[obj] = count

    def increment_all(self, objs_list, count):
        for obj in objs_list:
            self.increment_count(obj, count)

    def set_count(self, obj, count):
        self.counter[obj] = count

    def add(self, otherCounter):
        for key in otherCounter.counter.keys():
            self.increment_count(key, otherCounter.counter[key])

    # Bad O(n) implementation right now
    def argmax(self):
        best_key = None
        for key in self.counter.keys():
            if best_key is None or self.get_count(key) > self.get_count(best_key):
                best_key = key
        return best_key


# Beam data structure. Maintains a list of scored elements like a Counter, but only keeps the top n
# elements after every insertion operation. Insertion is O(n) (list is maintained in
# sorted order), access is O(1). Still fast enough for practical purposes for small beams.
class Beam(object):
    def __init__(self, size):
        self.size = size
        self.elts = []
        self.scores = []
        self.hidden_state = []
        self.path = []

    def __repr__(self):
        return "Beam(" + repr(self.get_elts_and_scores()) + ")"

    def __len__(self):
        return len(self.elts)

    # Adds the element to the beam with the given score if the beam has room or if the score
    # is better than the score of the worst element currently on the beam
    def add(self, elt, score, hidden_state, path):
        if len(self.elts) == self.size and score < self.scores[-1]:
            # Do nothing because this element is the worst
            return
        # If the list contains the item with a lower score, remove it
        i = 0
        while i < len(self.elts):
            if self.elts[i] == elt and score > self.scores[i]:
                del self.elts[i]
                del self.scores[i]
                del self.hidden_state[i]
                del self.path[i]
            i += 1
        # If the list is empty, just insert the item
        if len(self.elts) == 0:
            self.elts.insert(0, elt)
            self.scores.insert(0, score)
            self.hidden_state.insert(0, hidden_state)
            self.path.insert(0, path)
        # Find the insertion point with binary search
        else:
            lb = 0
            ub = len(self.scores) - 1
            # We're searching for the index of the first element with score less than score
            while lb < ub:
                m = (lb + ub) // 2
                # Check > because the list is sorted in descending order
                if self.scores[m] > score:
                    # Put the lower bound ahead of m because all elements before this are greater
                    lb = m + 1
                else:
                    # m could still be the insertion point
                    ub = m
            # lb and ub should be equal and indicate the index of the first element with score less than score.
            # Might be necessary to insert at the end of the list.
            if self.scores[lb] > score:
                self.elts.insert(lb + 1, elt)
                self.scores.insert(lb + 1, score)
                self.hidden_state.insert(lb + 1, hidden_state)
                self.path.insert(lb + 1, path)
            else:
                self.elts.insert(lb, elt)
                self.scores.insert(lb, score)
                self.hidden_state.insert(lb, hidden_state)
                self.path.insert(lb, path)
            # Drop and item from the beam if necessary
            if len(self.scores) > self.size:
                self.elts.pop()
                self.scores.pop()
                self.hidden_state.pop()
                self.path.pop()

    def get_elts(self):
        return self.elts

    def get_elts_and_scores(self):
        return zip(self.elts, self.scores, self.hidden_state, self.path)

    def head(self):
        return self.elts[0]

    def all_EOS(self, EOS_token):
        elts_items = [i.item() for i in self.elts]
        all_same = (len(set(elts_items)) ==1)
        is_EOS = (elts_items[0] == EOS_token)
        return all_same and is_EOS

    def head_path(self):
        return self.path[0]


# Indexes a string feat using feature_indexer and adds it to feats.
# If add_to_indexer is true, that feature is indexed and added even if it is new
# If add_to_indexer is false, unseen features will be discarded
def maybe_add_feature(feats, feature_indexer, add_to_indexer, feat):
    if add_to_indexer:
        feats.append(feature_indexer.get_index(feat))
    else:
        feat_idx = feature_indexer.index_of(feat)
        if feat_idx != -1:
            feats.append(feat_idx)


# Computes the dot product over a list of features (i.e., a sparse feature vector)
# and a weight vector (numpy array)
def score_indexed_features(feats, weights):
    score = 0.0
    for feat in feats:
        score += weights[feat]
    return score

def make_word_wectors(word_vectors, input_indexer, input_dim):
    print("HERE")
    new_word_vecs = [0 for i in range(len(input_indexer))]
    word_vectors_index = word_vectors.word_indexer
    word_vectors = word_vectors.vectors
    count = 0
    for i in input_indexer.objs_to_ints:
        ints = input_indexer.index_of(i)
        word_vec_int = word_vectors_index.index_of(i)
        if word_vec_int != -1:
            new_word_vecs[ints] = word_vectors[word_vec_int]
            count +=1
        else:
            new_word_vecs[ints] = np.random.rand(1, input_dim)[0]
    #print(type(new_word_vecs))
    #print("shape of new_word_vecs: ", new_word_vecs.size())
    return new_word_vecs


##################
# Tests
def test_counter():
    print("TESTING COUNTER")
    ctr = Counter()
    ctr.increment_count("a", 5)
    ctr.increment_count("b", 3)
    print(str(ctr.counter))
    for key in ctr.counter.keys():
        print(key)
    ctr2 = Counter()
    ctr2.increment_count("a", 3)
    ctr2.increment_count("c", 4)
    ctr.add(ctr2)
    #print(f"{ctr} should be ['a: 8', 'c: 4', 'b: 3']")


def test_beam():
    print("TESTING BEAM")
    beam = Beam(3)
    beam.add("a", 5)
    beam.add("b", 7)
    beam.add("c", 6)
    beam.add("d", 4)
    #print(f"Should contain b, c, a: {beam}")
    beam.add("e", 8)
    beam.add("f", 6.5)
    #print(f"Should contain e, b, f: {beam}")
    beam.add("f", 9.5)
    #print(f"Should contain f, e, b: {beam}")

    beam = Beam(5)
    beam.add("a", 5)
    beam.add("b", 7)
    beam.add("c", 6)
    beam.add("d", 4)
    #print(f"Should contain b, c, a, d: {beam}")
    beam.add("e", 8)
    beam.add("f", 6.5)
    #print(f"Should contain e, b, f, c, a: {beam}")

if __name__ == '__main__':
    test_counter()
    test_beam()
