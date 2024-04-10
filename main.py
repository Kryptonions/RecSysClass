from collections import defaultdict

class Node:
    def __init__(self):
        self.children = defaultdict
        self.flag = False

class Trie:
    def __init__(self, root):
        if root:
            self.root = root
        else:
            self.root = Node()

    def search(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.flag

    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = Node()
            node = node.children[char]
        node.flag = True