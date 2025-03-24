## 사전 검색을 빠르게 하기 위한 트라이 트리 구현 ## 현재 학습에만 104시간 걸림...

import logging

logger = logging.getLogger("BPE Tokenizer")

class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True

    def get_all_words(self):
        words = []
        def dfs(node, prefix):
            if node.is_end_of_word:
                words.append(prefix)
            for char, child_node in node.children.items():
                dfs(child_node, prefix + char)
        dfs(self.root, '')
        return words
    
    def get_longest_word(self):
        words = self.get_all_words()
        return max(words, key=len)
