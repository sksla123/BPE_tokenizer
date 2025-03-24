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
        '''
            word (str): 삽입할 단어
        '''
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True

    def find_prefix_node(self, word: str):
        """
        입력 단어와 일치하는 가장 깊은 노드를 찾습니다.
        
        Args:
            word (str): 탐색할 단어
        
        Returns:
            Tuple[TrieNode, str]: 마지막으로 탐색된 노드와 현재까지의 접두사
        """
        node = self.root
        prefix = ""
        
        for char in word:
            if char in node.children:
                node = node.children[char]
                prefix += char
            else:
                break
        
        return node, prefix

    def get_longest_match(self, node: TrieNode, prefix: str):
        """
        주어진 노드에서 가장 긴 일치 문자열을 찾습니다.
        
        Args:
            node (TrieNode): 탐색 시작 노드
            prefix (str): 현재까지의 접두사
        
        Returns:
            str: 가장 긴 일치 문자열
        """
        longest_match = ""
        
        # DFS를 사용하여 가장 긴 단어 탐색
        stack = [(node, prefix)]
        
        while stack:
            current_node, current_prefix = stack.pop()
            
            if current_node.is_end_of_word and len(current_prefix) > len(longest_match):
                longest_match = current_prefix
            
            for char, child_node in current_node.children.items():
                stack.append((child_node, current_prefix + char))
        
        return longest_match

    def get_token(self, word: str):
        """
        입력된 단어와 시작 알파벳부터 일치하는 값을 모두 탐색하고,
        그 중 가장 긴 단어를 반환합니다.
        
        Args:
            word (str): 검색할 단어
        
        Returns:
            str: 입력 단어와 일치하는 가장 긴 토큰 (없으면 빈 문자열 반환)
        """
        # 1. 접두사와 일치하는 가장 깊은 노드 찾기
        node, prefix = self.find_prefix_node(word)
        
        # 2. 해당 노드에서 가장 긴 매칭 단어 찾기
        if prefix:  # 접두사가 존재하면 탐색 진행
            return self.get_longest_match(node, prefix)
        
        return ""  # 접두사 매칭이 없으면 빈 문자열 반환
