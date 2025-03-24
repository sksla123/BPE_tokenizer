## 사전 검색을 빠르게 하기 위한 트라이 트리 구현 ## 현재 학습에만 104시간 걸림...

import logging

logger = logging.getLogger("BPE Tokenizer")

class TrieNode:
    def __init__(self, char: str):
        '''
        char (str): 현재 노드의 문자

        만약 본인이 root라면 해당 문자열은 ""이다.
        '''
        self.char = char
        self.children = {}
        self.is_end_of_word = False

class Trie:
    def __init__(self):
        """
        Trie 클래스는 트라이 자료구조를 관리합니다.
        """
        self.root = TrieNode("")  # 루트 노드는 빈 문자열로 초기화

    def insert(self, word: str):
        """
        단어를 트라이에 삽입합니다.
        
        word (str): 삽입할 단어
        """
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode(char)  # 현재 문자를 가진 새 노드 생성
            node = node.children[char]
        node.is_end_of_word = True  # 단어가 끝나는 위치 표시

    def get_token(self, word: str) -> str:
        """
        입력 단어에서 가장 긴 매칭 토큰을 찾습니다.
        
        word (str): 탐색할 단어
        return (str): 가장 긴 일치 문자열
        """
        node = self.root
        longest_match = ""
        current_prefix = ""
        
        for char in word:
            if char not in node.children:
                break
            node = node.children[char]
            current_prefix += char
            if node.is_end_of_word:  # 매 단계에서 최장 매칭 갱신
                longest_match = current_prefix
        
        return longest_match
