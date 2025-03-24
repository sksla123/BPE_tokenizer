from .Trie import Trie, TrieNode

import logging

logger = logging.getLogger("BPE Tokenizer")

# 어휘 집합 클래스
class Vobaulary:
    '''
    어휘 집합을 관리하는 클래스
    항상 정렬된 어휘 집합을 유지하기 위해 생성
    '''
    def __init__(self, vocab: list):
        self.vocab = []

        self.start_vocab = []
        self.hashed_vocab = []

        self.set_vocab(vocab)

    def __str__(self):
        return f"Vocabulary: {self.vocab}"

    def __len__(self):
        return len(self.vocab)

    def get_vocab(self):
        return self.vocab

    def get_hashed_vocab_start_idx(self):
        return self.hashed_vocab_start_idx
    
    def get_start_vocab(self):
        return self.start_vocab
    
    def get_hashed_vocab(self):
        return self.hashed_vocab
    
    def set_vocab(self, vocab: list):
        '''
        어휘 집합을 설정하는 함수
        어휘 집합을 정렬하고, 해쉬로 시작하는 보캡의 위치를 찾아서 시작 보캡과 해쉬 보캡을 분리

        vocab (list): 어휘 집합
        '''
        self.vocab = self._sort_vocab(list(set(vocab)))



        ## 해쉬로 시작하는 보캡의 위치를 찾기
        ## 그냥 for 문은 느릴 거 같아서 next 함수를 이용
        self.hashed_vocab_start_idx = next((idx for idx, voc in enumerate(self.vocab) if voc.startswith('##')), -1)
        ## 시작 보캡과 해쉬 보캡 분리
        self.start_vocab = self.vocab[:self.hashed_vocab_start_idx]
        self.hashed_vocab = self.vocab[self.hashed_vocab_start_idx:]
    
    def add(self, vocab: str):
        '''
        어휘 집합에 새로운 단어를 추가하는 함수
        '''
        logger.debug(f"어휘 집합에 추가된 새로운 단어: {vocab}")
        self.vocab.append(vocab)
        self.set_vocab(self.vocab)