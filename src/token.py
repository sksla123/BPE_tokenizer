import re
import sys

from .vocab import Vocabulary

import logging
from .logger import logger_name

logger = logging.getLogger(logger_name)

## 해당 Token이 subword인지 word인지 확인하기 위한 클래스
class Token:
    def __init__(self, token: str):
        self.string = token
        self.is_sub = True
    
    def __str__(self):
        return ("[subword] <<" if self.is_sub else "[word] <<") + self.string + ">>"

def pre_tokenize(corpus: str, method: str = "whitespace"):
    '''
    corpus (str): pre-tokenize 대상 코퍼스
    method (str): pre-tokenize 방법 [whitespace] # 현재는 whitespace만 지원(추후 개발)

    return (list): pre-tokenize 결과(tokenized-instances)
    ''' 

    if method == "whitespace":
        ret = re.split(r'\s+', corpus) ## whitespace 기준으로 분리
        
        # 비어있는 문자 제거
        if "" in ret:
            ret.remove("")
        
        return ret
    else:
        raise ValueError(f"지원하지 않는 pre-tokenize 방법입니다. {method}\n 지원하는 메소드 목록: [whitespace]")

## 토큰화 함수 정의
def tokenize(word: str, vocab: Vocabulary, mode: str = "infer"):
    '''
    word (str): 토큰화할 단어
    vocab (Vocabulary): 어휘 집합
    mode (str): 토큰화 모드 [train, infer]
    
    return (list): 토큰화된 단어
    '''
    logger.debug(f"[{mode} 모드] {word} 토큰화")
    
    tokens = []
    
    ## 학습 모드
    if mode == "train":
        token_string = vocab.get_token(word, "word")

        if token_string != "":
            tokens.append(Token(token_string, False))
        else:
            logger.error(f"토큰화 실패(word vocab에 없는 단어), word: {word}, token: {token_string}")
            raise ValueError(f"토큰화 실패, word: {word}, token: {token_string}\n 토큰화 실패 원인: {word}, \n word vocab: {vocab.word_vocab}")
        
        _word = word[len(token_string):]

        while _word != "":
            token_string = vocab.get_token(_word, "subword")

            if token_string != "":
                tokens.append(Token(token_string, True))
                _word = _word[len(token_string):]
            else:
                logger.error(f"토큰화 실패(subword vocab에 없는 단어), word: {word}, token: {token_string}\n 토큰화 실패 원인: {_word}, \n subword vocab: {vocab.subword_vocab}")
                raise ValueError(f"토큰화 실패, word: {word}, token: {token_string}")
            
            _word = _word[len(token_string):]
    ## 추론 모드
    elif mode == "infer":
        token_string = vocab.get_token(word, "word")

        if token_string != "":
            tokens.append(Token(token_string, False))
        else:
            ## BPE인데 단어 토큰화 실패?? 말도 안되거든요.
            tokens.append(Token("[UNK]", False))
            logger.error(f"토큰화 실패, word: {word}, token: {token_string}")
            return tokens
        
        _word = word[len(token_string):]

        while _word != "":
            token_string = vocab.get_token(_word, "subword")

            if token_string != "":
                tokens.append(Token(token_string, True))
                _word = _word[len(token_string):]
            else:
                token_string = vocab.get_token(_word, "word")

                if token_string != "":
                    tokens.append(Token(token_string, False))
                    _word = _word[len(token_string):]
                else:
                    ## BPE인데 단어 토큰화 실패?? 진짜 말도 안되는 사건이거든요.
                    tokens.append(Token("[UNK]", False))
                    logger.error(f"토큰화 실패, word: {word}, token: {token_string}")
                    return tokens
                    
            _word = _word[len(token_string):]
    
    logger.debug(f"[{mode} 모드] {word} 토큰화 완료\n토큰 목록: {tokens}")

    return tokens