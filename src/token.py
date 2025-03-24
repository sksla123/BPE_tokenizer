import logging
from .logger import logger_name

logger = logging.getLogger(logger_name)

## 해당 Token이 subword인지 word인지 확인하기 위한 클래스
class Token:
    def __init__(self, token: str, is_sub: bool):
        self.string = token
        self.is_sub = is_sub
    
    def __str__(self):
        return self.string

    def __repr__(self):
        return ("[subword] <<" if self.is_sub else "[word] <<") + self.string + ">>"
    
    def __eq__(self, other):
        return self.string == other.string and self.is_sub == other.is_sub
    
    def __hash__(self):
        return hash((self.string, self.is_sub))