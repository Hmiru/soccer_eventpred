from collections import defaultdict
from typing import Dict, Optional, Union

UNK_TOKEN = "[UNK]"
PAD_TOKEN = "[PAD]"


class Vocabulary:
    def __init__(self,token2index: Optional[Dict[str, Dict[str, int]]] = None) -> None:
        self._token2index: Dict[str, Dict[str, int]] = defaultdict(dict)
        self._index2token: Dict[str, Dict[int, str]] = defaultdict(dict)

        if token2index is not None:
            for namespace in token2index:
                self._token2index[namespace] = {
                    key: value for key, value in token2index[namespace].items()
                }
                self._index2token[namespace] = {
                    value: key for key, value in token2index[namespace].items()
                }

    def add(self, token: str, namespace: str = "tokens") -> int:
        if not token in self._token2index[namespace]:
            index = len(self._token2index[namespace])
            self._token2index[namespace][token] = index
            self._index2token[namespace][index] = token
        return self.get(token, namespace)

    def get(
        self, token_or_index: Union[str, int], namespace: str = "tokens"
    ) -> Union[str, int]:
        if isinstance(token_or_index, str):
            if token_or_index in self._token2index[namespace]:
                return self._token2index[namespace][token_or_index]
            else:
                return self._token2index[namespace][UNK_TOKEN]
        else:
            return self._index2token[namespace][token_or_index]

    def size(self, namespace: str = "tokens") -> int:
        return len(self._token2index[namespace])

    def get_namespace_tokens(self, namespace: str = "tokens") -> list[str]:
        """특정 네임스페이스의 모든 토큰을 반환합니다."""
        return list(self._token2index[namespace].keys())