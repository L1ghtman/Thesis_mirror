from typing import List
from gptcache.manager.scalar_data.base import CacheData, CacheStorage
from datetime import datetime

class CustomCacheStorage(CacheStorage):
    def __init__(self):
        self._data = {}
        self._sessions = {}
        self._deleted = set()
        self._reports = []

    def create(self):
        self._data.clear()
        self._sessions.clear()
        self._deleted.clear()
        self._reports.clear()

    def batch_insert(self, all_data: List[CacheData]):
        for data in all_data:
            self._data[data.question_id] = data

    def get_data_by_id(self, key):
        if key not in self._deleted:
            return self._data.get(key)
        return None
    
    def mark_deleted(self, keys):
        self._deleted.update(keys)

    def clear_deleted_data(self):
        for key in self._deleted:
            self._data.pop(key, None)
        self._deleted.clear()

    def get_ids(self, deleted=True):
        if deleted:
            return list(self._data.keys())
        return [k for k in self._data.keys() if k not in self._deleted]
    
    def count(self, state = 0, is_all: bool = False):
        if is_all:
            return len(self._data)
        return len([k for k in self._data.keys() if k not in self._deleted])
    
    def add_session(self, question_id, session_id, session_question):
        if session_id not in self._sessions:
            self._sessions[session_id] = {}
        self._sessions[session_id][question_id] = session_question
    def list_sessions(self, session_id, key):
        return self._sessions.get(session_id, {}).get(key)
    
    def delete_session(self, keys):
        for key in keys:
            self._sessions.pop(key, None)

    def report_cache(
            self,
            user_question,
            cache_question,
            cache_question_id,
            cache_answer,
            similarity_value,
            cache_delta_time,
    ):
        report = {
            'timestamp': datetime.now(),
            'user_question': user_question,
            'cache_question': cache_question,
            'cache_question_id': cache_question_id,
            'cache_answer': cache_answer,
            'similarity_value': similarity_value,
            'cache_delta_time': cache_delta_time
        }
        self._reports.append(report)

    def close(self):
        self.create()