from abc import ABC, abstractclassmethod
import json
from dotenv import load_dotenv
import os
from pymongo import MongoClient
from .db import DB
import pandas as pd

load_dotenv()

class Registry(ABC):
    @abstractclassmethod
    def register():
        pass

    def log(self, data):
        self.register('log', data)

    def error(self, data):
        self.register('error', data)

    def warn(self, data):
        self.register('warn', data)


class FileRegistry(Registry):
    def __init__(self, path) -> None:
        super().__init__()
        self.path = path
        self.file = open(path, 'a')
    
    def register(self, type: str, data):
        self.file.write(json.dumps({'type': type, **data}))

class CSVRegistry(Registry):
    df: pd.DataFrame
    def __init__(self, path) -> None:
        super().__init__()
        self.path = path
        self.df = pd.DataFrame()
    
    def register(self, type: str, data):
        self.df = self.df.append({'type': type, **data}, ignore_index=True)
        self.df.to_csv(self.path)

class DBRegistry(Registry):
    def __init__(self, collection_name: str) -> None:
        super().__init__()
        self.collection_name = collection_name
        self.collection = DB.get_collection(collection_name)
            
    
    def register(self, type: str, data):
        if self.collection is not None:
            self.collection.insert_one({'type': type, **data})


class CombineRegistry(Registry):
    def __init__(self, registries) -> None:
        super().__init__()
        self.registries = registries

    def register(self, type: str, data):
        for collection in self.registries:
            collection.register(type, data)