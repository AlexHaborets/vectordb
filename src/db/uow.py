from __future__ import annotations
import abc

from sqlalchemy.orm import Session, sessionmaker

from src.db.repos.collection import CollectionRepository
from src.db.repos.vector import VectorRepository


class UnitOfWork(abc.ABC):
    # Unit of work interface
    collections: CollectionRepository
    vectors: VectorRepository

    def __enter__(self) -> UnitOfWork:
        return self

    def __exit__(self, exc_type, exc_val, traceback):
        self.rollback()

    @abc.abstractmethod
    def commit(self):
        raise NotImplementedError

    @abc.abstractmethod
    def rollback(self):
        raise NotImplementedError


class DBUnitOfWork(UnitOfWork):
    def __init__(self, session_factory: sessionmaker):
        self.session_factory = session_factory

    def __enter__(self) -> UnitOfWork:
        self.session: Session = self.session_factory()
        self.collections = CollectionRepository(self.session)
        self.vectors = VectorRepository(self.session)
        return super().__enter__()

    def __exit__(self, exc_type, exc_val, traceback) -> None:
        if exc_type is None:
            self.commit()
        else:
            self.rollback()

        self.session.close()

    def commit(self) -> None:
        self.session.commit()

    def rollback(self) -> None:
        self.session.rollback()
