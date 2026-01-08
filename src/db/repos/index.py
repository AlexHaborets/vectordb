from collections import defaultdict
from typing import Dict, List, Optional

from sqlalchemy import except_, select
from sqlalchemy.orm.session import Session

import src.db.models as models


class IndexRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def get_graph(self, collection_id: int) -> Dict[int, List[int]]:
        edges = self.session.execute(
            select(
                models.graph_association_table.c.source_id,
                models.graph_association_table.c.neighbor_id,
            ).where(models.graph_association_table.c.collection_id == collection_id)
        ).all()

        graph = defaultdict(list)
        for source, neighbor in edges:
            graph[source].append(neighbor)
        return graph
    
    def update_graph(self, collection_id: int, subgraph: Dict[int, List[int]]):
        ids = subgraph.keys()
        if not ids:
            return
        
        self.session.execute(
            models.graph_association_table.delete().where(
                (models.graph_association_table.c.collection_id == collection_id) & 
                (models.graph_association_table.c.source_id.in_(ids)) 
            )
        )
        
        data = []
        for src, neighbors in subgraph.items():
            for neighbor in neighbors:
                data.append(
                    {
                        "source_id": src,
                        "neighbor_id": neighbor,
                        "collection_id": collection_id,
                    }
                )

        if data:
            self.session.execute(models.graph_association_table.insert(), data)

    def save_graph(
        self,
        collection_id: int,
        graph: Dict[int, List[int]],
    ) -> None:
        self.session.execute(
            models.graph_association_table.delete().where(
                models.graph_association_table.c.collection_id == collection_id
            )
        )

        if not graph:
            return

        BATCH_SIZE = 4096

        batch = []

        for src, neighbors in graph.items():
            for neighbor in neighbors:
                batch.append(
                    {
                        "source_id": src,
                        "neighbor_id": neighbor,
                        "collection_id": collection_id,
                    }
                )

                if len(batch) >= BATCH_SIZE:
                    self.session.execute(models.graph_association_table.insert(), batch)
                    batch = []

        if batch:
            self.session.execute(models.graph_association_table.insert(), batch)

    def get_unindexed_vector_ids(self, collection_id: int) -> List[int]:
        all_vectors_stmt = select(models.Vector.id).where(
            models.Vector.collection_id == collection_id,
            models.Vector.deleted == False,  # noqa: E712
        )

        indexed_vectors_stmt = select(models.graph_association_table.c.source_id).where(
            models.graph_association_table.c.collection_id == collection_id
        )

        stmt = except_(all_vectors_stmt, indexed_vectors_stmt)

        result = self.session.execute(stmt)
        return list(result.scalars().all())

    def set_index_metadata(self, collection_id: int, key: str, value: str) -> None:
        self.session.merge(
            models.IndexMetadata(collection_id=collection_id, key=key, value=value)
        )

    def get_index_metadata(self, collection_id: int, key: str) -> Optional[str]:
        metadata = self.session.get(models.IndexMetadata, (collection_id, key))
        return metadata.value if metadata else None