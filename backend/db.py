import sqlite3
import networkx as nx
import os

class DatabaseManager:
    """Manages the local SQLite database, NetworkX graph, and live Neo4j synchronization."""
    
    def __init__(self):
        self.db_path = "maiis_investigation.db"
        self.graph = nx.DiGraph()
        
        # Neo4j settings from environment variables
        self.neo4j_uri = os.getenv("NEO4J_URI")
        self.neo4j_user = os.getenv("NEO4J_USER", "neo4j")
        self.neo4j_password = os.getenv("NEO4J_PASSWORD")
        self.neo4j_driver = None
        
        self._init_sqlite()
        self._load_from_sqlite()
        self._init_neo4j()
        
    def _init_sqlite(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS entities (
                id TEXT PRIMARY KEY,
                name TEXT,
                type TEXT
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS relations (
                source TEXT,
                target TEXT,
                type TEXT,
                PRIMARY KEY (source, target, type)
            )
        """)
        conn.commit()
        conn.close()
        
    def _load_from_sqlite(self):
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Load entities
            cursor.execute("SELECT id, name, type FROM entities")
            for row in cursor.fetchall():
                self.graph.add_node(row[0], name=row[1], type=row[2])
                
            # Load relations
            cursor.execute("SELECT source, target, type FROM relations")
            for row in cursor.fetchall():
                self.graph.add_edge(row[0], row[1], type=row[2])
                
            conn.close()
        except Exception as e:
            print(f"Error loading graph from SQLite: {e}")
            
    def _init_neo4j(self):
        if self.neo4j_uri and self.neo4j_password:
            try:
                from neo4j import GraphDatabase
                self.neo4j_driver = GraphDatabase.driver(
                    self.neo4j_uri,
                    auth=(self.neo4j_user, self.neo4j_password)
                )
                print("Connected to live Neo4j database successfully.")
                # Auto-sync any existing data on startup
                self.sync_to_neo4j()
            except Exception as e:
                print(f"Neo4j connection failed: {e}")
                self.neo4j_driver = None
                
    def sync_to_neo4j(self):
        """Sync all current nodes and relationships from local NetworkX graph to Neo4j."""
        if not self.neo4j_driver:
            return False
            
        try:
            with self.neo4j_driver.session() as session:
                # Clear existing nodes of MAIIS to prevent duplication
                session.run("MATCH (n:Entity) DETACH DELETE n")
                
                # Add all nodes
                for node, attrs in self.graph.nodes(data=True):
                    session.run(
                        "MERGE (e:Entity {id: $id}) "
                        "SET e.name = $name, e.type = $type",
                        id=node,
                        name=attrs.get("name", node),
                        type=attrs.get("type", "entity")
                    )
                    
                # Add all edges
                for u, v, attrs in self.graph.edges(data=True):
                    rel_type = attrs.get("type", "RELATED_TO").upper().replace(" ", "_")
                    safe_rel = "".join([c if c.isalnum() else "_" for c in rel_type])
                    query = f"MATCH (a:Entity {{id: $source}}), (b:Entity {{id: $target}}) " \
                            f"MERGE (a)-[r:{safe_rel}]->(b)"
                    session.run(query, source=u, target=v)
                    
            print("Graph data successfully synchronized to live Neo4j database.")
            return True
        except Exception as e:
            print(f"Failed to sync graph to Neo4j: {e}")
            return False
            
    def add_entity(self, entity_id, name, entity_type):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("INSERT OR REPLACE INTO entities VALUES (?, ?, ?)", (entity_id, name, entity_type))
        conn.commit()
        conn.close()
        self.graph.add_node(entity_id, name=name, type=entity_type)
        
        # Real-time Neo4j sync
        if self.neo4j_driver:
            try:
                with self.neo4j_driver.session() as session:
                    session.run(
                        "MERGE (e:Entity {id: $id}) SET e.name = $name, e.type = $type",
                        id=entity_id, name=name, type=entity_type
                    )
            except Exception as e:
                print(f"Neo4j real-time node write failed: {e}")
        
    def add_relation(self, source, target, rel_type):
        # Enforce that nodes exist in the graph
        if not self.graph.has_node(source):
            self.add_entity(source, source, "Unknown")
        if not self.graph.has_node(target):
            self.add_entity(target, target, "Unknown")
            
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("INSERT OR IGNORE INTO relations VALUES (?, ?, ?)", (source, target, rel_type))
        conn.commit()
        conn.close()
        self.graph.add_edge(source, target, type=rel_type)
        
        # Real-time Neo4j sync
        if self.neo4j_driver:
            try:
                with self.neo4j_driver.session() as session:
                    safe_rel = "".join([c if c.isalnum() else "_" for c in rel_type.upper().replace(" ", "_")])
                    query = f"MATCH (a:Entity {{id: $source}}), (b:Entity {{id: $target}}) " \
                            f"MERGE (a)-[r:{safe_rel}]->(b)"
                    session.run(query, source=source, target=target)
            except Exception as e:
                print(f"Neo4j real-time relation write failed: {e}")
        
    def get_graph_data(self):
        nodes = []
        for node, attrs in self.graph.nodes(data=True):
            nodes.append({
                "id": node,
                "label": attrs.get("name", node),
                "type": attrs.get("type", "entity")
            })
        edges = []
        for u, v, attrs in self.graph.edges(data=True):
            edges.append({
                "source": u,
                "target": v,
                "label": attrs.get("type", "rel")
            })
        return {"nodes": nodes, "edges": edges}

    def clear(self):
        self.graph.clear()
        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except Exception:
                pass
        self._init_sqlite()
        
        # Clear Neo4j database
        if self.neo4j_driver:
            try:
                with self.neo4j_driver.session() as session:
                    session.run("MATCH (n:Entity) DETACH DELETE n")
                print("Neo4j database cleared successfully.")
            except Exception as e:
                print(f"Failed to clear Neo4j: {e}")

db_manager = DatabaseManager()
