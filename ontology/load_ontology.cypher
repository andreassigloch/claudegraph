// Load Ontology Schema v1.1.0 into Neo4j
// This script creates constraints and indexes for the ontology

// Create constraints for unique node types
CREATE CONSTRAINT node_uuid IF NOT EXISTS FOR (n:SYS) REQUIRE n.uuid IS UNIQUE;
CREATE CONSTRAINT node_uuid IF NOT EXISTS FOR (n:UC) REQUIRE n.uuid IS UNIQUE;
CREATE CONSTRAINT node_uuid IF NOT EXISTS FOR (n:ACTOR) REQUIRE n.uuid IS UNIQUE;
CREATE CONSTRAINT node_uuid IF NOT EXISTS FOR (n:FCHAIN) REQUIRE n.uuid IS UNIQUE;
CREATE CONSTRAINT node_uuid IF NOT EXISTS FOR (n:FUNC) REQUIRE n.uuid IS UNIQUE;
CREATE CONSTRAINT node_uuid IF NOT EXISTS FOR (n:REQ) REQUIRE n.uuid IS UNIQUE;
CREATE CONSTRAINT node_uuid IF NOT EXISTS FOR (n:TEST) REQUIRE n.uuid IS UNIQUE;
CREATE CONSTRAINT node_uuid IF NOT EXISTS FOR (n:MOD) REQUIRE n.uuid IS UNIQUE;
CREATE CONSTRAINT node_uuid IF NOT EXISTS FOR (n:SCHEMA) REQUIRE n.uuid IS UNIQUE;

// Create constraints for required properties
CREATE CONSTRAINT node_name IF NOT EXISTS FOR (n:SYS) REQUIRE n.Name IS NOT NULL;
CREATE CONSTRAINT node_name IF NOT EXISTS FOR (n:UC) REQUIRE n.Name IS NOT NULL;
CREATE CONSTRAINT node_name IF NOT EXISTS FOR (n:ACTOR) REQUIRE n.Name IS NOT NULL;
CREATE CONSTRAINT node_name IF NOT EXISTS FOR (n:FCHAIN) REQUIRE n.Name IS NOT NULL;
CREATE CONSTRAINT node_name IF NOT EXISTS FOR (n:FUNC) REQUIRE n.Name IS NOT NULL;
CREATE CONSTRAINT node_name IF NOT EXISTS FOR (n:REQ) REQUIRE n.Name IS NOT NULL;
CREATE CONSTRAINT node_name IF NOT EXISTS FOR (n:TEST) REQUIRE n.Name IS NOT NULL;
CREATE CONSTRAINT node_name IF NOT EXISTS FOR (n:MOD) REQUIRE n.Name IS NOT NULL;
CREATE CONSTRAINT node_name IF NOT EXISTS FOR (n:SCHEMA) REQUIRE n.Name IS NOT NULL;

CREATE CONSTRAINT node_descr IF NOT EXISTS FOR (n:SYS) REQUIRE n.Descr IS NOT NULL;
CREATE CONSTRAINT node_descr IF NOT EXISTS FOR (n:UC) REQUIRE n.Descr IS NOT NULL;
CREATE CONSTRAINT node_descr IF NOT EXISTS FOR (n:ACTOR) REQUIRE n.Descr IS NOT NULL;
CREATE CONSTRAINT node_descr IF NOT EXISTS FOR (n:FCHAIN) REQUIRE n.Descr IS NOT NULL;
CREATE CONSTRAINT node_descr IF NOT EXISTS FOR (n:FUNC) REQUIRE n.Descr IS NOT NULL;
CREATE CONSTRAINT node_descr IF NOT EXISTS FOR (n:REQ) REQUIRE n.Descr IS NOT NULL;
CREATE CONSTRAINT node_descr IF NOT EXISTS FOR (n:TEST) REQUIRE n.Descr IS NOT NULL;
CREATE CONSTRAINT node_descr IF NOT EXISTS FOR (n:MOD) REQUIRE n.Descr IS NOT NULL;
CREATE CONSTRAINT node_descr IF NOT EXISTS FOR (n:SCHEMA) REQUIRE n.Descr IS NOT NULL;

// Create constraint for SCHEMA Struct field
CREATE CONSTRAINT schema_struct IF NOT EXISTS FOR (n:SCHEMA) REQUIRE n.Struct IS NOT NULL;

// Create constraints for relationship UUIDs
CREATE CONSTRAINT rel_uuid IF NOT EXISTS FOR ()-[r:compose]-() REQUIRE r.uuid IS UNIQUE;
CREATE CONSTRAINT rel_uuid IF NOT EXISTS FOR ()-[r:flow]-() REQUIRE r.uuid IS UNIQUE;
CREATE CONSTRAINT rel_uuid IF NOT EXISTS FOR ()-[r:satisfy]-() REQUIRE r.uuid IS UNIQUE;
CREATE CONSTRAINT rel_uuid IF NOT EXISTS FOR ()-[r:verify]-() REQUIRE r.uuid IS UNIQUE;
CREATE CONSTRAINT rel_uuid IF NOT EXISTS FOR ()-[r:allocate]-() REQUIRE r.uuid IS UNIQUE;
CREATE CONSTRAINT rel_uuid IF NOT EXISTS FOR ()-[r:relation]-() REQUIRE r.uuid IS UNIQUE;

// Create constraints for flow relationship required properties
CREATE CONSTRAINT flow_descr IF NOT EXISTS FOR ()-[r:flow]-() REQUIRE r.FlowDescr IS NOT NULL;
CREATE CONSTRAINT flow_def IF NOT EXISTS FOR ()-[r:flow]-() REQUIRE r.FlowDef IS NOT NULL;

// Create indexes for performance
CREATE INDEX node_name_idx IF NOT EXISTS FOR (n:SYS) ON (n.Name);
CREATE INDEX node_name_idx IF NOT EXISTS FOR (n:UC) ON (n.Name);
CREATE INDEX node_name_idx IF NOT EXISTS FOR (n:ACTOR) ON (n.Name);
CREATE INDEX node_name_idx IF NOT EXISTS FOR (n:FCHAIN) ON (n.Name);
CREATE INDEX node_name_idx IF NOT EXISTS FOR (n:FUNC) ON (n.Name);
CREATE INDEX node_name_idx IF NOT EXISTS FOR (n:REQ) ON (n.Name);
CREATE INDEX node_name_idx IF NOT EXISTS FOR (n:TEST) ON (n.Name);
CREATE INDEX node_name_idx IF NOT EXISTS FOR (n:MOD) ON (n.Name);
CREATE INDEX node_name_idx IF NOT EXISTS FOR (n:SCHEMA) ON (n.Name);

// Create indexes for relationship types
CREATE INDEX rel_type_idx IF NOT EXISTS FOR ()-[r:compose]-() ON (r.type);
CREATE INDEX rel_type_idx IF NOT EXISTS FOR ()-[r:flow]-() ON (r.type);
CREATE INDEX rel_type_idx IF NOT EXISTS FOR ()-[r:satisfy]-() ON (r.type);
CREATE INDEX rel_type_idx IF NOT EXISTS FOR ()-[r:verify]-() ON (r.type);
CREATE INDEX rel_type_idx IF NOT EXISTS FOR ()-[r:allocate]-() ON (r.type);
CREATE INDEX rel_type_idx IF NOT EXISTS FOR ()-[r:relation]-() ON (r.type);

// Create text indexes for search
CREATE FULLTEXT INDEX node_description_idx IF NOT EXISTS FOR (n:SYS|UC|ACTOR|FCHAIN|FUNC|REQ|TEST|MOD|SCHEMA) ON EACH [n.Descr];
CREATE FULLTEXT INDEX flow_description_idx IF NOT EXISTS FOR ()-[r:flow]-() ON EACH [r.FlowDescr, r.FlowDef];

// Create metadata node for ontology version
MERGE (meta:METADATA {type: 'ontology'})
SET meta.version = '1.1.0',
    meta.loaded_at = datetime(),
    meta.change_comment = 'Schema Node added';

// Display current constraints and indexes
CALL db.constraints() YIELD name, description
RETURN 'Constraints' as type, name, description
UNION ALL
CALL db.indexes() YIELD name, labelsOrTypes, properties, type
RETURN 'Indexes' as type, name, labelsOrTypes + properties as description, type;