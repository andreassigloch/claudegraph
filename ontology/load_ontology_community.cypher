// Load Ontology Schema v1.1.0 into Neo4j Community Edition
// This script creates constraints and indexes for the ontology

// Create constraints for unique node UUIDs
CREATE CONSTRAINT node_uuid_sys IF NOT EXISTS FOR (n:SYS) REQUIRE n.uuid IS UNIQUE;
CREATE CONSTRAINT node_uuid_uc IF NOT EXISTS FOR (n:UC) REQUIRE n.uuid IS UNIQUE;
CREATE CONSTRAINT node_uuid_actor IF NOT EXISTS FOR (n:ACTOR) REQUIRE n.uuid IS UNIQUE;
CREATE CONSTRAINT node_uuid_fchain IF NOT EXISTS FOR (n:FCHAIN) REQUIRE n.uuid IS UNIQUE;
CREATE CONSTRAINT node_uuid_func IF NOT EXISTS FOR (n:FUNC) REQUIRE n.uuid IS UNIQUE;
CREATE CONSTRAINT node_uuid_req IF NOT EXISTS FOR (n:REQ) REQUIRE n.uuid IS UNIQUE;
CREATE CONSTRAINT node_uuid_test IF NOT EXISTS FOR (n:TEST) REQUIRE n.uuid IS UNIQUE;
CREATE CONSTRAINT node_uuid_mod IF NOT EXISTS FOR (n:MOD) REQUIRE n.uuid IS UNIQUE;
CREATE CONSTRAINT node_uuid_schema IF NOT EXISTS FOR (n:SCHEMA) REQUIRE n.uuid IS UNIQUE;

// Create constraints for relationship UUIDs
CREATE CONSTRAINT rel_uuid_compose IF NOT EXISTS FOR ()-[r:compose]-() REQUIRE r.uuid IS UNIQUE;
CREATE CONSTRAINT rel_uuid_flow IF NOT EXISTS FOR ()-[r:flow]-() REQUIRE r.uuid IS UNIQUE;
CREATE CONSTRAINT rel_uuid_satisfy IF NOT EXISTS FOR ()-[r:satisfy]-() REQUIRE r.uuid IS UNIQUE;
CREATE CONSTRAINT rel_uuid_verify IF NOT EXISTS FOR ()-[r:verify]-() REQUIRE r.uuid IS UNIQUE;
CREATE CONSTRAINT rel_uuid_allocate IF NOT EXISTS FOR ()-[r:allocate]-() REQUIRE r.uuid IS UNIQUE;
CREATE CONSTRAINT rel_uuid_relation IF NOT EXISTS FOR ()-[r:relation]-() REQUIRE r.uuid IS UNIQUE;

// Create indexes for performance
CREATE INDEX node_name_sys IF NOT EXISTS FOR (n:SYS) ON (n.Name);
CREATE INDEX node_name_uc IF NOT EXISTS FOR (n:UC) ON (n.Name);
CREATE INDEX node_name_actor IF NOT EXISTS FOR (n:ACTOR) ON (n.Name);
CREATE INDEX node_name_fchain IF NOT EXISTS FOR (n:FCHAIN) ON (n.Name);
CREATE INDEX node_name_func IF NOT EXISTS FOR (n:FUNC) ON (n.Name);
CREATE INDEX node_name_req IF NOT EXISTS FOR (n:REQ) ON (n.Name);
CREATE INDEX node_name_test IF NOT EXISTS FOR (n:TEST) ON (n.Name);
CREATE INDEX node_name_mod IF NOT EXISTS FOR (n:MOD) ON (n.Name);
CREATE INDEX node_name_schema IF NOT EXISTS FOR (n:SCHEMA) ON (n.Name);

// Create text indexes for search
CREATE FULLTEXT INDEX node_description_idx IF NOT EXISTS FOR (n:SYS|UC|ACTOR|FCHAIN|FUNC|REQ|TEST|MOD|SCHEMA) ON EACH [n.Descr];
CREATE FULLTEXT INDEX flow_description_idx IF NOT EXISTS FOR ()-[r:flow]-() ON EACH [r.FlowDescr, r.FlowDef];

// Create metadata node for ontology version
MERGE (meta:METADATA {type: 'ontology'})
SET meta.version = '1.1.0',
    meta.loaded_at = datetime(),
    meta.change_comment = 'Schema Node added - Community Edition',
    meta.edition = 'Community';

// Display current constraints and indexes
CALL db.constraints() YIELD name, description
RETURN 'Constraints' as item_type, name, description
UNION ALL
CALL db.indexes() YIELD name, labelsOrTypes, properties
RETURN 'Indexes' as item_type, name, labelsOrTypes + properties as description;