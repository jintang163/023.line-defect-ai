CREATE TABLE IF NOT EXISTS detection_records (
    record_id TEXT PRIMARY KEY,
    detection_id TEXT,
    sequence_id TEXT,
    product_id TEXT,
    product_name TEXT,
    product_batch TEXT,
    product_model TEXT,
    result TEXT,
    defect_types TEXT,
    defect_count INTEGER,
    inference_time_ms DOUBLE PRECISION,
    model_version TEXT,
    timestamp DOUBLE PRECISION,
    line_id TEXT,
    station_id TEXT,
    camera_id TEXT,
    original_image_path TEXT,
    annotated_image_path TEXT,
    thumbnail_path TEXT,
    defects_detail TEXT,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
) PARTITION BY RANGE (created_at);

CREATE INDEX IF NOT EXISTS idx_dr_product_id ON detection_records(product_id);
CREATE INDEX IF NOT EXISTS idx_dr_timestamp ON detection_records(timestamp);
CREATE INDEX IF NOT EXISTS idx_dr_result ON detection_records(result);
CREATE INDEX IF NOT EXISTS idx_dr_defect_types ON detection_records(defect_types);
CREATE INDEX IF NOT EXISTS idx_dr_product_model ON detection_records(product_model);
CREATE INDEX IF NOT EXISTS idx_dr_detection_id ON detection_records(detection_id);
CREATE INDEX IF NOT EXISTS idx_dr_created_at ON detection_records(created_at);

CREATE TABLE IF NOT EXISTS alert_events (
    alert_id TEXT PRIMARY KEY,
    level TEXT,
    category TEXT,
    message TEXT,
    source TEXT,
    action TEXT,
    grade TEXT,
    timestamp DOUBLE PRECISION,
    detection_id TEXT,
    defect_id TEXT,
    acknowledged BOOLEAN DEFAULT FALSE,
    acknowledged_by TEXT,
    acknowledged_at DOUBLE PRECISION,
    details JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_ae_level ON alert_events(level);
CREATE INDEX IF NOT EXISTS idx_ae_category ON alert_events(category);
CREATE INDEX IF NOT EXISTS idx_ae_grade ON alert_events(grade);
CREATE INDEX IF NOT EXISTS idx_ae_timestamp ON alert_events(timestamp);
CREATE INDEX IF NOT EXISTS idx_ae_acknowledged ON alert_events(acknowledged);
CREATE INDEX IF NOT EXISTS idx_ae_created_at ON alert_events(created_at);
