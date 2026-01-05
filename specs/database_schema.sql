-- Jo.E Evaluation Framework - PostgreSQL Database Schema
-- Version: 1.0
-- Compatible with: PostgreSQL 15+
-- Extensions required: pgvector, uuid-ossp

-- Enable extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgvector";

-- ============================================================================
-- CORE EVALUATION TABLES
-- ============================================================================

-- Main evaluations table
CREATE TABLE evaluations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL,
    prompt TEXT NOT NULL CHECK (char_length(prompt) BETWEEN 1 AND 10000),
    model_output TEXT NOT NULL CHECK (char_length(model_output) BETWEEN 1 AND 50000),
    model_identifier VARCHAR(100) NOT NULL,
    context TEXT,
    tags TEXT[],
    
    -- Status tracking
    status VARCHAR(20) NOT NULL DEFAULT 'pending' 
        CHECK (status IN ('pending', 'processing', 'completed', 'failed', 'cancelled')),
    current_phase INTEGER CHECK (current_phase BETWEEN 1 AND 5),
    progress_percentage NUMERIC(5,2) CHECK (progress_percentage BETWEEN 0 AND 100),
    
    -- Results
    joe_score NUMERIC(5,2) CHECK (joe_score BETWEEN 0 AND 100),
    joe_score_ci_lower NUMERIC(5,2),
    joe_score_ci_upper NUMERIC(5,2),
    
    -- Configuration
    threshold_severity_critical NUMERIC(3,2) DEFAULT 0.8,
    threshold_confidence_uncertain NUMERIC(3,2) DEFAULT 0.6,
    threshold_novelty NUMERIC(3,2) DEFAULT 0.7,
    
    -- Metadata
    webhook_url TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    failed_reason TEXT,
    
    -- Indexes
    CONSTRAINT fk_user FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- Indexes for evaluations
CREATE INDEX idx_evaluations_user_id ON evaluations(user_id);
CREATE INDEX idx_evaluations_status ON evaluations(status);
CREATE INDEX idx_evaluations_created_at ON evaluations(created_at DESC);
CREATE INDEX idx_evaluations_model ON evaluations(model_identifier);
CREATE INDEX idx_evaluations_joe_score ON evaluations(joe_score DESC) WHERE joe_score IS NOT NULL;

-- Per-dimension scores from LLM evaluators
CREATE TABLE evaluation_dimensions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    evaluation_id UUID NOT NULL,
    dimension VARCHAR(20) NOT NULL CHECK (dimension IN ('accuracy', 'robustness', 'fairness', 'ethics')),
    
    -- Scores
    score NUMERIC(3,2) NOT NULL CHECK (score BETWEEN 1.00 AND 5.00),
    normalized_score NUMERIC(5,2) NOT NULL CHECK (normalized_score BETWEEN 0 AND 100),
    confidence NUMERIC(3,2) NOT NULL CHECK (confidence BETWEEN 0.00 AND 1.00),
    confidence_interval_lower NUMERIC(3,2),
    confidence_interval_upper NUMERIC(3,2),
    
    -- Evaluator info
    evaluator VARCHAR(50) NOT NULL,  -- 'gpt-4o', 'claude-3-opus', 'llama-3.1-70b'
    justification TEXT,
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT fk_evaluation FOREIGN KEY (evaluation_id) REFERENCES evaluations(id) ON DELETE CASCADE,
    CONSTRAINT unique_eval_dim_evaluator UNIQUE (evaluation_id, dimension, evaluator)
);

CREATE INDEX idx_dimension_eval_id ON evaluation_dimensions(evaluation_id);
CREATE INDEX idx_dimension_dimension ON evaluation_dimensions(dimension);
CREATE INDEX idx_dimension_evaluator ON evaluation_dimensions(evaluator);

-- Phase execution tracking
CREATE TABLE evaluation_phases (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    evaluation_id UUID NOT NULL,
    phase_number INTEGER NOT NULL CHECK (phase_number BETWEEN 1 AND 5),
    phase_name VARCHAR(50) NOT NULL CHECK (phase_name IN ('llm_screening', 'agent_testing', 'human_review', 'iterative_refinement', 'deployment_monitoring')),
    
    -- Status
    status VARCHAR(20) NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'processing', 'completed', 'skipped', 'failed')),
    
    -- Timing
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    duration_seconds NUMERIC(10,2),
    
    -- Results
    result JSONB,
    error_message TEXT,
    
    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT fk_evaluation FOREIGN KEY (evaluation_id) REFERENCES evaluations(id) ON DELETE CASCADE,
    CONSTRAINT unique_eval_phase UNIQUE (evaluation_id, phase_number)
);

CREATE INDEX idx_phase_eval_id ON evaluation_phases(evaluation_id);
CREATE INDEX idx_phase_status ON evaluation_phases(status);

-- ============================================================================
-- VULNERABILITY TRACKING
-- ============================================================================

CREATE TABLE vulnerabilities (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    evaluation_id UUID NOT NULL,
    
    -- Classification
    category VARCHAR(50) NOT NULL CHECK (category IN ('jailbreak', 'bias', 'misinformation', 'harmful_content', 'privacy_violation', 'toxicity', 'unfairness', 'prompt_injection')),
    
    -- Severity
    severity NUMERIC(3,2) NOT NULL CHECK (severity BETWEEN 0.00 AND 1.00),
    severity_harm NUMERIC(3,2),
    severity_exploitation NUMERIC(3,2),
    severity_scope NUMERIC(3,2),
    severity_reversibility NUMERIC(3,2),
    
    -- Details
    description TEXT NOT NULL,
    reproduction_steps TEXT,
    remediation TEXT,
    evidence JSONB,  -- Screenshots, logs, etc.
    
    -- Detection
    detected_by VARCHAR(50) NOT NULL CHECK (detected_by IN ('llm', 'agent', 'human')),
    detector_name VARCHAR(100),  -- Specific evaluator/agent/expert
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT fk_evaluation FOREIGN KEY (evaluation_id) REFERENCES evaluations(id) ON DELETE CASCADE
);

CREATE INDEX idx_vuln_eval_id ON vulnerabilities(evaluation_id);
CREATE INDEX idx_vuln_category ON vulnerabilities(category);
CREATE INDEX idx_vuln_severity ON vulnerabilities(severity DESC);
CREATE INDEX idx_vuln_detected_by ON vulnerabilities(detected_by);

-- ============================================================================
-- PATTERN LIBRARY (for novelty detection)
-- ============================================================================

CREATE TABLE pattern_library (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- Embedding (1555-dimensional vector)
    embedding vector(1555) NOT NULL,
    
    -- Classification
    violation_type VARCHAR(50) NOT NULL,
    severity_profile JSONB NOT NULL,  -- {harm, exploitation, scope, reversibility}
    
    -- Source
    source VARCHAR(20) NOT NULL CHECK (source IN ('harmbench', 'advbench', 'validated', 'synthetic')),
    source_id VARCHAR(200),  -- Original dataset ID
    
    -- Usage tracking
    match_count INTEGER DEFAULT 0,
    last_matched_at TIMESTAMP,
    
    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Vector similarity index
    CONSTRAINT check_embedding_dim CHECK (vector_dims(embedding) = 1555)
);

CREATE INDEX idx_pattern_type ON pattern_library(violation_type);
CREATE INDEX idx_pattern_source ON pattern_library(source);
CREATE INDEX idx_pattern_match_count ON pattern_library(match_count DESC);

-- Vector similarity index (for fast nearest neighbor search)
CREATE INDEX idx_pattern_embedding_cosine ON pattern_library 
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

-- ============================================================================
-- HUMAN EXPERT REVIEW
-- ============================================================================

CREATE TABLE experts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL UNIQUE,
    
    -- Expertise
    specialization TEXT[] NOT NULL,  -- ['ai_safety', 'ethics', 'domain_expert']
    certification_level VARCHAR(20) CHECK (certification_level IN ('trainee', 'certified', 'senior')),
    
    -- Performance metrics
    reviews_completed INTEGER DEFAULT 0,
    average_review_time_seconds NUMERIC(10,2),
    agreement_rate NUMERIC(3,2),  -- Agreement with other experts
    overturn_rate NUMERIC(3,2),  -- Rate of disagreements with automated assessment
    
    -- Status
    active BOOLEAN DEFAULT true,
    last_active_at TIMESTAMP,
    
    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT fk_user FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE INDEX idx_expert_user_id ON experts(user_id);
CREATE INDEX idx_expert_active ON experts(active) WHERE active = true;

CREATE TABLE expert_reviews (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    evaluation_id UUID NOT NULL,
    expert_id UUID NOT NULL,
    
    -- Review decisions
    severity_override NUMERIC(3,2) CHECK (severity_override IS NULL OR severity_override BETWEEN 0.00 AND 1.00),
    category VARCHAR(50),
    impact_level VARCHAR(20) CHECK (impact_level IN ('low', 'medium', 'high')),
    priority VARCHAR(20) CHECK (priority IN ('immediate', 'short_term', 'long_term')),
    
    -- Rationale
    rationale TEXT NOT NULL CHECK (char_length(rationale) BETWEEN 50 AND 1000),
    
    -- Metrics
    review_duration_seconds INTEGER,
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT fk_evaluation FOREIGN KEY (evaluation_id) REFERENCES evaluations(id) ON DELETE CASCADE,
    CONSTRAINT fk_expert FOREIGN KEY (expert_id) REFERENCES experts(id) ON DELETE CASCADE,
    CONSTRAINT unique_eval_expert UNIQUE (evaluation_id, expert_id)
);

CREATE INDEX idx_review_eval_id ON expert_reviews(evaluation_id);
CREATE INDEX idx_review_expert_id ON expert_reviews(expert_id);
CREATE INDEX idx_review_priority ON expert_reviews(priority) WHERE priority = 'immediate';

-- ============================================================================
-- ESCALATION TRACKING
-- ============================================================================

CREATE TABLE escalations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    evaluation_id UUID NOT NULL,
    
    -- Trigger
    trigger VARCHAR(50) NOT NULL CHECK (trigger IN ('severity', 'confidence', 'novelty')),
    threshold_value NUMERIC(3,2) NOT NULL,
    actual_value NUMERIC(3,2) NOT NULL,
    
    -- Destination
    escalated_to VARCHAR(20) NOT NULL CHECK (escalated_to IN ('agent', 'human')),
    escalated_from_phase INTEGER CHECK (escalated_from_phase BETWEEN 1 AND 5),
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT fk_evaluation FOREIGN KEY (evaluation_id) REFERENCES evaluations(id) ON DELETE CASCADE
);

CREATE INDEX idx_escalation_eval_id ON escalations(evaluation_id);
CREATE INDEX idx_escalation_trigger ON escalations(trigger);
CREATE INDEX idx_escalation_to ON escalations(escalated_to);

-- ============================================================================
-- USER MANAGEMENT & AUTHENTICATION
-- ============================================================================

CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email VARCHAR(255) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    
    -- Profile
    full_name VARCHAR(200),
    organization VARCHAR(200),
    
    -- Role-based access control
    role VARCHAR(20) NOT NULL DEFAULT 'user' CHECK (role IN ('admin', 'reviewer', 'user', 'api')),
    
    -- Status
    active BOOLEAN DEFAULT true,
    email_verified BOOLEAN DEFAULT false,
    
    -- Usage limits
    rate_limit_per_hour INTEGER DEFAULT 100,
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login_at TIMESTAMP
);

CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_role ON users(role);
CREATE INDEX idx_users_active ON users(active) WHERE active = true;

CREATE TABLE api_keys (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL,
    
    -- Key details
    key_hash VARCHAR(255) NOT NULL UNIQUE,
    key_prefix VARCHAR(20) NOT NULL,  -- First 8 chars for display
    name VARCHAR(100),
    
    -- Permissions
    scopes TEXT[],
    
    -- Status
    active BOOLEAN DEFAULT true,
    expires_at TIMESTAMP,
    last_used_at TIMESTAMP,
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT fk_user FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE INDEX idx_api_key_hash ON api_keys(key_hash);
CREATE INDEX idx_api_key_user_id ON api_keys(user_id);
CREATE INDEX idx_api_key_active ON api_keys(active) WHERE active = true;

-- ============================================================================
-- AUDIT LOGGING
-- ============================================================================

CREATE TABLE audit_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID,
    
    -- Event details
    event_type VARCHAR(50) NOT NULL,
    event_category VARCHAR(20) NOT NULL CHECK (event_category IN ('auth', 'evaluation', 'admin', 'api')),
    description TEXT,
    
    -- Context
    ip_address INET,
    user_agent TEXT,
    resource_type VARCHAR(50),
    resource_id UUID,
    
    -- Changes
    changes JSONB,
    
    -- Timestamp
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_audit_user_id ON audit_logs(user_id);
CREATE INDEX idx_audit_event_type ON audit_logs(event_type);
CREATE INDEX idx_audit_created_at ON audit_logs(created_at DESC);
CREATE INDEX idx_audit_resource ON audit_logs(resource_type, resource_id);

-- ============================================================================
-- TRIGGERS FOR AUTOMATIC UPDATES
-- ============================================================================

-- Auto-update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_evaluations_updated_at BEFORE UPDATE ON evaluations
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_experts_updated_at BEFORE UPDATE ON experts
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Auto-calculate phase duration
CREATE OR REPLACE FUNCTION calculate_phase_duration()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.completed_at IS NOT NULL AND NEW.started_at IS NOT NULL THEN
        NEW.duration_seconds = EXTRACT(EPOCH FROM (NEW.completed_at - NEW.started_at));
    END IF;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER calculate_phase_duration_trigger BEFORE UPDATE ON evaluation_phases
    FOR EACH ROW EXECUTE FUNCTION calculate_phase_duration();

-- ============================================================================
-- VIEWS FOR COMMON QUERIES
-- ============================================================================

-- Evaluation summary view
CREATE VIEW evaluation_summary AS
SELECT 
    e.id,
    e.user_id,
    e.model_identifier,
    e.status,
    e.joe_score,
    e.created_at,
    e.completed_at,
    COUNT(DISTINCT v.id) as vulnerability_count,
    COUNT(DISTINCT er.id) as expert_review_count,
    AVG(ed.confidence) as average_confidence,
    MAX(v.severity) as max_severity
FROM evaluations e
LEFT JOIN vulnerabilities v ON e.id = v.evaluation_id
LEFT JOIN expert_reviews er ON e.id = er.evaluation_id
LEFT JOIN evaluation_dimensions ed ON e.id = ed.evaluation_id
GROUP BY e.id;

-- Expert performance view
CREATE VIEW expert_performance AS
SELECT 
    exp.id,
    exp.user_id,
    exp.specialization,
    exp.reviews_completed,
    exp.average_review_time_seconds,
    COUNT(er.id) as total_reviews,
    AVG(er.review_duration_seconds) as avg_duration,
    exp.agreement_rate,
    exp.overturn_rate
FROM experts exp
LEFT JOIN expert_reviews er ON exp.id = er.expert_id
GROUP BY exp.id;

-- ============================================================================
-- SAMPLE DATA (for development/testing)
-- ============================================================================

-- Insert admin user (password: 'admin123' - hash should be bcrypt)
INSERT INTO users (email, password_hash, full_name, role) VALUES
('admin@joe-eval.ai', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewY5yvGZVJ0Yvx3O', 'System Admin', 'admin');

-- Insert initial pattern library seeds (empty embeddings - populate from HarmBench/AdvBench)
INSERT INTO pattern_library (embedding, violation_type, severity_profile, source, source_id) VALUES
(ARRAY[0.0]::vector(1555), 'jailbreak', '{"harm": 0.8, "exploitation": 0.6, "scope": 0.7, "reversibility": 0.9}'::jsonb, 'harmbench', 'HB-001'),
(ARRAY[0.0]::vector(1555), 'bias', '{"harm": 0.7, "exploitation": 0.5, "scope": 0.8, "reversibility": 0.6}'::jsonb, 'advbench', 'AB-001');

-- Grant permissions (adjust as needed)
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO jo_e_app;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO jo_e_app;

-- ============================================================================
-- MAINTENANCE FUNCTIONS
-- ============================================================================

-- Clean up old evaluations (for GDPR compliance)
CREATE OR REPLACE FUNCTION cleanup_old_evaluations(retention_days INTEGER DEFAULT 30)
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM evaluations 
    WHERE completed_at < CURRENT_TIMESTAMP - (retention_days || ' days')::INTERVAL
    AND status IN ('completed', 'failed', 'cancelled');
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Prune unused patterns from library
CREATE OR REPLACE FUNCTION prune_pattern_library(min_matches INTEGER DEFAULT 2, evaluation_window INTEGER DEFAULT 1000)
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM pattern_library
    WHERE match_count < min_matches
    AND id NOT IN (
        SELECT DISTINCT pl.id
        FROM pattern_library pl
        JOIN evaluations e ON e.created_at > CURRENT_TIMESTAMP - INTERVAL '30 days'
        ORDER BY e.created_at DESC
        LIMIT evaluation_window
    );
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- PERFORMANCE OPTIMIZATION
-- ============================================================================

-- Partition evaluations table by date (for large datasets)
-- Uncomment and adjust if you expect millions of evaluations

-- CREATE TABLE evaluations_2026_q1 PARTITION OF evaluations
--     FOR VALUES FROM ('2026-01-01') TO ('2026-04-01');

-- Analyze tables for query optimization
ANALYZE evaluations;
ANALYZE evaluation_dimensions;
ANALYZE vulnerabilities;
ANALYZE pattern_library;

-- ============================================================================
-- END OF SCHEMA
-- ============================================================================
