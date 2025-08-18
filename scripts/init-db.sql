-- Initialize QuantumForge database schema
-- Create tables for computational results
CREATE TABLE IF NOT EXISTS calculations (
    id SERIAL PRIMARY KEY,
    molecule_name VARCHAR(255) NOT NULL,
    basis_set VARCHAR(100) NOT NULL,
    functional VARCHAR(100) NOT NULL,
    total_energy DOUBLE PRECISION,
    calculation_time DOUBLE PRECISION,
    status VARCHAR(50) DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
-- Create table for molecular structures
CREATE TABLE IF NOT EXISTS molecules (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL UNIQUE,
    formula VARCHAR(100),
    xyz_structure TEXT,
    num_atoms INTEGER,
    charge INTEGER DEFAULT 0,
    multiplicity INTEGER DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
-- Create table for density functional models
CREATE TABLE IF NOT EXISTS ml_models (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    model_type VARCHAR(100) NOT NULL,
    architecture TEXT,
    parameters_count INTEGER,
    training_accuracy DOUBLE PRECISION,
    validation_accuracy DOUBLE PRECISION,
    model_path VARCHAR(500),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT FALSE
);
-- Create table for benchmarks
CREATE TABLE IF NOT EXISTS benchmarks (
    id SERIAL PRIMARY KEY,
    model_id INTEGER REFERENCES ml_models(id),
    molecule_id INTEGER REFERENCES molecules(id),
    reference_energy DOUBLE PRECISION,
    predicted_energy DOUBLE PRECISION,
    error_mae DOUBLE PRECISION,
    error_rmse DOUBLE PRECISION,
    calculation_time DOUBLE PRECISION,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_calculations_molecule ON calculations(molecule_name);
CREATE INDEX IF NOT EXISTS idx_calculations_status ON calculations(status);
CREATE INDEX IF NOT EXISTS idx_benchmarks_model ON benchmarks(model_id);
CREATE INDEX IF NOT EXISTS idx_benchmarks_molecule ON benchmarks(molecule_id);
-- Insert some sample data
INSERT INTO molecules (name, formula, xyz_structure, num_atoms)
VALUES (
        'Water',
        'H2O',
        'O 0.0 0.0 0.0\nH 0.0 0.757 0.587\nH 0.0 -0.757 0.587',
        3
    ),
    (
        'Methane',
        'CH4',
        'C 0.0 0.0 0.0\nH 0.629 0.629 0.629\nH -0.629 -0.629 0.629\nH -0.629 0.629 -0.629\nH 0.629 -0.629 -0.629',
        5
    ),
    (
        'Hydrogen',
        'H2',
        'H 0.0 0.0 0.0\nH 0.0 0.0 0.74',
        2
    ) ON CONFLICT (name) DO NOTHING;
