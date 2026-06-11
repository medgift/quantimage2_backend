-- Fix: clinical_feature_definition is missing the new NOT NULL FK column
--      `clinical_feature_file_id` (added by the clinical-features update, but the
--      Alembic migration never ran on this server, so the ALTER never happened).
--
-- Strategy: PRESERVE existing data. Create one clinical_feature_file per existing
--           (user_id, album_id) group, link every old definition to it, then
--           enforce NOT NULL + the foreign key.
--
-- Safe to run once. Take a DB backup first. Wrapped in a transaction.

START TRANSACTION;

-- 1) Add the column as NULLABLE first (existing rows have no file yet).
ALTER TABLE clinical_feature_definition
  ADD COLUMN clinical_feature_file_id INT NULL;

-- 2) Create one owning file per (user_id, album_id) group of existing definitions.
--    `name` is fixed so the (user_id, album_id, name) unique constraint yields
--    exactly one file per group.
INSERT INTO clinical_feature_file (name, album_id, user_id, created_at, updated_at)
SELECT 'Migrated clinical features', album_id, user_id, NOW(), NOW()
FROM clinical_feature_definition
GROUP BY user_id, album_id;

-- 3) Point each existing definition at the file created for its group.
UPDATE clinical_feature_definition d
JOIN clinical_feature_file f
  ON  f.user_id  = d.user_id
  AND f.album_id = d.album_id
  AND f.name     = 'Migrated clinical features'
SET d.clinical_feature_file_id = f.id;

-- 4) Enforce NOT NULL now that every row is backfilled.
ALTER TABLE clinical_feature_definition
  MODIFY COLUMN clinical_feature_file_id INT NOT NULL;

-- 5) Add the foreign key with the same ON DELETE/UPDATE CASCADE the model declares.
ALTER TABLE clinical_feature_definition
  ADD CONSTRAINT fk_clinical_feature_definition_file
  FOREIGN KEY (clinical_feature_file_id)
  REFERENCES clinical_feature_file (id)
  ON DELETE CASCADE ON UPDATE CASCADE;

COMMIT;

-- Sanity checks (run after committing):
--   SELECT COUNT(*) FROM clinical_feature_definition WHERE clinical_feature_file_id IS NULL;  -- expect 0
--   SELECT id, name, user_id, album_id FROM clinical_feature_file;
