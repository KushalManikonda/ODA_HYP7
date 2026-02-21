CREATE DATABASE oil_spill_db;

USE oil_spill_db;

CREATE TABLE users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(100) NOT NULL,
    password VARCHAR(100) NOT NULL,
    email VARCHAR(255) NOT NULL DEFAULT '',
    phone VARCHAR(20) NOT NULL DEFAULT ''
);

-- If the table already exists, run these to add the columns:
-- ALTER TABLE users ADD COLUMN email VARCHAR(255) NOT NULL DEFAULT '';
-- ALTER TABLE users ADD COLUMN phone VARCHAR(20) NOT NULL DEFAULT '';
