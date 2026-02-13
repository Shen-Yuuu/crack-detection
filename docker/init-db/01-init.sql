-- ============================================
-- 道路裂纹检测系统数据库初始化脚本
-- MySQL 8.0+
-- ============================================

-- 创建数据库（如果不存在）
CREATE DATABASE IF NOT EXISTS crack_detection 
DEFAULT CHARACTER SET utf8mb4 
DEFAULT COLLATE utf8mb4_unicode_ci;

USE crack_detection;

-- 用户表
CREATE TABLE IF NOT EXISTS users (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(50) NOT NULL UNIQUE,
    password VARCHAR(255) NOT NULL,
    email VARCHAR(100) UNIQUE,
    phone VARCHAR(20),
    nickname VARCHAR(50),
    avatar VARCHAR(500),
    role VARCHAR(20) DEFAULT 'user',
    status INT DEFAULT 1,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    deleted INT DEFAULT 0,
    INDEX idx_users_username (username),
    INDEX idx_users_email (email),
    INDEX idx_users_status (status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 数据集表
CREATE TABLE IF NOT EXISTS datasets (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    user_id BIGINT NOT NULL,
    name VARCHAR(100) NOT NULL,
    description TEXT,
    source VARCHAR(50),
    total_images INT DEFAULT 0,
    train_count INT DEFAULT 0,
    val_count INT DEFAULT 0,
    test_count INT DEFAULT 0,
    status VARCHAR(20) DEFAULT 'active',
    storage_path VARCHAR(500),
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    deleted INT DEFAULT 0,
    INDEX idx_datasets_user_id (user_id),
    INDEX idx_datasets_name (name),
    INDEX idx_datasets_status (status),
    CONSTRAINT fk_datasets_user FOREIGN KEY (user_id) REFERENCES users(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 图像表
CREATE TABLE IF NOT EXISTS images (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    dataset_id BIGINT NOT NULL,
    filename VARCHAR(255) NOT NULL,
    original_filename VARCHAR(255),
    file_path VARCHAR(500) NOT NULL,
    mask_path VARCHAR(500),
    file_size BIGINT,
    width INT,
    height INT,
    format VARCHAR(20),
    split_type VARCHAR(20) DEFAULT 'train',
    has_crack TINYINT(1),
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    deleted INT DEFAULT 0,
    INDEX idx_images_dataset_id (dataset_id),
    INDEX idx_images_split_type (split_type),
    INDEX idx_images_filename (filename),
    CONSTRAINT fk_images_dataset FOREIGN KEY (dataset_id) REFERENCES datasets(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 检测任务表
CREATE TABLE IF NOT EXISTS detection_jobs (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    user_id BIGINT NOT NULL,
    image_url VARCHAR(500),
    image_name VARCHAR(255),
    dataset_id BIGINT,
    job_type VARCHAR(20) DEFAULT 'single',
    status VARCHAR(20) DEFAULT 'pending',
    total_images INT DEFAULT 1,
    processed_images INT DEFAULT 0,
    progress DECIMAL(5,2) DEFAULT 0,
    model_version VARCHAR(50),
    use_tta TINYINT(1) DEFAULT 1,
    config TEXT,
    result_id BIGINT,
    error_message TEXT,
    started_at DATETIME,
    completed_at DATETIME,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    deleted INT DEFAULT 0,
    INDEX idx_jobs_user_id (user_id),
    INDEX idx_jobs_status (status),
    INDEX idx_jobs_created_at (created_at),
    CONSTRAINT fk_jobs_user FOREIGN KEY (user_id) REFERENCES users(id),
    CONSTRAINT fk_jobs_dataset FOREIGN KEY (dataset_id) REFERENCES datasets(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 检测结果表
CREATE TABLE IF NOT EXISTS detection_results (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    job_id BIGINT NOT NULL,
    image_id BIGINT,
    mask_url VARCHAR(500),
    overlay_url VARCHAR(500),
    heatmap_url VARCHAR(500),
    original_image_url VARCHAR(500),
    mask_image_url VARCHAR(500),
    overlay_image_url VARCHAR(500),
    heatmap_image_url VARCHAR(500),
    vectors TEXT,
    attributes TEXT,
    confidence DECIMAL(5,4),
    crack_count INT,
    total_area DECIMAL(10,6),
    max_length DECIMAL(10,2),
    max_width DECIMAL(10,2),
    severity_level VARCHAR(20),
    bbox_json TEXT,
    processing_time DECIMAL(10,3),
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    deleted INT DEFAULT 0,
    INDEX idx_results_job_id (job_id),
    INDEX idx_results_image_id (image_id),
    INDEX idx_results_confidence (confidence),
    CONSTRAINT fk_results_job FOREIGN KEY (job_id) REFERENCES detection_jobs(id),
    CONSTRAINT fk_results_image FOREIGN KEY (image_id) REFERENCES images(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 报告表
CREATE TABLE IF NOT EXISTS reports (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    user_id BIGINT NOT NULL,
    title VARCHAR(200) NOT NULL,
    report_type VARCHAR(20) DEFAULT 'pdf',
    result_ids TEXT,
    file_url VARCHAR(500),
    status VARCHAR(20) DEFAULT 'pending',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    deleted INT DEFAULT 0,
    INDEX idx_reports_user_id (user_id),
    INDEX idx_reports_status (status),
    CONSTRAINT fk_reports_user FOREIGN KEY (user_id) REFERENCES users(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 操作日志表
CREATE TABLE IF NOT EXISTS operation_logs (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    user_id BIGINT,
    operation VARCHAR(50) NOT NULL,
    module VARCHAR(50),
    description TEXT,
    request_method VARCHAR(10),
    request_url VARCHAR(500),
    request_params TEXT,
    response_code INT,
    ip_address VARCHAR(50),
    user_agent VARCHAR(500),
    duration BIGINT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_logs_user_id (user_id),
    INDEX idx_logs_operation (operation),
    INDEX idx_logs_created_at (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 系统配置表
CREATE TABLE IF NOT EXISTS system_config (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    config_key VARCHAR(100) NOT NULL UNIQUE,
    config_value TEXT,
    description VARCHAR(500),
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 插入默认管理员用户 (密码: admin123，使用BCrypt加密)
INSERT INTO users (username, password, email, role, nickname) 
VALUES ('admin', '$2a$10$QiJp4ZUnu.oWnUDe3fLVS.Cto7.Y89CEGF.SiymGjvniBnoep6Wb.', 'admin@crack.com', 'admin', '系统管理员')
ON DUPLICATE KEY UPDATE username = username;

-- 插入默认配置
INSERT INTO system_config (config_key, config_value, description) VALUES
('model.version', '1.0.0', '当前模型版本'),
('model.threshold', '0.5', '默认检测阈值'),
('upload.max_size', '52428800', '最大上传文件大小（字节）'),
('upload.allowed_types', 'jpg,jpeg,png,bmp,tiff', '允许上传的文件类型')
ON DUPLICATE KEY UPDATE config_key = config_key;

ALTER TABLE detection_jobs
  ADD COLUMN image_url VARCHAR(500) AFTER user_id,
  ADD COLUMN image_name VARCHAR(255) AFTER image_url;

-- 完成提示
SELECT '数据库初始化完成!' AS message;
