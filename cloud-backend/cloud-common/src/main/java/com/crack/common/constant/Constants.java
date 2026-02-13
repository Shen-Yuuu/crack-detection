package com.crack.common.constant;

/**
 * 系统常量
 */
public class Constants {

    /**
     * Token相关
     */
    public static final String TOKEN_PREFIX = "Bearer ";
    public static final String TOKEN_HEADER = "Authorization";
    
    /**
     * Redis Key前缀
     */
    public static final String REDIS_USER_TOKEN_PREFIX = "user:token:";
    public static final String REDIS_USER_INFO_PREFIX = "user:info:";
    public static final String REDIS_INFERENCE_RESULT_PREFIX = "inference:result:";
    public static final String REDIS_RATE_LIMIT_PREFIX = "rate:limit:";
    
    /**
     * 用户状态
     */
    public static final int USER_STATUS_DISABLED = 0;
    public static final int USER_STATUS_ENABLED = 1;
    
    /**
     * 任务状态
     */
    public static final String JOB_STATUS_PENDING = "pending";
    public static final String JOB_STATUS_RUNNING = "running";
    public static final String JOB_STATUS_COMPLETED = "completed";
    public static final String JOB_STATUS_FAILED = "failed";
    
    /**
     * 数据集状态
     */
    public static final String DATASET_STATUS_PENDING = "pending";
    public static final String DATASET_STATUS_PROCESSING = "processing";
    public static final String DATASET_STATUS_COMPLETED = "completed";
    public static final String DATASET_STATUS_FAILED = "failed";
    
    /**
     * 文件夹
     */
    public static final String FOLDER_IMAGES = "images";
    public static final String FOLDER_MASKS = "masks";
    public static final String FOLDER_OVERLAYS = "overlays";
    public static final String FOLDER_HEATMAPS = "heatmaps";
    public static final String FOLDER_REPORTS = "reports";
    public static final String FOLDER_DATASETS = "datasets";
    
    /**
     * 文件大小限制
     */
    public static final long MAX_IMAGE_SIZE = 50 * 1024 * 1024; // 50MB
    public static final long MAX_DATASET_SIZE = 1024 * 1024 * 1024; // 1GB
    
    /**
     * 允许的图像类型
     */
    public static final String[] ALLOWED_IMAGE_TYPES = {
            "image/jpeg", "image/jpg", "image/png", "image/bmp", "image/tiff"
    };
    
    /**
     * 分页默认值
     */
    public static final int DEFAULT_PAGE_NUM = 1;
    public static final int DEFAULT_PAGE_SIZE = 10;
    public static final int MAX_PAGE_SIZE = 100;

    private Constants() {
        // 私有构造函数，防止实例化
    }
}
