package com.crack.common.result;

import lombok.AllArgsConstructor;
import lombok.Getter;

/**
 * 响应状态码枚举
 */
@Getter
@AllArgsConstructor
public enum ResultCode {

    // 成功
    SUCCESS(200, "操作成功"),

    // 客户端错误 4xx
    BAD_REQUEST(400, "请求参数错误"),
    UNAUTHORIZED(401, "未登录或登录已过期"),
    FORBIDDEN(403, "无权限访问"),
    NOT_FOUND(404, "资源不存在"),
    METHOD_NOT_ALLOWED(405, "请求方法不允许"),
    CONFLICT(409, "数据冲突"),
    UNPROCESSABLE_ENTITY(422, "请求参数验证失败"),
    TOO_MANY_REQUESTS(429, "请求过于频繁"),

    // 服务端错误 5xx
    ERROR(500, "服务器内部错误"),
    SERVICE_UNAVAILABLE(503, "服务暂不可用"),

    // 业务错误 1xxx
    USER_NOT_FOUND(1001, "用户不存在"),
    USER_PASSWORD_ERROR(1002, "用户名或密码错误"),
    USER_DISABLED(1003, "用户已被禁用"),
    USER_EXISTS(1004, "用户已存在"),
    TOKEN_INVALID(1005, "Token无效"),
    TOKEN_EXPIRED(1006, "Token已过期"),

    // 数据集错误 2xxx
    DATASET_NOT_FOUND(2001, "数据集不存在"),
    DATASET_FORMAT_ERROR(2002, "数据集格式错误"),
    DATASET_UPLOAD_FAILED(2003, "数据集上传失败"),
    IMAGE_NOT_FOUND(2004, "图像不存在"),

    // 推理错误 3xxx
    INFERENCE_FAILED(3001, "推理失败"),
    INFERENCE_TIMEOUT(3002, "推理超时"),
    MODEL_NOT_FOUND(3003, "模型不存在"),
    JOB_NOT_FOUND(3004, "任务不存在"),
    JOB_STILL_RUNNING(3005, "任务正在执行中"),

    // 文件错误 4xxx
    FILE_UPLOAD_FAILED(4001, "文件上传失败"),
    FILE_NOT_FOUND(4002, "文件不存在"),
    FILE_TYPE_NOT_ALLOWED(4003, "文件类型不允许"),
    FILE_SIZE_EXCEEDED(4004, "文件大小超限"),

    // 报告错误 5xxx
    REPORT_GENERATION_FAILED(5001, "报告生成失败"),
    REPORT_NOT_FOUND(5002, "报告不存在");

    /**
     * 状态码
     */
    private final Integer code;

    /**
     * 消息
     */
    private final String message;
}
