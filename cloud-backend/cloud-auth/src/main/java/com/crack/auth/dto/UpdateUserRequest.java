package com.crack.auth.dto;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.Email;
import jakarta.validation.constraints.Pattern;
import lombok.Data;

/**
 * 更新用户信息请求DTO
 */
@Data
@Schema(description = "更新用户信息请求")
public class UpdateUserRequest {

    @Email(message = "邮箱格式不正确")
    @Schema(description = "邮箱", example = "john@example.com")
    private String email;

    @Pattern(regexp = "^1[3-9]\\d{9}$", message = "手机号格式不正确")
    @Schema(description = "手机号", example = "13800138000")
    private String phone;

    @Schema(description = "头像URL")
    private String avatar;
}
