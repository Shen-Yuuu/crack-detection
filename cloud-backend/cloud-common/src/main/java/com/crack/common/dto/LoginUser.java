package com.crack.common.dto;

import lombok.Data;

/**
 * 登录用户信息（用于请求上下文）
 */
@Data
public class LoginUser {

    /**
     * 用户ID
     */
    private Long userId;

    /**
     * 用户名
     */
    private String username;

    /**
     * Token
     */
    private String token;
}
