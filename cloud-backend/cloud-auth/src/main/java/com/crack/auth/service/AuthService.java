package com.crack.auth.service;

import com.crack.auth.dto.*;

/**
 * 认证服务接口
 */
public interface AuthService {

    /**
     * 用户注册
     *
     * @param request 注册请求
     * @return 用户信息
     */
    UserInfo register(RegisterRequest request);

    /**
     * 用户登录
     *
     * @param request 登录请求
     * @return 登录响应（包含Token和用户信息）
     */
    LoginResponse login(LoginRequest request);

    /**
     * 用户登出
     *
     * @param userId 用户ID
     */
    void logout(Long userId);

    /**
     * 刷新令牌
     *
     * @param request 刷新令牌请求
     * @return 登录响应（包含新Token）
     */
    LoginResponse refreshToken(RefreshTokenRequest request);

    /**
     * 获取当前用户信息
     *
     * @param userId 用户ID
     * @return 用户信息
     */
    UserInfo getUserInfo(Long userId);

    /**
     * 更新用户信息
     *
     * @param userId  用户ID
     * @param request 更新请求
     * @return 更新后的用户信息
     */
    UserInfo updateUserInfo(Long userId, UpdateUserRequest request);

    /**
     * 修改密码
     *
     * @param userId  用户ID
     * @param request 修改密码请求
     */
    void changePassword(Long userId, ChangePasswordRequest request);

    /**
     * 验证Token
     *
     * @param token JWT Token
     * @return 用户信息，验证失败返回null
     */
    UserInfo validateToken(String token);
}
