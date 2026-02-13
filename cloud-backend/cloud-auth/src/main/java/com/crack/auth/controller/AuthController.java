package com.crack.auth.controller;

import com.crack.auth.dto.*;
import com.crack.auth.service.AuthService;
import com.crack.common.constant.Constants;
import com.crack.common.result.Result;
import com.crack.common.utils.UserContext;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;

/**
 * 认证控制器
 */
@RestController
@RequestMapping("/api/v1/auth")
@RequiredArgsConstructor
@Tag(name = "认证管理", description = "用户认证相关接口")
public class AuthController {

    private final AuthService authService;

    @PostMapping("/register")
    @Operation(summary = "用户注册", description = "注册新用户")
    public Result<UserInfo> register(@Valid @RequestBody RegisterRequest request) {
        UserInfo userInfo = authService.register(request);
        return Result.success("注册成功", userInfo);
    }

    @PostMapping("/login")
    @Operation(summary = "用户登录", description = "用户登录获取Token")
    public Result<LoginResponse> login(@Valid @RequestBody LoginRequest request) {
        LoginResponse response = authService.login(request);
        return Result.success("登录成功", response);
    }

    @PostMapping("/logout")
    @Operation(summary = "用户登出", description = "退出登录")
    public Result<Void> logout(HttpServletRequest request) {
        Long userId = getUserIdFromRequest(request);
        if (userId != null) {
            authService.logout(userId);
        }
        return Result.success("登出成功", null);
    }

    @PostMapping("/refresh")
    @Operation(summary = "刷新令牌", description = "使用刷新令牌获取新的访问令牌")
    public Result<LoginResponse> refreshToken(@Valid @RequestBody RefreshTokenRequest request) {
        LoginResponse response = authService.refreshToken(request);
        return Result.success("刷新成功", response);
    }

    @GetMapping("/user/info")
    @Operation(summary = "获取用户信息", description = "获取当前登录用户信息")
    public Result<UserInfo> getUserInfo(HttpServletRequest request) {
        Long userId = getUserIdFromRequest(request);
        if (userId == null) {
            return Result.error(401, "未登录或登录已过期");
        }
        UserInfo userInfo = authService.getUserInfo(userId);
        return Result.success(userInfo);
    }

    @PutMapping("/user/info")
    @Operation(summary = "更新用户信息", description = "更新当前登录用户信息")
    public Result<UserInfo> updateUserInfo(
            HttpServletRequest request,
            @Valid @RequestBody UpdateUserRequest updateRequest) {
        Long userId = getUserIdFromRequest(request);
        if (userId == null) {
            return Result.error(401, "未登录或登录已过期");
        }
        UserInfo userInfo = authService.updateUserInfo(userId, updateRequest);
        return Result.success("更新成功", userInfo);
    }

    @PutMapping("/user/password")
    @Operation(summary = "修改密码", description = "修改当前登录用户密码")
    public Result<Void> changePassword(
            HttpServletRequest request,
            @Valid @RequestBody ChangePasswordRequest passwordRequest) {
        Long userId = getUserIdFromRequest(request);
        if (userId == null) {
            return Result.error(401, "未登录或登录已过期");
        }
        authService.changePassword(userId, passwordRequest);
        return Result.success("密码修改成功，请重新登录", null);
    }

    @GetMapping("/validate")
    @Operation(summary = "验证Token", description = "验证Token是否有效（供网关调用）")
    public Result<UserInfo> validateToken(
            @Parameter(description = "JWT Token") @RequestParam String token) {
        UserInfo userInfo = authService.validateToken(token);
        if (userInfo != null) {
            return Result.success(userInfo);
        }
        return Result.error(401, "Token无效或已过期");
    }

    /**
     * 从请求中获取用户ID
     */
    private Long getUserIdFromRequest(HttpServletRequest request) {
        // 优先从UserContext获取
        if (UserContext.getUserId() != null) {
            return UserContext.getUserId();
        }

        // 从Header中解析Token
        String authorization = request.getHeader(Constants.TOKEN_HEADER);
        if (authorization != null && authorization.startsWith(Constants.TOKEN_PREFIX)) {
            String token = authorization.substring(Constants.TOKEN_PREFIX.length());
            UserInfo userInfo = authService.validateToken(token);
            if (userInfo != null) {
                return userInfo.getId();
            }
        }
        return null;
    }
}
