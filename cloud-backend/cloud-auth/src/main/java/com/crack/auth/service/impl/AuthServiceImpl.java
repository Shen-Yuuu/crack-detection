package com.crack.auth.service.impl;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.crack.auth.dto.*;
import com.crack.auth.mapper.UserMapper;
import com.crack.auth.service.AuthService;
import com.crack.common.constant.Constants;
import com.crack.common.entity.User;
import com.crack.common.exception.BusinessException;
import com.crack.common.result.ResultCode;
import com.crack.common.utils.JwtUtils;
import com.crack.common.utils.RedisService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

/**
 * 认证服务实现
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class AuthServiceImpl implements AuthService {

    private final UserMapper userMapper;
    private final PasswordEncoder passwordEncoder;
    private final JwtUtils jwtUtils;
    private final RedisService redisService;

    @Override
    @Transactional(rollbackFor = Exception.class)
    public UserInfo register(RegisterRequest request) {
        log.info("用户注册: {}", request.getUsername());

        // 验证两次密码是否一致
        if (!request.getPassword().equals(request.getConfirmPassword())) {
            throw new BusinessException("两次输入的密码不一致");
        }

        // 检查用户名是否已存在
        Long count = userMapper.selectCount(new LambdaQueryWrapper<User>()
                .eq(User::getUsername, request.getUsername()));
        if (count > 0) {
            throw new BusinessException(ResultCode.USER_EXISTS);
        }

        // 检查邮箱是否已存在
        if (request.getEmail() != null) {
            Long emailCount = userMapper.selectCount(new LambdaQueryWrapper<User>()
                    .eq(User::getEmail, request.getEmail()));
            if (emailCount > 0) {
                throw new BusinessException("邮箱已被注册");
            }
        }

        // 创建用户
        User user = new User();
        user.setUsername(request.getUsername());
        user.setPassword(passwordEncoder.encode(request.getPassword()));
        user.setEmail(request.getEmail());
        user.setPhone(request.getPhone());
        user.setStatus(Constants.USER_STATUS_ENABLED);

        userMapper.insert(user);

        log.info("用户注册成功: {}", user.getId());
        return convertToUserInfo(user);
    }

    @Override
    public LoginResponse login(LoginRequest request) {
        log.info("用户登录: {}", request.getUsername());

        // 查询用户
        User user = userMapper.selectOne(new LambdaQueryWrapper<User>()
                .eq(User::getUsername, request.getUsername()));

        if (user == null) {
            throw new BusinessException(ResultCode.USER_PASSWORD_ERROR);
        }

        // 验证密码
        if (!passwordEncoder.matches(request.getPassword(), user.getPassword())) {
            throw new BusinessException(ResultCode.USER_PASSWORD_ERROR);
        }

        // 检查用户状态
        if (user.getStatus() == Constants.USER_STATUS_DISABLED) {
            throw new BusinessException(ResultCode.USER_DISABLED);
        }

        // 生成Token
        String accessToken = jwtUtils.generateAccessToken(user.getId(), user.getUsername());
        String refreshToken = jwtUtils.generateRefreshToken(user.getId(), user.getUsername());

        // 缓存Token到Redis
        String tokenKey = Constants.REDIS_USER_TOKEN_PREFIX + user.getId();
        redisService.setEx(tokenKey, accessToken, jwtUtils.getAccessTokenExpiration());

        log.info("用户登录成功: {}", user.getId());

        return LoginResponse.builder()
                .accessToken(accessToken)
                .refreshToken(refreshToken)
                .tokenType("Bearer")
                .expiresIn(jwtUtils.getAccessTokenExpiration())
                .userInfo(convertToUserInfo(user))
                .build();
    }

    @Override
    public void logout(Long userId) {
        log.info("用户登出: {}", userId);

        // 删除Redis中的Token
        String tokenKey = Constants.REDIS_USER_TOKEN_PREFIX + userId;
        redisService.delete(tokenKey);

        log.info("用户登出成功: {}", userId);
    }

    @Override
    public LoginResponse refreshToken(RefreshTokenRequest request) {
        String refreshToken = request.getRefreshToken();

        // 验证刷新令牌
        if (!jwtUtils.validateToken(refreshToken)) {
            throw new BusinessException(ResultCode.TOKEN_INVALID);
        }

        // 检查令牌类型
        String tokenType = jwtUtils.getTokenType(refreshToken);
        if (!"refresh".equals(tokenType)) {
            throw new BusinessException(ResultCode.TOKEN_INVALID);
        }

        // 获取用户信息
        Long userId = jwtUtils.getUserId(refreshToken);
        String username = jwtUtils.getUsername(refreshToken);

        // 查询用户
        User user = userMapper.selectById(userId);
        if (user == null || user.getStatus() == Constants.USER_STATUS_DISABLED) {
            throw new BusinessException(ResultCode.USER_NOT_FOUND);
        }

        // 生成新的访问令牌
        String newAccessToken = jwtUtils.generateAccessToken(userId, username);
        // 生成新的刷新令牌
        String newRefreshToken = jwtUtils.generateRefreshToken(userId, username);

        // 更新Redis中的Token
        String tokenKey = Constants.REDIS_USER_TOKEN_PREFIX + userId;
        redisService.setEx(tokenKey, newAccessToken, jwtUtils.getAccessTokenExpiration());

        log.info("刷新令牌成功: {}", userId);

        return LoginResponse.builder()
                .accessToken(newAccessToken)
                .refreshToken(newRefreshToken)
                .tokenType("Bearer")
                .expiresIn(jwtUtils.getAccessTokenExpiration())
                .build();
    }

    @Override
    public UserInfo getUserInfo(Long userId) {
        User user = userMapper.selectById(userId);
        if (user == null) {
            throw new BusinessException(ResultCode.USER_NOT_FOUND);
        }
        return convertToUserInfo(user);
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    public UserInfo updateUserInfo(Long userId, UpdateUserRequest request) {
        log.info("更新用户信息: {}", userId);

        User user = userMapper.selectById(userId);
        if (user == null) {
            throw new BusinessException(ResultCode.USER_NOT_FOUND);
        }

        // 更新字段
        if (request.getEmail() != null) {
            // 检查邮箱是否被其他用户使用
            Long emailCount = userMapper.selectCount(new LambdaQueryWrapper<User>()
                    .eq(User::getEmail, request.getEmail())
                    .ne(User::getId, userId));
            if (emailCount > 0) {
                throw new BusinessException("邮箱已被其他用户使用");
            }
            user.setEmail(request.getEmail());
        }

        if (request.getPhone() != null) {
            user.setPhone(request.getPhone());
        }

        if (request.getAvatar() != null) {
            user.setAvatar(request.getAvatar());
        }

        userMapper.updateById(user);

        log.info("用户信息更新成功: {}", userId);
        return convertToUserInfo(user);
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    public void changePassword(Long userId, ChangePasswordRequest request) {
        log.info("修改密码: {}", userId);

        // 验证两次密码是否一致
        if (!request.getNewPassword().equals(request.getConfirmPassword())) {
            throw new BusinessException("两次输入的密码不一致");
        }

        User user = userMapper.selectById(userId);
        if (user == null) {
            throw new BusinessException(ResultCode.USER_NOT_FOUND);
        }

        // 验证原密码
        if (!passwordEncoder.matches(request.getOldPassword(), user.getPassword())) {
            throw new BusinessException("原密码错误");
        }

        // 更新密码
        user.setPassword(passwordEncoder.encode(request.getNewPassword()));
        userMapper.updateById(user);

        // 清除Token，要求重新登录
        String tokenKey = Constants.REDIS_USER_TOKEN_PREFIX + userId;
        redisService.delete(tokenKey);

        log.info("密码修改成功: {}", userId);
    }

    @Override
    public UserInfo validateToken(String token) {
        try {
            if (!jwtUtils.validateToken(token)) {
                return null;
            }

            Long userId = jwtUtils.getUserId(token);

            // 检查Redis中是否存在该Token
            String tokenKey = Constants.REDIS_USER_TOKEN_PREFIX + userId;
            String cachedToken = redisService.get(tokenKey);
            if (cachedToken == null || !cachedToken.equals(token)) {
                return null;
            }

            User user = userMapper.selectById(userId);
            if (user == null || user.getStatus() == Constants.USER_STATUS_DISABLED) {
                return null;
            }

            return convertToUserInfo(user);
        } catch (Exception e) {
            log.warn("Token验证失败: {}", e.getMessage());
            return null;
        }
    }

    /**
     * 转换为UserInfo
     */
    private UserInfo convertToUserInfo(User user) {
        return UserInfo.builder()
                .id(user.getId())
                .username(user.getUsername())
                .email(user.getEmail())
                .phone(user.getPhone())
                .avatar(user.getAvatar())
                .status(user.getStatus())
                .createdAt(user.getCreatedAt())
                .build();
    }
}
