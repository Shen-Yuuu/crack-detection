package com.crack.common.utils;

import com.crack.common.dto.LoginUser;

/**
 * 用户上下文工具类
 * 使用ThreadLocal存储当前请求的用户信息
 */
public class UserContext {

    private static final ThreadLocal<LoginUser> USER_HOLDER = new ThreadLocal<>();

    /**
     * 设置当前用户
     */
    public static void setUser(LoginUser user) {
        USER_HOLDER.set(user);
    }

    /**
     * 获取当前用户
     */
    public static LoginUser getUser() {
        return USER_HOLDER.get();
    }

    /**
     * 获取当前用户ID
     */
    public static Long getUserId() {
        LoginUser user = USER_HOLDER.get();
        return user != null ? user.getUserId() : null;
    }

    /**
     * 获取当前用户名
     */
    public static String getUsername() {
        LoginUser user = USER_HOLDER.get();
        return user != null ? user.getUsername() : null;
    }

    /**
     * 清除当前用户
     */
    public static void clear() {
        USER_HOLDER.remove();
    }
}
