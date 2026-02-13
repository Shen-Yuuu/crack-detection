import request from '@/utils/request'

// 登录
export function login(data) {
  return request({
    url: '/auth/login',
    method: 'post',
    data
  })
}

// 注册
export function register(data) {
  return request({
    url: '/auth/register',
    method: 'post',
    data
  })
}

// 登出
export function logout() {
  return request({
    url: '/auth/logout',
    method: 'post'
  })
}

// 获取用户信息
export function getUserInfo() {
  return request({
    url: '/auth/user/info',
    method: 'get'
  })
}

// 刷新Token
export function refreshToken(data) {
  return request({
    url: '/auth/refresh',
    method: 'post',
    data
  })
}

// 修改密码
export function changePassword(data) {
  return request({
    url: '/auth/user/password',
    method: 'put',
    data
  })
}

// 更新用户信息
export function updateUserInfo(data) {
  return request({
    url: '/auth/user/info',
    method: 'put',
    data
  })
}

// 更新用户资料 (别名)
export const updateProfile = updateUserInfo

