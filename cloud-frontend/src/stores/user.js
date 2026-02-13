import { defineStore } from 'pinia'
import { login, logout, getUserInfo, refreshToken } from '@/api/auth'
import router from '@/router'

export const useUserStore = defineStore('user', {
  state: () => ({
    token: localStorage.getItem('token') || '',
    refreshToken: localStorage.getItem('refreshToken') || '',
    userInfo: null
  }),

  getters: {
    isLoggedIn: (state) => !!state.token,
    username: (state) => state.userInfo?.username || '',
    avatar: (state) => state.userInfo?.avatar || ''
  },

  actions: {
    // 登录
    async login(loginForm) {
      try {
        const res = await login(loginForm)
        this.token = res.data.accessToken
        this.refreshToken = res.data.refreshToken
        localStorage.setItem('token', this.token)
        localStorage.setItem('refreshToken', this.refreshToken)
        
        // 获取用户信息
        await this.getUserInfo()
        
        return res
      } catch (error) {
        throw error
      }
    },

    // 获取用户信息
    async getUserInfo() {
      try {
        const res = await getUserInfo()
        this.userInfo = res.data
        return res
      } catch (error) {
        console.error('获取用户信息失败:', error)
        throw error
      }
    },

    // 刷新 Token
    async refreshUserToken() {
      try {
        const res = await refreshToken({ refreshToken: this.refreshToken })
        this.token = res.data.accessToken
        // 如果返回了新的refreshToken则更新，否则保留原值
        if (res.data.refreshToken) {
          this.refreshToken = res.data.refreshToken
          localStorage.setItem('refreshToken', this.refreshToken)
        }
        localStorage.setItem('token', this.token)
        return res
      } catch (error) {
        // 刷新失败，由调用方处理登出
        throw error
      }
    },

    // 清除认证信息（不调用API）
    clearAuth() {
      this.token = ''
      this.refreshToken = ''
      this.userInfo = null
      localStorage.removeItem('token')
      localStorage.removeItem('refreshToken')
      router.push('/login')
    },

    // 登出
    async logout() {
      try {
        if (this.token) {
          await logout()
        }
      } catch (error) {
        console.error('登出失败', error)
      } finally {
        this.clearAuth()
      }
    }
  }
})
