import axios from 'axios'
import { ElMessage, ElMessageBox } from 'element-plus'
import { useUserStore } from '@/stores/user'

// 创建 axios 实例
const service = axios.create({
  baseURL: '/api/v1',
  timeout: 60000
})

// 标记是否正在刷新token
let isRefreshing = false
// 等待刷新token的请求队列
let refreshSubscribers = []

// 将请求添加到等待队列
const subscribeTokenRefresh = (callback) => {
  refreshSubscribers.push(callback)
}

// 刷新成功后执行等待队列中的请求
const onRefreshed = (token) => {
  refreshSubscribers.forEach(callback => callback(token))
  refreshSubscribers = []
}

// 请求拦截器
service.interceptors.request.use(
  (config) => {
    const userStore = useUserStore()
    if (userStore.token) {
      config.headers['Authorization'] = `Bearer ${userStore.token}`
    }
    return config
  },
  (error) => {
    console.error('请求错误:', error)
    return Promise.reject(error)
  }
)

// 响应拦截器
service.interceptors.response.use(
  (response) => {
    const res = response.data
    
    // 如果响应成功
    if (res.code === 200) {
      return res
    }
    
    // 业务层面的401错误（Token无效）
    if (res.code === 401) {
      const userStore = useUserStore()
      // 不显示错误消息，让调用方处理
      userStore.clearAuth()
      return Promise.reject(new Error(res.message || '登录已过期'))
    }
    
    // 其他业务错误
    ElMessage.error(res.message || '请求失败')
    return Promise.reject(new Error(res.message || '请求失败'))
  },
  async (error) => {
    const { response, config } = error
    const userStore = useUserStore()

    if (response) {
      switch (response.status) {
        case 401:
          // 如果是刷新token的请求失败，直接登出
          if (config.url?.includes('/auth/refresh')) {
            userStore.clearAuth()
            return Promise.reject(error)
          }
          
          // Token 过期，尝试刷新
          if (userStore.refreshToken) {
            if (!isRefreshing) {
              isRefreshing = true
              try {
                await userStore.refreshUserToken()
                isRefreshing = false
                onRefreshed(userStore.token)
                // 重试原请求
                config.headers['Authorization'] = `Bearer ${userStore.token}`
                return service(config)
              } catch (refreshError) {
                isRefreshing = false
                refreshSubscribers = []
                // 刷新失败，跳转登录
                ElMessageBox.confirm('登录已过期，请重新登录', '提示', {
                  confirmButtonText: '确定',
                  type: 'warning',
                  showCancelButton: false
                }).then(() => {
                  userStore.clearAuth()
                })
                return Promise.reject(refreshError)
              }
            } else {
              // 等待token刷新完成
              return new Promise((resolve) => {
                subscribeTokenRefresh((token) => {
                  config.headers['Authorization'] = `Bearer ${token}`
                  resolve(service(config))
                })
              })
            }
          } else {
            userStore.clearAuth()
          }
          break
        case 403:
          ElMessage.error('没有权限访问')
          break
        case 404:
          ElMessage.error('请求的资源不存在')
          break
        case 500:
          ElMessage.error('服务器内部错误')
          break
        default:
          ElMessage.error(response.data?.message || '请求失败')
      }
    } else {
      ElMessage.error('网络连接失败，请检查网络')
    }
    
    return Promise.reject(error)
  }
)

export default service
