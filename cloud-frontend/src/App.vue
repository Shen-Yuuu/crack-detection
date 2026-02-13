<template>
  <router-view />
</template>

<script setup>
import { useUserStore } from '@/stores/user'

const userStore = useUserStore()

// 初始化用户信息
onMounted(async () => {
  if (userStore.token) {
    try {
      await userStore.getUserInfo()
    } catch (error) {
      console.error('初始化获取用户信息失败:', error)
      // 仅当token存在但获取用户信息失败时，不立即登出
      // 让请求拦截器处理401错误和token刷新
    }
  }
})
</script>

<style>
#app {
  width: 100%;
  height: 100%;
}
</style>
