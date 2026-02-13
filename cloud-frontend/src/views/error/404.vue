<template>
  <div class="error-page">
    <div class="error-content">
      <div class="error-code">404</div>
      <h2 class="error-title">页面未找到</h2>
      <p class="error-message">抱歉，您访问的页面不存在或已被移除</p>
      <div class="error-actions">
        <el-button type="primary" @click="goHome">
          <el-icon><HomeFilled /></el-icon>返回首页
        </el-button>
        <el-button @click="goBack">
          <el-icon><Back /></el-icon>返回上页
        </el-button>
      </div>
    </div>
    <div class="error-illustration">
      <svg viewBox="0 0 400 300" xmlns="http://www.w3.org/2000/svg">
        <defs>
          <linearGradient id="gradient" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" style="stop-color:#409eff;stop-opacity:0.8" />
            <stop offset="100%" style="stop-color:#67c23a;stop-opacity:0.8" />
          </linearGradient>
        </defs>
        <rect x="50" y="100" width="300" height="150" rx="10" fill="#f5f7fa" stroke="#dcdfe6" stroke-width="2"/>
        <circle cx="120" cy="130" r="20" fill="#f56c6c"/>
        <circle cx="200" cy="130" r="20" fill="#e6a23c"/>
        <circle cx="280" cy="130" r="20" fill="#67c23a"/>
        <rect x="80" y="170" width="240" height="60" rx="5" fill="#ebeef5"/>
        <text x="200" y="210" text-anchor="middle" fill="#909399" font-size="24" font-weight="bold">
          Page Not Found
        </text>
        <path d="M50 250 Q100 230 150 250 T250 250 T350 250" stroke="url(#gradient)" stroke-width="3" fill="none"/>
      </svg>
    </div>
  </div>
</template>

<script setup>
import { useRouter } from 'vue-router'

const router = useRouter()

const goHome = () => {
  router.push('/')
}

const goBack = () => {
  router.back()
}
</script>

<style lang="scss" scoped>
.error-page {
  min-height: 100vh;
  display: flex;
  align-items: center;
  justify-content: center;
  background: linear-gradient(135deg, #f5f7fa 0%, #e4e7ed 100%);
  padding: 20px;
}

.error-content {
  text-align: center;
  margin-right: 60px;

  .error-code {
    font-size: 120px;
    font-weight: bold;
    background: linear-gradient(135deg, #409eff 0%, #67c23a 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1;
  }

  .error-title {
    font-size: 28px;
    color: #303133;
    margin: 20px 0 10px;
  }

  .error-message {
    font-size: 16px;
    color: #909399;
    margin-bottom: 30px;
  }

  .error-actions {
    display: flex;
    gap: 16px;
    justify-content: center;
  }
}

.error-illustration {
  width: 400px;
  
  svg {
    width: 100%;
    height: auto;
  }
}

@media (max-width: 768px) {
  .error-page {
    flex-direction: column;
  }

  .error-content {
    margin-right: 0;
    margin-bottom: 40px;
  }

  .error-illustration {
    width: 280px;
  }
}
</style>
