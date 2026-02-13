import { createRouter, createWebHistory } from 'vue-router'
import NProgress from 'nprogress'
import 'nprogress/nprogress.css'
import { useUserStore } from '@/stores/user'

// 路由配置
const routes = [
  {
    path: '/login',
    name: 'Login',
    component: () => import('@/views/login/index.vue'),
    meta: { title: '登录', public: true }
  },
  {
    path: '/register',
    name: 'Register',
    component: () => import('@/views/login/register.vue'),
    meta: { title: '注册', public: true }
  },
  {
    path: '/',
    component: () => import('@/layout/index.vue'),
    redirect: '/dashboard',
    children: [
      {
        path: 'dashboard',
        name: 'Dashboard',
        component: () => import('@/views/dashboard/index.vue'),
        meta: { title: '工作台', icon: 'Odometer' }
      },
      {
        path: 'detection',
        name: 'Detection',
        component: () => import('@/views/detection/index.vue'),
        meta: { title: '裂纹检测', icon: 'Camera' }
      },
      {
        path: 'detection/result/:jobId',
        name: 'DetectionResult',
        component: () => import('@/views/detection/result.vue'),
        meta: { title: '检测结果', hidden: true }
      },
      {
        path: 'dataset',
        name: 'Dataset',
        component: () => import('@/views/dataset/index.vue'),
        meta: { title: '数据集管理', icon: 'FolderOpened' }
      },
      {
        path: 'dataset/:id',
        name: 'DatasetDetail',
        component: () => import('@/views/dataset/detail.vue'),
        meta: { title: '数据集详情', hidden: true }
      },
      {
        path: 'history',
        name: 'History',
        component: () => import('@/views/history/index.vue'),
        meta: { title: '检测历史', icon: 'Timer' }
      },
      {
        path: 'report',
        name: 'Report',
        component: () => import('@/views/report/index.vue'),
        meta: { title: '报告管理', icon: 'Document' }
      },
      {
        path: 'profile',
        name: 'Profile',
        component: () => import('@/views/profile/index.vue'),
        meta: { title: '个人中心', icon: 'User', hidden: true }
      }
    ]
  },
  {
    path: '/:pathMatch(.*)*',
    name: 'NotFound',
    component: () => import('@/views/error/404.vue'),
    meta: { title: '404', public: true }
  }
]

const router = createRouter({
  history: createWebHistory(),
  routes
})

// 路由守卫
router.beforeEach((to, from, next) => {
  NProgress.start()
  document.title = `${to.meta.title} - 道路裂纹检测系统`

  const userStore = useUserStore()
  const isPublic = to.meta.public

  if (!isPublic && !userStore.token) {
    next({ name: 'Login', query: { redirect: to.fullPath } })
  } else {
    next()
  }
})

router.afterEach(() => {
  NProgress.done()
})

export default router
