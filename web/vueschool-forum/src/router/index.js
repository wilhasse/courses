import PageHome from '@/pages/MyHome.vue'
import PageNotFound from '@/pages/NotFound.vue'
import PageThreadShow from '@/pages/ThreadShow.vue'
import ForumShow from '@/pages/ForumShow.vue'
import CategoryShow from '@/pages/CategoryShow.vue'
import { createRouter, createWebHistory } from 'vue-router'
import sourceData from '@/data.json'
import UserProfile from '@/pages/UserProfile.vue'

const routes = [
  {
    path: '/',
    name: 'Home',
    component: PageHome
  },
  {
    path: '/thread/:id',
    name: 'ThreadShow',
    component: PageThreadShow,
    props: true,
    beforeEnter (to, from, next) {
      // check if thread exists
      const threadExists = sourceData.threads.find(thread => thread.id === to.params.id)
      // if exists continue
      if (threadExists) {
        return next()
      } else {
        next({
          name: 'NotFound',
          params: { pathMatch: to.path.substring(1).split('/') },
          // preserve existing query and hash
          query: to.query,
          hash: to.hash
        })
      }
      // if doesnt exist redirect to not found
    }
  },
  {
    path: '/forum/:id',
    name: 'Forum',
    component: ForumShow,
    props: true
  },
  {
    path: '/me',
    name: 'Profile',
    component: UserProfile,
    meta: { toTop: true, smoothScroll: true }
  },
  {
    path: '/me/edit',
    name: 'ProfileEdit',
    component: UserProfile,
    props: { edit: true }
  },  
  {
    path: '/category/:id',
    name: 'Category',
    component: CategoryShow,
    props: true
  },
  {
    path: '/:pathMatch(.*)*',
    name: 'NotFound',
    component: PageNotFound
  }
]

export default createRouter({
  history: createWebHistory(),
  routes,
  scrollBehavior (to) {
    const scroll = {}
    if (to.meta.toTop) scroll.top = 0
    if (to.meta.smoothScroll) scroll.behavior = 'smooth'
    return scroll
  }
})