import { createRouter, createWebHistory } from 'vue-router'
import Home from './views/Home.vue'
import Guide from './views/Guide.vue'
import Play from './views/Play.vue'
import About from './views/About.vue'
// 引入新创建的Vue文件
import Guide01 from './views/Guide01.vue';
import Guide02 from './views/Guide02.vue';
import Guide03 from './views/Guide03.vue';
import Guide04 from './views/Guide04.vue';
import Guide05 from './views/Guide05.vue';

const routes = [
    { path: '/', component: Home },
    { path: '/guide', component: Guide },
    { path: '/play', component: Play },
    { path: '/about', component: About },
    {
        path: '/guide01',
        name: 'Guide01',
        component: Guide01,
      },
      {
        path: '/guide02',
        name: 'Guide02',
        component: Guide02,
      },
      {
        path: '/guide03',
        name: 'Guide03',
        component: Guide03,
      },
      {
        path: '/guide04',
        name: 'Guide04',
        component: Guide04,
      },
      {
        path: '/guide05',
        name: 'Guide05',
        component: Guide05,
      },
];

const router = createRouter({
    history: createWebHistory(),
    routes
});

export default router;
