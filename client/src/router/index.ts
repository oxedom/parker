import { createRouter, createWebHistory, type RouteRecordRaw } from "vue-router";

const routes: RouteRecordRaw[] = [
  {
    path: "/",
    name: "home",
    component: () => import("@/pages/HomePage.vue"),
    meta: { title: "Parker: Empower your parking" },
  },
  {
    path: "/vision",
    name: "vision",
    component: () => import("@/pages/VisionPage.vue"),
    meta: { title: "Parker: Vision" },
  },
  {
    path: "/about",
    name: "about",
    component: () => import("@/pages/AboutPage.vue"),
    meta: { title: "Parker: About" },
  },
  {
    path: "/view",
    name: "view",
    component: () => import("@/pages/ViewPage.vue"),
    meta: { title: "Parker: View" },
  },
  {
    path: "/input",
    name: "input",
    component: () => import("@/pages/InputPage.vue"),
    meta: { title: "Parker: Input" },
  },
  {
    path: "/reroute",
    name: "reroute",
    component: () => import("@/pages/ReroutePage.vue"),
    meta: { title: "Parker: Menu" },
  },
  {
    path: "/:pathMatch(.*)*",
    redirect: { name: "home" },
  },
];

export const router = createRouter({
  history: createWebHistory(),
  routes,
});

router.afterEach((to) => {
  const title = (to.meta?.title as string | undefined) ?? "Parker";
  if (typeof document !== "undefined") document.title = title;
});
