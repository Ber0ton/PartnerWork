<template>
  <div class="Guide04">
    <!-- 标题部分 -->
    <div class="header">
      <h1 class="title">04——久棋的行棋及胜利阵型</h1>
      <p class="description">
        本章介绍久棋中的两种胜利棋型：平口褡裢和斜口褡裢。
      </p>
    </div>

    <!-- 内容部分 -->
    <div class="content">
      <div class="guide-text">
        <!-- 平口褡裢 -->
        <h3>1. 平口褡裢</h3>
        <p>
          平口褡裢由 7 枚棋子组成，通过来回移动一枚棋子，可以连续成方吃掉对方的棋子。
          平口褡裢和斜口褡裢是两种不同的褡裢棋型。
        </p>
        <img src="..\assets\Win1.jpg" alt="平口褡裢示意图" class="strategy-image" />
        <p>右图展示的是8种不同的平口褡裢棋型。</p>

        <!-- 斜口褡裢 -->
        <h3>2. 斜口褡裢</h3>
        <p>
          斜口褡裢与平口褡裢类似，但其布局为斜线。通过来回移动一枚棋子，也可以连续成方吃掉对方棋子。
        </p>
        <img src="..\assets\Win2.jpg" alt="斜口褡裢示意图" class="strategy-image" />
        <p>右图展示的是4种不同的斜口褡裢棋型。</p>
      </div>
    </div>

    <!-- 按钮部分 -->
    <div class="button-container">
      <button @click="goToUpPage" class="btn-with-bg">
        <span class="tibetan-text">སྟེང་ཤོག་ངོས། </span> <!-- 藏文文本 -->
        <span class="chinese-text">上一页</span>
      </button>
      <button @click="goToNextPage" class="btn-with-bg">
        <span class="tibetan-text">རྗེས་མའི་ཤོག་ངོས།</span> <!-- 藏文文本 -->
        <span class="chinese-text">下一页</span>
      </button>
      <button @click="goToGuide" class="btn-with-bg">
        <span class="tibetan-text">ཕྱིར།</span> <!-- 藏文文本 -->
        <span class="chinese-text">返回</span>
      </button>
      <button @click="startGame" class="btn-with-bg">
        <span class="tibetan-text">ཐོག་མར་རོལ་རྩེད།</span> <!-- 藏文文本 -->
        <span class="chinese-text">开始游戏</span>
      </button>
    </div>
  </div>
</template>

<script setup>
import { useRouter } from 'vue-router';

const router = useRouter();

const goToUpPage = () => {
  router.push({ name: 'Guide03' });
}

// 跳转到下一页 Guide05
const goToNextPage = () => {
  router.push({ name: 'Guide05' });
}

// 返回 Guide 页面
const goToGuide = () => {
  router.push({ path: '/guide' });
}

// 跳转到开始游戏 Play 页面
const startGame = () => {
  router.push({ path: '/play' });
}
</script>

<style scoped>
/* 页面样式 */
.Guide04 {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: flex-start;
  padding: 20px;
  background-image: url('../assets/Play-BackGround.png');
  background-size: cover;
  background-repeat: no-repeat;
  overflow-y: auto; /* 允许垂直滚动 */
}

.header {
  background-color: #e06b42;
  padding: 15px;
  border-radius: 10px;
  text-align: center;
  color: #fff;
}

.title {
  font-size: 28px;
  font-weight: bold;
}

.description {
  margin-top: 10px;
  font-size: 18px;
  background: #ffefd5;
  padding: 5px 15px;
  border-radius: 10px;
  display: inline-block;
  color: #b24312;
}

.content {
  display: flex;
  flex-direction: column;
  align-items: center;
  margin-top: 20px;
}

.strategy-image {
  display: block; /* 使图片成为块级元素 */
  margin: 15px auto; /* 自动水平居中 */
  background-image: transparent;
  width: 300px;
  height: auto;
  border-radius: 10px;
}

.guide-text {
  color: white;
  font-size: medium;
  text-align: left;
  padding: 20px;
}

.guide-text h3, h4 {
  margin-top: 15px;
}

/* 通用按钮样式 */
.btn-with-bg {
  width: 160px;
  height: 90px;
  background-color: #00384d; /* 深蓝色背景 */
  border-radius: 20px; /* 圆角 */
  display: flex;
  justify-content: center;
  align-items: center;
  flex-direction: column;
  position: relative;
  box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
  color: white;
  margin: 10px;
  transition: all 0.3s ease;
  border: 5px solid #002c41; /* 深蓝色边框，宽度为5px */
}

/* 藏文文本样式 */
.tibetan-text {
  font-family: 'Tibetan', sans-serif; /* 你需要引入合适的藏文字体 */
  font-size: 20px;
  color: white;
  margin-bottom: 8px;
}

/* 中文文本样式 */
.chinese-text {
  background: linear-gradient(to right, #e06b42, #b24312); /* 渐变橙色 */
  padding: 6px 25px;
  font-size: 16px;
  color: white;
  position: relative;
  clip-path: polygon(10% 0%, 90% 0%, 100% 25%, 100% 75%, 90% 100%, 10% 100%, 0% 75%, 0% 25%); /* 内角形状 */
  border-radius: 5px; /* 圆角内边界 */
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
  text-align: center; /* 文本居中 */
}

/* 鼠标悬停时按钮放大效果 */
.btn-with-bg:hover {
  transform: scale(1.1); /* 放大1.1倍 */
  box-shadow: 0 6px 12px rgba(0, 0, 0, 0.3); /* 增加阴影，使放大效果更明显 */
  background-color: #87ceeb; /* 深蓝色背景 */
}

/* 设置按钮的父容器为横向排列 */
.button-container {
  display: flex;
  justify-content: center; /* 水平居中 */
  gap: 20px; /* 按钮之间的间距 */
}
</style>
