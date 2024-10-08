<template>
  <div class="Guide01">
    <!-- 标题部分 -->
    <div class="header">
      <h1 class="title">01——认识久棋基础</h1>
      <p class="description">
        藏棋“久”，是一种智力游戏，起源于藏区，蕴含着悠久岁月的藏文明和灿烂文化。
      </p>
    </div>

    <!-- 图片和内容部分 -->
    <div class="content">
      <!-- 棋盘图片 -->
      <div class="chess-image">
        <img src="..\assets\ChessBoard.png" alt="Chess Board" />
      </div>

      <!-- 文本介绍部分 -->
      <div class="guide-text">
        <h2>一、棋盘</h2>
        <p>
          棋盘有纵横各十四条等距的平行线构成，形成196个交叉点，在藏棋中每个交叉点就是落子的地方。藏棋称为落子点。<br />
          棋盘整体形状类似以及每个格子纵横向比相等。棋盘的中间棋格有一条对角斜线称为“久物”（如图1）。
        </p>
        <h2>二、下法</h2>
        <ol>
          <li>对局双方各执一色棋子；</li>
          <li>空框开局；</li>
          <li>落子先白后黑，行棋先黑后白，交替着下子于棋盘的交点上；</li>
          <li>棋子下定后，不再向其他点移动；</li>
          <li>轮流下子和行棋是对局双方的权利。</li>
        </ol>
      </div>
    </div>
     <!-- 按钮部分 -->
     <div class="button-container">
      <button @click="goToNextPage" class="btn-with-bg">
        <span class="tibetan-text">རྗེས་མའི་ཤོག་ངོས། </span> <!-- 藏文文本 -->
        <span class="chinese-text">下一页</span>
      </button>
      <button @click="goToGuide" class="btn-with-bg">
        <span class="tibetan-text">ཕྱིར། </span> <!-- 藏文文本 -->
        <span class="chinese-text">返回</span>
      </button>
      <button @click="startGame" class="btn-with-bg">
        <span class="tibetan-text">ཐོག་མར་རོལ་རྩེད། </span> <!-- 藏文文本 -->
        <span class="chinese-text">开始游戏</span>
      </button>
    </div>
  </div>
</template>

  
<script setup>
import { useRouter } from 'vue-router';

const router = useRouter();

// 跳转到下一页 Guide02
const goToNextPage = () => {
  router.push({ name: 'Guide02' });
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
  /* 设置页面背景图片 */
.Guide01 {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 100vh;
  background-image: url('../assets/Play-BackGround.png');
  background-size: cover;
  background-repeat: no-repeat;
  overflow: hidden;
}
/* 标题部分样式 */
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

/* 棋盘和文本容器样式 */
.content {
  display: flex;
  justify-content: space-between;
  margin-top: 20px;
}

/* 棋盘图片样式 */
.chess-image img {
  width: 300px;
  height: auto;
  border-radius: 10px;
  border: 2px solid #fff;
}

/* 文本部分样式 */
.guide-text {
  color: white;
  flex: 1;
  padding-left: 20px;
}

.guide-text h2 {
  margin-top: 20px;
  font-size: 20px;
}

.guide-text p {
  margin-top: 10px;
  font-size: 16px;
  line-height: 1.6;
}

ol {
  margin-left: 20px;
}

ol li {
  margin-top: 8px;
  font-size: 16px;
  line-height: 1.5;
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









