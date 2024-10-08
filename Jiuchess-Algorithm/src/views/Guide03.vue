<template>
  <div class="Guide03">
    <!-- 标题部分 -->
    <div class="header">
      <h1 class="title">03——久棋的行棋策略</h1>
      <p class="description">
        久棋对弈分为布子、走子、飞子三个阶段，我们将详细介绍这些策略。
      </p>
    </div>

    <!-- 文本和图片部分 -->
    <div class="content">
      <div class="guide-text">
        <!-- 布子阶段 -->
        <h3>1. 布子阶段</h3>
        <p>
          在“久棋”对弈中，双方各执一色棋子，棋盘空秤开局。先行者开始布子，按照“白先黑后”的规则。布子时，双方必须首先在棋盘中央的对角线上各放置一枚棋子，先行者决定落子的位置。随后，双方轮流在棋盘上交替着子，直到将棋盘的棋位布满为止。
        </p>
        
        <!-- 走子阶段 -->
        <h3>2. 走子阶段</h3>
        <p>
          布子结束后，双方首先将中格对角线上的棋子移去，由后布子方开始走子阶段（黑先白后）。走子阶段中，每次只能移动一个棋子，双方轮流走子。在这个阶段中，玩家通过“成方”、“跳吃”或“成阵型”来提取对方棋子。
        </p>
        <img src="..\assets\Jump-Eat.jpg" alt="跳吃示意图" class="strategy-image" />
        <p>黑棋从 (4, D) 移动到 (2, D)，通过跳吃吃掉白棋 (3, D)。</p>

        <!-- 飞子阶段 -->
        <h3>3. 飞子阶段</h3>
        <p>
          在对弈的后期，当一方的棋子数等于或少于棋盘路数时（14x14棋盘，棋子等于或少于14枚），少子方进入飞子阶段。此时，少子方移动棋子不再受限，可以落至任意空棋位，阻止对方“成方”或“成阵型”。通过飞子策略，弱势方有机会逆转局势，反败为胜。
        </p>
        
        <!-- 吃棋子策略 -->
        <h3>4. 吃棋子策略</h3>
        <p>
          行棋时，己方的棋子与对方棋子相邻，当在同一行或列的第三个点为空时，己方可以通过跳吃来吃掉对方的棋子。跳吃可以单吃或连吃。（如右图所示）
        </p>

        <!-- 棋门策略 -->
        <h4>（1）棋门</h4>
        <p>
          棋盘上共有196个格子，形成“棋门”是久棋的关键策略之一。每当己方棋子占据某一格子四个顶点，便形成一个棋门。每形成一个棋门，己方可以吃掉对方任意一枚棋子。双棋门则是两个单棋门连接在一起，每形成一个双棋门，可以吃掉对方任意两枚棋子。
        </p>
        <img src="..\assets\Single-Door.png" alt="单棋门示意图" class="strategy-image" />
        <p>黑棋通过形成单棋门，吃掉对方的一枚棋子。</p>

        <!-- 褡裢策略 -->
        <h4>（2）褡裢</h4>
        <p>
          单褡裢是指在两个棋门之间有一个空格，己方通过来回移动棋子关闭两边棋门的形式。双褡裢则是在两个棋门之间来回移动棋子，关闭棋门以吃掉对方任意两枚棋子，这是赢棋的基本条件和重要阵型。
        </p>
        <img src="..\assets\Double-Pouch.png" alt="双褡裢示意图" class="strategy-image" />
        <p>平口褡裢与斜口褡裢示意图。</p>

        <!-- 抢和煞策略 -->
        <h4>（3）抢和煞</h4>
        <p>
          抢：当己方走一颗棋子时，一条直线上全为对方棋子，己方可以吃掉对方任意两枚棋子，这种形状称为“抢”。煞：当己方在一条直线上棋子全为己方且形成棋门时，可以吃掉对方三至四颗棋子，这种形状称为“煞”。
        </p>
        <img src="..\assets\Qiang_Sha.png" alt="抢与煞示意图" class="strategy-image" />
        <p>抢与煞的棋形示意图。</p>
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
  router.push({ name: 'Guide02' });
}

// 跳转到下一页 Guide04
const goToNextPage = () => {
  router.push({ name: 'Guide04' });
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
.Guide03 {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  background-image: url('../assets/Play-BackGround.png');
  background-size: cover;
  background-repeat: no-repeat;
  overflow-y: scroll
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
