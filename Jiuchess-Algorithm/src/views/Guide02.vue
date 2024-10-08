<template>
  <div class="Guide02">
    <!-- 标题部分 -->
    <div class="header">
      <h1 class="title">02——久棋的开局意识</h1>
      <p class="description">
        开局后：<br />
        从棋盘中央棋格对角线对角线两端开始布子，双方轮流走子，直至布满整个棋盘。
      </p>
    </div>

    <!-- 图片和内容部分 -->
    <div class="content">
      <!-- 棋盘图片 -->
      <div class="chess-image">
        <img src="..\assets\ChessBoard-Start.png" alt="Chess Board" />
      </div>

      <!-- 文本介绍部分 -->
      <div class="guide-text">
        <h3>1.</h3>
        <p>
          布局时，围绕着中间棋子，棋手需要尽量让己方棋子聚集，合拢为一个整体，以便形成“方”形阵型。<br />这种“成方”的策略不仅能够帮助己方更好地掌控棋局，还能够在对抗中占据主动。同时，<br />棋手也要时刻注意对方的布局动向，尽可能阻止对方形成“方”，从而打乱对方的战略布局，削弱其攻防能力。
          </p>
          <h3>2.</h3>
          <p>
          在久棋的对局中，黑白双方按照各自的思路和策略在棋盘上布满棋子。每一步落子都至关重要，落子无悔，<br />意味着每个决定都不可更改，这进一步增加了对局的紧张感与策略性。久棋从开局的布<br />局阶段到对局的走子阶段，棋手对棋阵模式的依赖性非常高，成功的阵型往往决定了对局的最终走向。
          </p>
          <h3>3.</h3>
          <p>
          根据久棋的规则，形成“方”形阵型在战略上具有极高的价值。这种阵型不仅能够在进攻时起到至关重要的作用，<br />成为摧垮对方防线的关键一环；另一方面，成“方”也有助于棋手构建有效的防守阵<br />型，使己方阵线更加稳固，减少被对方突破的可能性。通过灵活运用“方”形阵型，棋手能够在攻守转换中迅速调整策略，<br />进一步提高己方在对局中的优势。
          </p>
          
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
  router.push({ name: 'Guide01' });
}

// 跳转到下一页 Guide03
const goToNextPage = () => {
  router.push({ name: 'Guide03' });
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
/* 整体页面背景及布局样式 */
.Guide02 {
  width: auto;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  background-image: url('../assets/Play-BackGround.png');
  background-size: cover;
  background-repeat: no-repeat;
  overflow-y: scroll;
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
  align-items: center;       /* 垂直居中内容 */
  margin-top: 20px;
  min-height: 500px;         /* 设置容器最小高度 */
}


/* 棋盘图片容器样式 */
.chess-image {
  display: flex;
  justify-content: center;  /* 水平居中 */
  align-items: center;      /* 垂直居中 */
  height: 100%;             /* 使容器高度撑满可用空间 */
}

/* 棋盘图片样式 */
.chess-image img {
  width: 300px;
  height: auto;
  border-radius: 10px;
  border: 2px solid #fff;
  max-height: 100%;         /* 确保图片不会超过容器高度 */
}


/* 文本部分样式 */
.guide-text {
  color: white;
  font-size: medium;
  text-align: left;
  flex: 1;
  padding-left: 20px;
}

.guide-text p {
  margin-top: 10px;
  font-size: 16px;
  line-height: 1.6;
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
  background-color: skyblue; /* 深蓝色背景 */
}

/* 设置按钮的父容器为横向排列 */
.button-container {
  display: flex;
  justify-content: center; /* 水平居中 */
  gap: 20px; /* 按钮之间的间距 */
}
</style>
