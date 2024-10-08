<template>
    <body>
        <div class="parent">
            <Board />
        </div>
        <div class="buttons">
            <button @click="backToMenu" class="btn-with-bg btn1">
                <span class="tibetan-text">ཕྱིར། </span> <!-- 示例藏文文本 -->
                <span class="chinese-text">回到主页</span>
            </button>
            <button @click="resetGame" class="btn-with-bg btn">
                <span class="tibetan-text">བསྐྱར་དུ་འགོ་རྩོམ་པ། </span> <!-- 示例藏文文本 -->
                <span class="chinese-text">重新开始</span>
            </button>
            <button @click="goToGuide" class="btn-with-bg btn2">
                <span class="tibetan-text">སྒྲིག་ལམ་ངོ་སྤྲོད། </span> <!-- 示例藏文文本 -->
                <span class="chinese-text">规则介绍</span>
            </button>
        </div>
    </body>
</template>

<script setup>
import Board from '../components/board.vue';
import { useRouter } from 'vue-router';

const resetGame = () => {
    window.location.reload();
};

const router = useRouter();

const backToMenu = () => {
    router.push('/');
};

const goToGuide = () => {
    router.push('/guide');
};
</script>

<style scoped>

body {
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

.parent {
    display: flex;
    justify-content: center;
    align-items: center;
    margin-top: 20px;
}

.buttons {
    display: flex;
    justify-content: center;
    align-items: center;
    gap: 80px; /* 按钮之间的间距 */
}

/* 通用按钮样式 */
.btn-with-bg {
    width: 140px;
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
    font-size: 16px;
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
</style>
