<template>
    <div class="guide">
        <div class="Guide-Button">
            <!-- 按钮1 -->
            <div class="button-with-text btn1">
                <button @click="goToGuide('Guide01')" class="btn-with-bg-G"></button>
                <div class="text-container">
                    <span class="tibetan-text-G">ངོས་འཛིན་ཡུན་རིང་འགྲིལ་བ་རྨང་གཞི། </span>
                    <span class="chinese-text-G">01-认识久棋基础</span>
                </div>
            </div>
            <!-- 按钮2 -->
            <div class="button-with-text btn2">
                <button @click="goToGuide('Guide02')" class="btn-with-bg-G">
                </button>
                <div class="text-container">
                    <span class="tibetan-text-G">ཡུན་རིང་འགྲིལ་བ་བསྐྲུན་པའི་འདུ་ཤེས། </span>
                    <span class="chinese-text-G">02-久棋的开局意识</span>
                </div>
            </div>
            <!-- 按钮3 -->
            <div class="button-with-text btn3">
                <button @click="goToGuide('Guide03')" class="btn-with-bg-G">
                </button>
                <div class="text-container">
                    <span class="tibetan-text-G">བྱ་སྤྱོད་འགྲིལ་བ་ཐབས་ཇུས་འགྲེལ་བཤད། </span>
                    <span class="chinese-text-G">03-行棋策略讲解</span>
                </div>
            </div>
            <!-- 按钮4 -->
            <div class="button-with-text btn4">
                <button @click="goToGuide('Guide04')" class="btn-with-bg-G">
                </button>
                <div class="text-container">
                    <span class="tibetan-text-G">ཡུན་རིང་འགྲིལ་བའི་བྱ་སྤྱོད་འགྲིལ་བ། </span>
                    <span class="chinese-text-G">04-行棋及胜利阵型</span>
                </div>
            </div>
            <!-- 按钮5 -->
            <div class="button-with-text btn5">
                <button @click="goToGuide('Guide05')" class="btn-with-bg-G">
                </button>
                <div class="text-container">
                    <span class="tibetan-text-G">ཡུན་རིང་འགྲིལ་བ་མཇུག་འབྲས། </span>
                    <span class="chinese-text-G">05-终局判定及特殊阵型</span>
                </div>
            </div>
        </div>
        <div class="button-container">
            <button @click="backToMenu" class="btn-with-bg">
                <span class="tibetan-text">ཕྱིར།</span>
                <span class="chinese-text">回到主页</span>
            </button>
            <button @click="startGame" class="btn-with-bg">
                <span class="tibetan-text">ཐོག་མར་རོལ་རྩེད། </span>
                <span class="chinese-text">开始游戏</span>
            </button>
        </div>
    </div>
</template>

<style scoped>
/* 包含按钮和文本的容器 */
.button-with-text {
    position: absolute;
}

/* 鼠标悬停时文本容器的背景颜色变化 */
.button-with-text:hover .text-container {
    transform: scale(1.1);
    background-color: rgba(255, 193, 7, 0.8); /* 半透明的金黄色背景 */
    box-shadow: 0 4px 10px rgba(247, 174, 7, 0.8); /* 完整的box-shadow属性 */
    transition: transform 0.3s ease, background-color 0.6s ease, box-shadow 0.3s ease;
}

/* 鼠标悬停时文本容器和按钮背景同时发生变化 */
.button-with-text:hover .btn-with-bg-G {
    transform: scale(1.2);
    background-color: rgba(255, 193, 7, 0.2); /* 半透明的金黄色背景 */
    box-shadow: 0 4px 10px rgba(247, 174, 7, 0.8); /* 完整的box-shadow属性 */
    transition: transform 0.3s ease, background-color 0.6s ease, box-shadow 0.3s ease;
}

/* 文本容器的样式 */
.text-container {
    bottom: -70px;
    height: 60px;
    width: auto;
    position: relative;
    background-color: #00384d;
    border: 3px solid #002c41;

    /* 半透明深蓝色背景 */
    border-radius: 10px;
    display: flex;
    flex-direction: column;
    /* 将中文和藏文文本垂直排列 */
    justify-content: center;
    /* 水平居中 */
    align-items: center;
    /* 垂直居中 */
}

/* 藏文文本样式 */
.tibetan-text-G {
    font-family: 'Tibetan', sans-serif;
    /* 你需要引入合适的藏文字体 */
    font-size: 15px;
    color: white;
    margin-bottom: 4px;
    /* 调整藏文与中文文本之间的间距 */
    text-align: center;
    /* 确保文本在容器中居中对齐 */
}

/* 中文文本样式 */
.chinese-text-G {
    font-family: 'Tibetan', sans-serif;
    /* 你需要引入合适的藏文字体 */
    font-size: 15px;
    color: white;
    text-align: center;
    /* 确保文本在容器中居中对齐 */
    margin-bottom: 4px;
    /* 调整文本与容器底部的间距 */
}

/* 按钮默认样式 */
.btn-with-bg-G {
    position: relative;
    background-color: transparent;
    border: 0;
    width: 180px;
    height: 180px;
    background-size: 50%;
    border-radius: 50%;
    overflow: hidden; /* 确保遮罩层不超出按钮 */
}

/* 返回按钮的通用样式 */
.btn-with-bg {
    width: 160px;
    height: 90px;
    background-color: #00384d;
    /* 深蓝色背景 */
    border-radius: 20px;
    /* 圆角 */
    display: flex;
    justify-content: center;
    align-items: center;
    flex-direction: column;
    position: relative;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    color: white;
    margin: 20px;
    transition: transform 0.3s ease, box-shadow 0.3s ease;
    /* 放大动画的过渡效果 */
    border: 5px solid #002c41;
    /* 深蓝色边框，宽度为5px */
    transition: transform 0.3s ease, box-shadow 0.3s ease; /* 放大动画的过渡效果 */
    border: 5px solid #002c41; /* 深蓝色边框，宽度为5px */
}

/* 鼠标悬停时按钮放大效果 */
.btn-with-bg:hover {
    transform: scale(1.02);
    /* 放大1.1倍 */
    box-shadow: 0 6px 12px rgba(0, 0, 0, 0.3);
    /* 增加阴影，使放大效果更明显 */
    background-color: skyblue;
    /* 深蓝色背景 */
  transform: scale(1.02); /* 放大1.1倍 */
  box-shadow: 0 6px 12px rgba(0, 0, 0, 0.3); /* 增加阴影，使放大效果更明显 */
  background-color: skyblue; /* 深蓝色背景 */
}


/* 按钮中的中文文本样式 */
.chinese-text {
    background: linear-gradient(to right, #e06b42, #b24312);
    /* 渐变橙色背景 */
    padding: 10px 25px;
    font-size: 16px;
    background: linear-gradient(to right, #e06b42, #b24312); /* 渐变橙色背景 */
    padding: 10px 25px;
    font-size: 20px;
    color: white;
    margin-bottom: 8px;
}

/* 中文文本样式 */
.chinese-text {
    background: linear-gradient(to right, #e06b42, #b24312);
    /* 渐变橙色 */
    padding: 6px 25px;
    font-size: 16px;
    color: white;
    position: relative;
    clip-path: polygon(10% 0%, 90% 0%, 100% 25%, 100% 75%, 90% 100%, 10% 100%, 0% 75%, 0% 25%);
    /* 内角形状 */
    border-radius: 5px;
    /* 圆角内边界 */
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    text-align: center;
    /* 文本居中 */
}

/* 藏文文本样式 */
.tibetan-text {
    font-family: 'Tibetan', sans-serif;
    /* 如果有藏文字体，可以指定 */
    font-size: 16px;
    color: #fff;
    /* 白色字体 */
    font-family: 'Tibetan', sans-serif; /* 如果有藏文字体，可以指定 */
    font-size: 30px;
    color: #fff; /* 白色字体 */
    margin-bottom: 10px;
}

/* 设置按钮的父容器为横向排列 */
.button-container {
    position: absolute;
    bottom: 5%;
    display: flex;
    justify-content: center;
    /* 水平居中 */
    gap: 20px;
    /* 按钮之间的间距 */
}

.Guide-Button {
    display: flex;
    justify-content: center;
    /* 水平居中 */
    gap: 20px;
    /* 按钮之间的间距 */
}

/* 设置页面背景图片 */
.guide {
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

/* 背景图片及鼠标悬停效果 */
.btn1 {
    top: 10%;
    left: 10%;
    background-image: url('../assets/B01.png');
    background-size: cover;
}

.btn2 {
    bottom: 40%;
    left: 28%;
    background-image: url('../assets/B02.png');
    background-size: cover;
}

.btn3 {
    top: 10%;
    left: 45%;
    background-image: url('../assets/B03.png');
    background-size: cover;
}

.btn4 {
    bottom: 40%;
    right: 28%;
    background-image: url('../assets/B04.png');
    background-size: cover;
}

.btn5 {
    top: 10%;
    right: 10%;
    background-image: url('../assets/B05.png');
    background-size: cover;
}

/* 鼠标悬停时放大效果 */
.btn-with-bg:hover {
    transform: scale(1.1);
    box-shadow: 0 6px 12px rgba(0, 0, 0, 0.3);
}
</style>

<script setup>
import { useRouter } from 'vue-router';

const router = useRouter();

// 跳转到不同的指南页面
const goToGuide = (guideId) => {
    switch (guideId) {
        case 'Guide01':
            router.push({ name: 'Guide01' });
            break;
        case 'Guide02':
            router.push({ name: 'Guide02' });
            break;
        case 'Guide03':
            router.push({ name: 'Guide03' });
            break;
        case 'Guide04':
            router.push({ name: 'Guide04' });
            break;
        case 'Guide05':
            router.push({ name: 'Guide05' });
            break;
        default:
            console.error('Unknown guide:', guideId);
    }
}

const backToMenu = () => {
    router.push('/');
}

const startGame = () => {
    router.push('/play');
}
</script>
