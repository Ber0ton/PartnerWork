<template>
  <div class="Board">
    <div class="Status-bar">
      <div v-if="gamePhase === 'placing'">
        <h3>
          <span class="chinese-text">布子阶段</span><br>
          <span class="tibetan-text">གཞུང་ཚིག</span>
        </h3>
      </div>
      <div v-else-if="gamePhase === 'moving'">
        <span class="chinese-text">走子阶段</span><br>
        <span class="tibetan-text">གཞུང་ཚིག</span>
      </div>
      <div v-else-if="gamePhase === 'removing'">
        <span class="chinese-text">走子阶段</span><br>
        <span class="tibetan-text">གཞུང་ཚིག</span>
      </div>
    </div>

    <div class="chessboard">
      <div v-for="(row, rowIndex) in board" :key="rowIndex" class="row">
        <div v-for="(cell, colIndex) in row" :key="colIndex" class="cell"
          :class="{ 'highlight': canMoveTo(rowIndex, colIndex) }" @click="handleCellClick(rowIndex, colIndex)">
          <div v-if="cell !== null" :class="['piece',
            cell === 0 ? 'black-piece' : 'white-piece',
            isSelected(rowIndex, colIndex) ? 'selected-piece' : '']">
          </div>
        </div>
      </div>
    </div>

    <div class="game-info left">
      <!-- 白棋玩家的图片 -->
      <div class="player-container">
        <img src="..\assets\Player-L.png" alt="Player L" class="player-L-image" />
        <div class="circle-L" :class="{ 'highlight-Left': currentPlayer === 1 }"></div>
      </div>
      <div class="LeftInfo">
        <div v-if="gamePhase === 'placing'">
          <p>白子剩余 {{ whitePieceLeft }} 颗</p>
        </div>
        <div v-else-if="gamePhase === 'moving'">
          <p>白子请移动棋子</p>
          <p v-if="isWhiteFlying">白子飞行模式</p>
        </div>
        <div v-else-if="gamePhase === 'removing'">
          <p>白子请移除黑子</p>
          <p>待移除的棋子数：{{ piecessToRemove }}</p>
        </div>
      </div>
    </div>

    <div class="game-info right">
      <!-- 黑棋玩家的图片 -->
      <div class="player-container">
        <img src="..\assets\Player-R.png" alt="Player R" class="player-R-image" />
        <div class="circle-R" :class="{ 'highlight-Right': currentPlayer === 0 }"></div> <!-- 圆形 -->
      </div>
      <div class="RightInfo">
        <div v-if="gamePhase === 'placing'">
          <p>黑子剩余 {{ blackPieceLeft }} 颗</p>
        </div>
        <div v-else-if="gamePhase === 'moving'">
          <p>黑子请移动棋子</p>
          <p v-if="isBlackFlying">黑子飞行模式</p>
        </div>
        <div v-else-if="gamePhase === 'removing'">
          <p>黑子请移除白子</p>
          <p>待移除的棋子数：{{ piecessToRemove }}</p>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.Board {
  position: relative;
  display: flex;
  justify-content: center;
  align-items: center;
  margin: 0 auto;
  max-width: 1000px;
}

.row {
  display: contents;
}

.cell {
  width: 50px;
  height: 50px;
  /* border: 1px solid black; */
  width: 38px;
  height: 38px;
  border: 1px solid black;
  display: flex;
  justify-content: center;
  align-items: center;
  position: relative;
}

.cell.highlight {
  background-color: lightgreen;
}

.black-piece,
.white-piece {
  width: 30px;
  height: 30px;
  border-radius: 50%;
}

.black-piece {
  background-color: black;
}

.white-piece {
  background-color: white;
}

.selected-piece {
  border: 2px solid red;
}

.chessboard {
  text-align: center;
  width: 560px;
  height: 560px;
  width: 532px;
  display: grid;
  grid-template-columns: repeat(14, 38px);
  grid-template-rows: repeat(14, 38px);
  gap: 0;
  background-image: url('../assets/Chessboard.png');
  background-size: 125%;
  background-repeat: no-repeat;
  background-position: -70px -70px;
  background-color: #deb887;
  border: 1px solid black;
  margin-top: 10px;
}

.Status-bar {
  position: absolute;
  border-radius: 20px;
  /* 圆角 */
  left: -400px;
  top: -5px;
  text-align: center;
  /* 水平居中 */
  width: 150px;
  height: 100px;
  margin-top: 15px;
  margin-left: 15px;
}


.chinese-text {
  font-size: 25px;
  margin-bottom: 10px;
  /* 调整与藏文之间的距离 */
  color: white;
}

.tibetan-text {
  margin-top: 10px;
  /* 增加与汉字的距离 */
  color: white;
}

.player-container {
  position: relative;
  width: 150px;
  height: 200px;
  /* 控制容器高度，根据图片大小调整 */
}

.player-L-image {
  width: auto;
  height: 290px;
  position: relative;
  z-index: 1;
  /* 确保图片在上层 */
}

.player-R-image {
  width: auto;
  height: 310px;
  position: relative;
  z-index: 1;
  /* 确保图片在上层 */
}

.circle-L {
  width: 220px;
  /* 圆的大小 */
  height: 220px;
  background-color: #F7AE07;
  /* 圆的颜色 */
  border-radius: 50%;
  position: absolute;
  bottom: -80px;
  /* 控制圆的垂直位置，让它部分显示在图片下方 */
  left: 23px;
  z-index: 0;
  /* 确保圆在图片下层 */
  border: 5px solid #042433;
  /* 添加轮廓 */
}

.circle-R {
  width: 220px;
  /* 圆的大小 */
  height: 220px;
  background-color: #074059;
  /* 圆的颜色 */
  border-radius: 50%;
  position: absolute;
  bottom: -108px;
  /* 控制圆的垂直位置，让它部分显示在图片下方 */
  left: -8px;
  z-index: 0;
  /* 确保圆在图片下层 */
  border: 5px solid #042433;
  /* 添加轮廓 */
}

.highlight-Left {
  box-shadow: 0 0 20px 10px #F7AE07;
  /* 黄色阴影 */
}

.highlight-Right {
  box-shadow: 0 0 20px 10px skyblue;
  /* 蓝色阴影 */
}


.game-info {
  position: absolute;
  width: 150px;
  color: white;
  text-align: center;
}

.LeftInfo {
  position: relative;
  width: 250px;
  top: 120px;
  left: 10px;
  background-color: orange;
  /* 深蓝色背景 */
  justify-content: center;
  align-items: center;
  border: 3px solid #042433;
  /* 深蓝色边框，宽度为5px */
  border-radius: 20px;
  /* 圆角 */
  font-size: 18px;
}

.RightInfo {
  position: relative;
  width: 250px;
  top: 120px;
  right: 10px;
  background-color: #156082;
  /* 深蓝色背景 */
  justify-content: center;
  align-items: center;
  border: 3px solid #042433;
  /* 深蓝色边框，宽度为5px */
  border-radius: 20px;
  /* 圆角 */
  font-size: 18px;
}

.left {
  left: -370px;
  top: 30%;
}

.right {
  right: -270px;
  top: 30%;

}
</style>

<script setup>
import { ref, computed } from 'vue';

const size = 14;
const board = ref(Array.from({ length: size }, () => Array(size).fill(null)));

const gamePhase = ref('placing');
const isInMultiJump = ref(false);

const currentPlayer = ref(1);

const whitePieceLeft = ref(98);
const blackPieceLeft = ref(98);

const totalMoves = ref(0);

const selectedPiece = ref(null);

const piecessToRemove = ref(0);

const isWhiteFlying = computed(() => gamePhase.value === 'moving' && whitePieceLeft.value <= size);
const isBlackFlying = computed(() => gamePhase.value === 'moving' && blackPieceLeft.value <= size);

const initializeGame = () => {
  const center = Math.floor(size / 2);
  board.value[center - 1][center] = 1;
  board.value[center][center - 1] = 0;
  whitePieceLeft.value--;
  blackPieceLeft.value--;
  totalMoves.value += 2;
};

const enterMovingPhase = () => {
  currentPlayer.value = 0;
  const center = Math.floor(size / 2);
  board.value[center][center] = null;
  board.value[center - 1][center - 1] = null;
  gamePhase.value = 'moving';
  whitePieceLeft.value = 97;
  blackPieceLeft.value = 97;
};

const isSelected = (row, col) => {
  return selectedPiece.value && selectedPiece.value[0] === row && selectedPiece.value[1] === col;
};

const canMoveTo = (row, col) => {
  if (gamePhase.value !== 'moving' || !selectedPiece.value) {
    return false;
  }

  const [selectedRow, selectedCol] = selectedPiece.value;
  const opponent = currentPlayer.value === 1 ? 0 : 1;

  if ((currentPlayer.value === 1 && isWhiteFlying.value) || (currentPlayer.value === 0 && isBlackFlying.value)) {
    return board.value[row][col] === null;
  }

  const isAdjacent =
    (col === selectedCol && Math.abs(row - selectedRow) === 1)
    ||
    (row === selectedRow && Math.abs(col - selectedCol) === 1);

  const isJump =
    (row === selectedRow && Math.abs(col - selectedCol) === 2 && board.value[selectedRow][(col + selectedCol) / 2] === opponent)
    ||
    (col === selectedCol && Math.abs(row - selectedRow) === 2 && board.value[(row + selectedRow) / 2][selectedCol] === opponent);

  if (isInMultiJump.value) {
    return isJump && board.value[row][col] === null;
  }

  return (isAdjacent || isJump) && board.value[row][col] === null;
};

const handleCellClick = (row, col) => {
  if (gamePhase.value === 'placing') {
    placePiece(row, col);
  } else if (gamePhase.value === 'moving') {
    movePiece(row, col);
  } else {
    removePiece(row, col);
  }
};

const placePiece = (row, col) => {
  if (board.value[row][col] === null) {
    board.value[row][col] = currentPlayer.value;

    if (currentPlayer.value === 1) {
      whitePieceLeft.value--;
      totalMoves.value++;
      currentPlayer.value = 0;
    } else {
      blackPieceLeft.value--;
      totalMoves.value++;
      currentPlayer.value = 1;
    }

    // 检查是否已经填满所有点
    if (totalMoves.value === size * size) {
      // alert('棋盘已满, 进入走子阶段！');
      enterMovingPhase();
    }
  }
};

const movePiece = (row, col) => {
  if (board.value[row][col] === currentPlayer.value && !isInMultiJump.value) {
    selectedPiece.value = [row, col];
  }
  else if (canMoveTo(row, col)) {
    const [selectedRow, selectedCol] = selectedPiece.value;

    const isJumpMove = Math.abs(row - selectedRow) === 2 || Math.abs(col - selectedCol) === 2;

    if (isJumpMove) {
      const jumpedRow = (row + selectedRow) / 2;
      const jumpedCol = (col + selectedCol) / 2;
      board.value[jumpedRow][jumpedCol] = null;
      if (currentPlayer.value === 1) {
        blackPieceLeft.value--;
      } else {
        whitePieceLeft.value--;
      }
    }

    board.value[selectedRow][selectedCol] = null;
    board.value[row][col] = currentPlayer.value;

    if (isJumpMove && canJumpAgain(row, col)) {
      selectedPiece.value = [row, col];
      isInMultiJump.value = true;
    } else {
      selectedPiece.value = null;
      isInMultiJump.value = false;
    }

    handleSpecialForm(row, col, currentPlayer.value);

    if (gamePhase.value === 'moving') {
      currentPlayer.value = currentPlayer.value === 1 ? 0 : 1;
    }

  } else if (isInMultiJump.value && selectedPiece.value[0] === row && selectedPiece.value[1] === col) {
    isInMultiJump.value = false;
    selectedPiece.value = null;

    handleSpecialForm(row, col, currentPlayer.value);

    if (gamePhase.value === 'moving') {
      currentPlayer.value = currentPlayer.value === 1 ? 0 : 1;
    }
  }
};

const removePiece = (row, col) => {
  if (board.value[row][col] === (currentPlayer.value === 1 ? 0 : 1)) {
    board.value[row][col] = null;
    if (currentPlayer.value === 1) {
      blackPieceLeft.value--;
    } else {
      whitePieceLeft.value--;
    }
    piecessToRemove.value--;

    if (piecessToRemove.value === 0) {
      currentPlayer.value = currentPlayer.value === 1 ? 0 : 1;
      gamePhase.value = 'moving';
    }
  }
};

const canJumpAgain = (row, col) => {
  const opponent = currentPlayer.value === 1 ? 0 : 1;

  return (
    (row > 1 && board.value[row - 1][col] === opponent && board.value[row - 2][col] === null)
    ||
    (row < size - 2 && board.value[row + 1][col] === opponent && board.value[row + 2][col] === null)
    ||
    (col > 1 && board.value[row][col - 1] === opponent && board.value[row][col - 2] === null)
    ||
    (col < size - 2 && board.value[row][col + 1] === opponent && board.value[row][col + 2] === null)
  );
};

const handleSpecialForm = (row, col, player) => {
  const count = checkFourInSquare(row, col, player);

  if (count > 0) {
    gamePhase.value = 'removing';
    piecessToRemove.value = count;
  }
};

const checkFourInSquare = (row, col, player) => {
  let count = 0;

  if (row <= size - 2 && col <= size - 2) {
    if (
      board.value[row][col] === player &&
      board.value[row][col + 1] === player &&
      board.value[row + 1][col] === player &&
      board.value[row + 1][col + 1] === player
    ) {
      count++;
    }
  }

  if (row >= 1 && col >= 1) {
    if (
      board.value[row][col] === player &&
      board.value[row][col - 1] === player &&
      board.value[row - 1][col] === player &&
      board.value[row - 1][col - 1] === player
    ) {
      count++;
    }
  }

  if (row >= 1 && col <= size - 2) {
    if (
      board.value[row][col] === player &&
      board.value[row][col + 1] === player &&
      board.value[row - 1][col] === player &&
      board.value[row - 1][col + 1] === player
    ) {
      count++;
    }
  }

  if (row <= size - 2 && col >= 1) {
    if (
      board.value[row][col] === player &&
      board.value[row][col - 1] === player &&
      board.value[row + 1][col] === player &&
      board.value[row + 1][col - 1] === player
    ) {
      count++;
    }
  }

  return count;
};

initializeGame();

</script>