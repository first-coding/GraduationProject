#include <Keypad.h>

// 定义行和列的GPIO引脚
const byte ROWS = 4;  // 行数
const byte COLS = 4;  // 列数
byte rowPins[ROWS] = {1, 0, 45, 47};  // 修正为行连接到的GPIO引脚
byte colPins[COLS] = {14,46,3,38};   // 列连接到的GPIO引脚

// 定义键盘的按键映射
char keys[ROWS][COLS] = {
  {'1', '2', '3', 'A'},
  {'4', '5', '6', 'B'},
  {'7', '8', '9', 'C'},
  {'*', '0', '#', 'D'}
};

// 创建Keypad对象
Keypad keypad = Keypad(makeKeymap(keys), rowPins, colPins, ROWS, COLS);

void setup() {
  Serial.begin(9600);  // 初始化串口
  Serial.println("4x4 Keypad Ready.");
}

void loop() {
  char key = keypad.getKey();

  if (key) {
    Serial.print("Key pressed: ");
    Serial.println(key);
  }

}
