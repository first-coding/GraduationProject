#include <Keypad.h>
#include <Arduino.h>

// 定义行和列的GPIO引脚
const byte ROWS = 4;
const byte COLS = 4;
byte rowPins[ROWS] = {1, 0, 45, 47};    // 行连接的引脚
byte colPins[COLS] = {21, 46, 3, 38};   // 列连接的引脚

// 定义按键布局
char keys[ROWS][COLS] = {
  {'1', '2', '3', 'A'},
  {'4', '5', '6', 'B'},
  {'7', '8', '9', 'C'},
  {'*', '0', '#', 'D'}
};

// 创建 Keypad 对象
Keypad keypad = Keypad(makeKeymap(keys), rowPins, colPins, ROWS, COLS);
bool videoUnlockFlag = false;
// 密码设置
String correctPassword = "223A";
String inputPassword = "";

// 输出引脚定义
const int outputPin = 14;

void setup() {
  Serial.begin(115200);
  Serial.println("Keypad system ready...");

  pinMode(outputPin, OUTPUT);
  digitalWrite(outputPin, LOW);  // 初始设置为低电平
}

void loop() {
  char key = keypad.getKey();

  if (key) {
    Serial.print("Key pressed: ");
    Serial.println(key);

    if (key == '#') {
      if (inputPassword == correctPassword) {
        Serial.println("✅ Password correct! Triggering output...");
        triggerOutput();
      } else {
        Serial.println("❌ Password incorrect!");
      }
      inputPassword = "";  // 清空输入
    } 
    else if (key == 'D') {
      inputPassword = "";
      Serial.println("🔄 Input cleared.");
    } 
    else {
      inputPassword += key;
    }

    Serial.print("Current input: ");
    Serial.println(inputPassword);
  }
}

// 设置 GPIO43 高电平3秒后恢复低电平
void triggerOutput() {
  digitalWrite(outputPin, HIGH);
  delay(3000);
  digitalWrite(outputPin, LOW);
}
