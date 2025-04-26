
#include <C:\Users\wangdahai\AppData\Local\Arduino15\packages\esp32\hardware\esp32\3.1.3\libraries\BluetoothSerial\src\BluetoothSerial.h>
#define LOCK_NC_ADC 1  // NC 连接*的 ADC1_CH0
BluetoothSerial SerialBT;  // 创建蓝牙对象
String device_name = "ESP32-BT-Slave";
void setup() {
    Serial.begin(115200);
    SerialBT.begin(device_name);  // 设置蓝牙名称
    analogReadResolution(12); // 12位ADC，范围0-4095
    Serial.println("蓝牙已启动，等待连接...");
}

void loop() {
    int adcValue = analogRead(LOCK_NC_ADC);  // 读取 NC 端的 ADC 值
    float voltage = adcValue * (3.3 / 4095.0);  // 转换为电压

    // 串口调试输出
    Serial.print("ADC值: ");
    Serial.print(adcValue);
    Serial.print("， 电压: ");
    Serial.print(voltage);
    Serial.println("V");

    String lockStatus;
    if (adcValue == 0) {
        lockStatus = "🔒 电磁锁：锁定（NC 和 COM 闭合）";
    } else {
        lockStatus = "🔓 电磁锁：解锁（NC 和 COM 断开）";
    }

    // 串口输出状态
    Serial.println(lockStatus);
    if (Serial.available()) {
        SerialBT.write(Serial.read());
    }
    if (SerialBT.available()) {
        Serial.write(SerialBT.read());
    }
    delay(500);
}
