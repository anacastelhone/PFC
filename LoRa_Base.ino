/* BASE: recebe o GPS do robo por LoRa e repassa pela USB (serial 115200) para o base_inspecao.py.
 *
 * Na serial sai UMA linha por pacote valido, no formato que o base_inspecao le:
 *   LAT:-19.747230,LON:-47.938110,SAT:8,SPD:1.20,RSSI:-87
 * Mensagens de diagnostico comecam com "#" (o base_inspecao ignora linhas sem "LAT:").
 * A configuracao do radio precisa ser IGUAL a do GPS no LoRa_Robo (915 MHz, BW 125 kHz, SF7).
 */
#include <Arduino.h>
#include <Wire.h>
#include "HT_SSD1306Wire.h"
#include "LoRaWan_APP.h"

// ================= LORA (igual ao GPS_* do LoRa_Robo) =================
#define RF_FREQUENCY          915000000
#define LORA_BANDWIDTH        0          // 125 kHz
#define LORA_SPREADING_FACTOR 7
#define LORA_CODINGRATE       1
#define LORA_PREAMBLE_LENGTH  8
#define LORA_SYMBOL_TIMEOUT   0
#define LORA_FIX_LENGTH_PAYLOAD_ON false
#define LORA_IQ_INVERSION_ON  false

#define SEM_PACOTE_MS 5000   // avisa no OLED se o robo ficar mudo

// ================= OLED =================
static SSD1306Wire display(0x3c, 500000, SDA_OLED, SCL_OLED, GEOMETRY_128_64, RST_OLED);

#define BUFFER_RX 128
char rxpacket[BUFFER_RX];
static RadioEvents_t RadioEvents;
bool lora_idle = true;
unsigned long ultimoPacote = 0;
bool avisouMudo = false;

// ================= OLED =================
void mostrar(const String &lat, const String &lon, const String &sats, const String &spd, int16_t rssi) {
  display.clear();
  display.setTextAlignment(TEXT_ALIGN_LEFT);
  display.setFont(ArialMT_Plain_10);
  display.drawString(0, 0, (lat.toFloat() == 0.0 && lon.toFloat() == 0.0) ? "GPS ROBO: SEM FIX" : "GPS ROBO");
  display.drawString(0, 13, "LAT: " + lat);
  display.drawString(0, 26, "LON: " + lon);
  display.drawString(0, 39, "SAT: " + sats);
  display.drawString(64, 39, "SPD: " + spd);
  display.drawString(0, 52, "RSSI: " + String(rssi));
  display.display();
}

// pega o valor entre "chave" e a proxima virgula (ou o fim)
String campo(const String &s, const char *chave) {
  int i = s.indexOf(chave);
  if (i < 0) return "";
  i += strlen(chave);
  int j = s.indexOf(',', i);
  return j < 0 ? s.substring(i) : s.substring(i, j);
}

// ================= CALLBACKS =================
void OnRxDone(uint8_t *payload, uint16_t size, int16_t rssi, int8_t snr) {
  if (size >= BUFFER_RX) size = BUFFER_RX - 1;   // evita estourar o buffer
  memcpy(rxpacket, payload, size);
  rxpacket[size] = '\0';
  Radio.Sleep();
  lora_idle = true;

  String data = String(rxpacket);
  if (!data.startsWith("LAT:") || data.indexOf(",LON:") < 0 || data.indexOf(",SAT:") < 0) {
    Serial.println("# pacote ignorado: " + data);   // nao e do robo (ou corrompido)
    return;
  }
  ultimoPacote = millis();
  avisouMudo = false;

  Serial.println(data + ",RSSI:" + String(rssi));   // linha lida pelo base_inspecao.py
  mostrar(campo(data, "LAT:"), campo(data, "LON:"), campo(data, "SAT:"), campo(data, "SPD:"), rssi);
}

void OnRxError(void)   { Radio.Sleep(); lora_idle = true; }
void OnRxTimeout(void) { Radio.Sleep(); lora_idle = true; }

// ================= SETUP =================
void setup() {
  Serial.begin(115200);
  Mcu.begin(HELTEC_BOARD, SLOW_CLK_TPYE);

  pinMode(Vext, OUTPUT);
  digitalWrite(Vext, LOW);
  delay(100);
  display.init();
  display.clear();
  display.setTextAlignment(TEXT_ALIGN_CENTER);
  display.setFont(ArialMT_Plain_16);
  display.drawString(64, 10, "LoRa Base");
  display.setFont(ArialMT_Plain_24);
  display.drawString(64, 30, "RX");
  display.display();

  RadioEvents.RxDone = OnRxDone;
  RadioEvents.RxError = OnRxError;
  RadioEvents.RxTimeout = OnRxTimeout;
  Radio.Init(&RadioEvents);
  Radio.SetChannel(RF_FREQUENCY);
  Radio.SetRxConfig(MODEM_LORA, LORA_BANDWIDTH, LORA_SPREADING_FACTOR, LORA_CODINGRATE, 0,
                    LORA_PREAMBLE_LENGTH, LORA_SYMBOL_TIMEOUT, LORA_FIX_LENGTH_PAYLOAD_ON,
                    0, true, 0, 0, LORA_IQ_INVERSION_ON, true);
  Serial.println("# base pronta, aguardando GPS do robo");
  ultimoPacote = millis();
}

// ================= LOOP =================
void loop() {
  if (lora_idle) {
    lora_idle = false;
    Radio.Rx(0);
  }
  if (!avisouMudo && millis() - ultimoPacote > SEM_PACOTE_MS) {
    avisouMudo = true;
    Serial.println("# nenhum pacote do robo ha mais de 5 s");
    display.clear();
    display.setTextAlignment(TEXT_ALIGN_CENTER);
    display.setFont(ArialMT_Plain_10);
    display.drawString(64, 25, "SEM SINAL DO ROBO");
    display.display();
  }
  Radio.IrqProcess();
}
