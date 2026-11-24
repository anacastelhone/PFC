/* ROBO: recebe os comandos do Controle2 (como o LoRa_Garca) e envia o GPS para a LoRa_Base.
 *
 * Um radio so faz uma coisa por vez, entao o robo fica ESCUTANDO o controle e, a cada
 * GPS_PERIODO_MS, interrompe a escuta, troca para o canal do GPS, envia a posicao e volta.
 * O controle reenvia o comando a cada ~100 ms, entao perder um pacote durante o envio
 * do GPS (~60 ms) nao atrapalha.
 *
 *   Controle2 --(868 MHz, BW 250 kHz)--> ROBO --(915 MHz, BW 125 kHz)--> LoRa_Base --USB--> base_inspecao.py
 *
 * Pacote de GPS: "LAT:-19.747230,LON:-47.938110,SAT:8,SPD:1.20"
 * Sem fix, envia LAT:0.000000,LON:0.000000 (o base_inspecao trata como "sem fix").
 */
#include <Arduino.h>
#include <Wire.h>
#include "HT_SSD1306Wire.h"
#include "HT_TinyGPS++.h"
#include "LoRaWan_APP.h"
#include <HardwareSerial.h>

// ================= RADIO: COMANDOS (igual ao Controle2 / LoRa_Garca) =================
#define CMD_FREQUENCY        868000000
#define CMD_BANDWIDTH        1          // 250 kHz
#define CMD_SPREADING_FACTOR 7
#define CMD_CODINGRATE       1
// ================= RADIO: GPS (igual a LoRa_Base) =================
#define GPS_FREQUENCY        915000000
#define GPS_BANDWIDTH        0          // 125 kHz
#define GPS_SPREADING_FACTOR 7
#define GPS_CODINGRATE       1
#define GPS_TX_POWER         14         // dBm

#define LORA_PREAMBLE_LENGTH 8
#define LORA_SYMBOL_TIMEOUT  0
#define LORA_FIX_LENGTH_PAYLOAD_ON false
#define LORA_IQ_INVERSION_ON false

#define GPS_PERIODO_MS   1000   // intervalo de envio do GPS
#define FAILSAFE_MS      1000   // sem comando do controle por esse tempo -> para tudo
#define GPS_IDADE_MAX_MS 2000   // posicao mais velha que isso = sem fix

// ================= OLED =================
static SSD1306Wire display(0x3c, 500000, SDA_OLED, SCL_OLED, GEOMETRY_128_64, RST_OLED);

// ================= GPS =================
TinyGPSPlus gps;
HardwareSerial gpsSerial(1);
#define RXD2 6
#define TXD2 5
#define GPS_BAUD 9600

// ================= MOTORES (mesmos pinos do LoRa_Garca) =================
#define ATUADOR_AVANCO 46
#define ATUADOR_RECUO  45
#define MOTOR_AVANCO   41
#define MOTOR_RECUO    42

// ================= ESTADO =================
#define BUFFER_RX 32
char rxpacket[BUFFER_RX];
char txpacket[96];

static RadioEvents_t RadioEvents;

enum EstadoRadio { OCIOSO, ESCUTANDO, TRANSMITINDO };
volatile EstadoRadio estado = OCIOSO;

unsigned long ultimoGPS = 0;
unsigned long ultimoComando = 0;
bool motoresParados = true;
int PWM = 0;
int16_t rssiCmd = 0;

// ================= MOTORES =================
void pararTudo() {
  analogWrite(MOTOR_AVANCO, 0);
  analogWrite(MOTOR_RECUO, 0);
  analogWrite(ATUADOR_AVANCO, 0);
  analogWrite(ATUADOR_RECUO, 0);
  motoresParados = true;
}

// Mesma interpretacao do LoRa_Garca:
//   |PWM| <= 255  -> motores (joystick X)
//   |PWM|  > 256  -> atuador, PWM/1000 (joystick Y, o controle multiplica por 1000)
void aplicarPWM(int valor) {
  if (valor == 0) {
    pararTudo();
    return;
  }
  motoresParados = false;
  if (valor < -256 || valor > 256) {
    int p = constrain(abs(valor / 1000), 0, 255);
    analogWrite(MOTOR_AVANCO, 0);
    analogWrite(MOTOR_RECUO, 0);
    analogWrite(ATUADOR_AVANCO, valor > 0 ? p : 0);
    analogWrite(ATUADOR_RECUO,  valor > 0 ? 0 : p);
  } else {
    int p = constrain(abs(valor), 0, 255);
    analogWrite(ATUADOR_AVANCO, 0);
    analogWrite(ATUADOR_RECUO, 0);
    analogWrite(MOTOR_AVANCO, valor > 0 ? p : 0);
    analogWrite(MOTOR_RECUO,  valor > 0 ? 0 : p);
  }
}

// ================= RADIO =================
// O driver do SX1262 aplica a configuracao na hora, entao a de RX precisa ser refeita
// depois de cada TX (e vice-versa).
void escutarControle() {
  Radio.Sleep();
  Radio.SetChannel(CMD_FREQUENCY);
  Radio.SetRxConfig(MODEM_LORA, CMD_BANDWIDTH, CMD_SPREADING_FACTOR, CMD_CODINGRATE, 0,
                    LORA_PREAMBLE_LENGTH, LORA_SYMBOL_TIMEOUT, LORA_FIX_LENGTH_PAYLOAD_ON,
                    0, true, 0, 0, LORA_IQ_INVERSION_ON, true);
  estado = ESCUTANDO;
  Radio.Rx(0);
}

void enviarGPS() {
  bool fix = gps.location.isValid() && gps.location.age() < GPS_IDADE_MAX_MS;
  double lat = fix ? gps.location.lat() : 0.0;
  double lon = fix ? gps.location.lng() : 0.0;
  int sats = gps.satellites.isValid() ? (int)gps.satellites.value() : 0;
  double spd = gps.speed.isValid() ? gps.speed.kmph() : 0.0;

  snprintf(txpacket, sizeof(txpacket), "LAT:%.6f,LON:%.6f,SAT:%d,SPD:%.2f", lat, lon, sats, spd);
  Serial.println(txpacket);

  display.clear();
  display.setTextAlignment(TEXT_ALIGN_LEFT);
  display.setFont(ArialMT_Plain_10);
  display.drawString(0, 0, fix ? "GPS OK" : "GPS SEM FIX");
  display.drawString(0, 13, "LAT: " + String(lat, 6));
  display.drawString(0, 26, "LON: " + String(lon, 6));
  display.drawString(0, 39, "SAT: " + String(sats));
  display.drawString(0, 52, "PWM: " + String(PWM) + "  RSSI: " + String(rssiCmd));
  display.display();

  Radio.Sleep();                        // interrompe a escuta do controle
  Radio.SetChannel(GPS_FREQUENCY);
  Radio.SetTxConfig(MODEM_LORA, GPS_TX_POWER, 0, GPS_BANDWIDTH, GPS_SPREADING_FACTOR,
                    GPS_CODINGRATE, LORA_PREAMBLE_LENGTH, LORA_FIX_LENGTH_PAYLOAD_ON,
                    true, 0, 0, LORA_IQ_INVERSION_ON, 3000);
  estado = TRANSMITINDO;
  Radio.Send((uint8_t *)txpacket, strlen(txpacket));
}

// ================= CALLBACKS (chamados dentro de Radio.IrqProcess) =================
void OnRxDone(uint8_t *payload, uint16_t size, int16_t rssi, int8_t snr) {
  if (size >= BUFFER_RX) size = BUFFER_RX - 1;   // evita estourar o buffer
  memcpy(rxpacket, payload, size);
  rxpacket[size] = '\0';
  rssiCmd = rssi;
  ultimoComando = millis();

  // A, B, C, D: botoes do controle (sem acao por enquanto, como no LoRa_Garca)
  if (strcmp(rxpacket, "A") && strcmp(rxpacket, "B") && strcmp(rxpacket, "C") && strcmp(rxpacket, "D")) {
    PWM = String(rxpacket).toInt();
    aplicarPWM(PWM);
  }
  estado = OCIOSO;   // o loop volta a escutar
}

void OnRxError(void)   { estado = OCIOSO; }
void OnRxTimeout(void) { estado = OCIOSO; }
void OnTxDone(void)    { estado = OCIOSO; }
void OnTxTimeout(void) { Serial.println("TX GPS timeout"); estado = OCIOSO; }

// ================= SETUP =================
void setup() {
  Serial.begin(115200);
  Mcu.begin(HELTEC_BOARD, SLOW_CLK_TPYE);

  // motores parados ANTES de qualquer outra coisa
  pinMode(MOTOR_AVANCO, OUTPUT);
  pinMode(MOTOR_RECUO, OUTPUT);
  pinMode(ATUADOR_AVANCO, OUTPUT);
  pinMode(ATUADOR_RECUO, OUTPUT);
  pararTudo();

  // OLED (precisa do Vext ligado)
  pinMode(Vext, OUTPUT);
  digitalWrite(Vext, LOW);
  delay(100);
  display.init();
  display.clear();
  display.setTextAlignment(TEXT_ALIGN_CENTER);
  display.setFont(ArialMT_Plain_16);
  display.drawString(64, 10, "LoRa Robo");
  display.setFont(ArialMT_Plain_10);
  display.drawString(64, 35, "RX controle / TX GPS");
  display.display();

  gpsSerial.begin(GPS_BAUD, SERIAL_8N1, RXD2, TXD2);
  Serial.println("GPS iniciado");

  RadioEvents.RxDone = OnRxDone;
  RadioEvents.RxError = OnRxError;
  RadioEvents.RxTimeout = OnRxTimeout;
  RadioEvents.TxDone = OnTxDone;
  RadioEvents.TxTimeout = OnTxTimeout;
  Radio.Init(&RadioEvents);

  ultimoComando = millis();
  escutarControle();
}

// ================= LOOP =================
void loop() {
  while (gpsSerial.available()) {
    gps.encode(gpsSerial.read());
  }

  // FAILSAFE: o controle manda pacote o tempo todo (ate "0" parado). Se parar de chegar
  // (controle desligado, fora de alcance), para os motores.
  if (!motoresParados && millis() - ultimoComando > FAILSAFE_MS) {
    Serial.println("FAILSAFE: sem comando, motores parados");
    PWM = 0;
    pararTudo();
  }

  if (estado != TRANSMITINDO && millis() - ultimoGPS >= GPS_PERIODO_MS) {
    ultimoGPS = millis();
    enviarGPS();
  } else if (estado == OCIOSO) {
    escutarControle();
  }

  Radio.IrqProcess();
}
