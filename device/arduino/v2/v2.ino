const int PIN_HALL_A = 4;
const int PIN_HALL_B = 5;
const int PIN_LED = 13;
const int PIN_CAMERA = 7;
//const unsigned long TRIGGER_PER_10_ROUND = 1184;
const unsigned long TRIGGER_PER_10_ROUND = 1184;
const unsigned long PRINT_10_ROUND = 1;

void setup() {
  pinMode(PIN_HALL_A, INPUT);
  pinMode(PIN_HALL_B, INPUT);
  pinMode(PIN_LED, OUTPUT);
  pinMode(PIN_CAMERA, OUTPUT);
  digitalWrite(PIN_LED, LOW);
  digitalWrite(PIN_CAMERA, LOW);
  Serial.begin(115200);
}

void loop() {
  const unsigned long millis_start = millis();
  unsigned long i_round = 0;
  for (unsigned long i_trigger = 0; i_trigger < PRINT_10_ROUND * TRIGGER_PER_10_ROUND; ++i_trigger) {
    while ((digitalRead(PIN_HALL_A) ^ (i_trigger & 1)) | (digitalRead(PIN_HALL_B) ^ (i_trigger & 1)));
    if (i_trigger == i_round * TRIGGER_PER_10_ROUND / 10) {
      digitalWrite(PIN_LED, !digitalRead(PIN_LED));
      digitalWrite(PIN_CAMERA, !digitalRead(PIN_CAMERA));
      ++i_round;
    }
//    delayMicroseconds(100);
  }
  const unsigned long millis_end = millis();
  float rps = 10 * (float)PRINT_10_ROUND / (millis_end - millis_start) * 1e3 * 60;
  Serial.println(rps);
}
