const unsigned int PIN_STEPPER = 6;
const unsigned int PIN_LED = 13;
const unsigned int PIN_CAMERA = 7;
const unsigned int DELAY = 280;
// Pulse Per Ten Circles
const unsigned int PPTC = 5576;
const unsigned int SLOW_START_CIRCLE = 2;
const unsigned int SLOW_START_INIT = 8;
const unsigned int SLOW_START_ADD_PULSE = 8;
unsigned int pulse_count = 0;

void pulse(int delay) {
  digitalWrite(PIN_STEPPER, HIGH);
  delayMicroseconds(delay);
  digitalWrite(PIN_STEPPER, LOW);
  delayMicroseconds(delay);
  pulse_count += 1;
  if (10 * pulse_count % PPTC < 10) {
    digitalWrite(PIN_LED, !digitalRead(PIN_LED));
    digitalWrite(PIN_CAMERA, !digitalRead(PIN_CAMERA));
  }
  if (pulse_count == PPTC) {
    pulse_count = 0;
  }
}

void smooth_start() {
  for (unsigned int i = 0; i < SLOW_START_ADD_PULSE; ++i) {
    pulse((unsigned long)DELAY * (SLOW_START_CIRCLE * PPTC / 10 + SLOW_START_INIT) / SLOW_START_INIT);
  }
  for (unsigned int i = 0; i < SLOW_START_CIRCLE * PPTC / 10; ++i) {
    pulse((unsigned long)DELAY * (SLOW_START_CIRCLE * PPTC / 10 + SLOW_START_INIT) / (i + SLOW_START_INIT));
  }
}

void smooth_stop() {
  for (unsigned int i = SLOW_START_CIRCLE * PPTC / 10; i > 0; --i) {
    pulse((unsigned long)DELAY * (SLOW_START_CIRCLE * PPTC / 10 + SLOW_START_INIT) / (i + SLOW_START_INIT));
  }
  for (unsigned int i = 0; i < SLOW_START_ADD_PULSE; ++i) {
    pulse((unsigned long)DELAY * (SLOW_START_CIRCLE * PPTC / 10 + SLOW_START_INIT) / SLOW_START_INIT);
  }
}

void setup() {
  Serial.begin(115200);
  pinMode(PIN_STEPPER, OUTPUT);
  digitalWrite(PIN_STEPPER, LOW);
  pinMode(PIN_LED, OUTPUT);
  digitalWrite(PIN_LED, LOW);
  pinMode(PIN_CAMERA, OUTPUT);
  digitalWrite(PIN_CAMERA, LOW);
  smooth_start();
  for (unsigned long i = 0; i < (unsigned long)PPTC * 20; ++i) {
    pulse(DELAY);
  }
  smooth_stop();
}

void loop() {
}
