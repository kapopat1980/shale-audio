SHALE on ESP8266 (NodeMCU 1.0, ESP-12E) - the sketch used for Table 13 of the paper.

Port of ../shale_esp32: identical Ascon code; codec.c differs only in allocating one channel
buffer instead of four for mono input (same bitstream); the speech clip is stored in flash (PROGMEM).
Only the 20 ms record mode is run (the 100 ms mode needs more RAM than an ESP8266 has).

Arduino IDE: install "esp8266 by ESP8266 Community", select "NodeMCU 1.0 (ESP-12E Module)",
CPU frequency 160 MHz, upload, open Serial Monitor at 115200 baud.
