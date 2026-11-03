SHALE on ESP32 (Arduino IDE)

1. Arduino IDE -> Tools -> Board -> Boards Manager: install "esp32 by Espressif Systems" (if not already).
2. Unzip; keep the folder name "shale_esp32" (it must match the .ino file name).
   Open shale_esp32/shale_esp32.ino. The other files (codec.c, ascon.c, clip.h) open as tabs automatically.
3. Tools -> Board: select your ESP32 board (e.g. "ESP32 Dev Module"). Select the port.
4. Upload. Then open Tools -> Serial Monitor, set 115200 baud, press the board's EN/RESET button once.
5. After about 10 seconds the benchmark prints a block ending with "===== done =====".
   Copy the WHOLE block and send it back.

What it does: encrypts a real 2.3 s speech clip (stored in flash) with the full SHALE pipeline
(lossless coder + ECLQ padding + Ascon-AEAD128) in 20 ms and 100 ms records, decrypts it on the board,
checks that every sample is bit-exact, and measures time per record, CPU load and bytes sent,
against Ascon on raw PCM.

Optional (power): if you have a USB power meter, note the current/power while the benchmark runs
and when the board is idle after "done".
