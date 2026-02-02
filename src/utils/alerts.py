import os
import threading
import time
import src.config as config

class AudioAlert:
    def __init__(self):
        self.last_alert_time = 0
        self.cooldown = 3.0  # Wait 3 seconds between beeps

    def trigger(self, alert_type="warning"):
        """
        Plays a sound in a background thread so video doesn't freeze.
        """
        if not config.ENABLE_AUDIO:
            return

        current_time = time.time()
        
        # Prevent sound from spamming (playing too fast)
        if current_time - self.last_alert_time > self.cooldown:
            self.last_alert_time = current_time
            
            # Run sound in separate thread
            t = threading.Thread(target=self._play_sound, args=(alert_type,))
            t.daemon = True
            t.start()

    def _play_sound(self, alert_type):
        try:
            # Mac-specific system sounds
            if alert_type == "danger":
                # Louder/More urgent sound
                os.system('afplay /System/Library/Sounds/Sosumi.aiff')
            else:
                # Softer warning sound
                os.system('afplay /System/Library/Sounds/Tink.aiff')
        except:
            # Fallback for non-Mac (Standard System Beep)
            print('\a')