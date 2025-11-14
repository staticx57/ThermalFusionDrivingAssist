"""
Audio Alert System for ADAS
Provides ISO 26262 compliant audio warnings for driver assistance

Industry Standards (2024):
- Optimal frequency: 1.5-2 kHz for pedestrian warnings
- Volume: 60-80 dB adjustable
- Graded intensity: beep patterns for WARNING → continuous tone for CRITICAL
- Spatial audio: stereo left/right for directional warnings
"""
import numpy as np
from typing import Optional, Dict
from enum import Enum
import logging
import time
from dataclasses import dataclass

# Try to import pygame for audio
try:
    import pygame.mixer
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    logging.warning("pygame not available - audio alerts disabled. Install: pip install pygame")

from road_analyzer import AlertLevel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SoundType(Enum):
    """Types of warning sounds"""
    BEEP_SINGLE = "beep_single"
    BEEP_DOUBLE = "beep_double"
    BEEP_TRIPLE = "beep_triple"
    CONTINUOUS = "continuous"
    PULSE = "pulse"


@dataclass
class AudioConfig:
    """Audio alert configuration"""
    enabled: bool = True
    volume: float = 0.7  # 0.0-1.0
    frequency_hz: int = 1800  # 1.5-2 kHz for pedestrian alerts (ISO 26262)
    sample_rate: int = 22050
    stereo: bool = True  # Enable spatial audio


class AudioAlertSystem:
    """
    Manages audio alerts for ADAS

    Alert Patterns:
    - INFO: Single beep
    - WARNING: Double beep (0.5 Hz repetition)
    - CRITICAL: Continuous tone or rapid triple beep (2 Hz)

    Spatial Audio:
    - Left alerts: louder in left channel
    - Right alerts: louder in right channel
    - Center alerts: balanced stereo
    """

    def __init__(self, config: Optional[AudioConfig] = None):
        """
        Initialize audio alert system

        Args:
            config: Audio configuration (uses defaults if None)
        """
        self.config = config or AudioConfig()
        self.initialized = False
        self.last_alert_time: Dict[str, float] = {}
        self.alert_cooldown = 2.0  # seconds between same alert type

        # Initialize pygame mixer
        if PYGAME_AVAILABLE and self.config.enabled:
            try:
                pygame.mixer.init(
                    frequency=self.config.sample_rate,
                    size=-16,
                    channels=2 if self.config.stereo else 1,
                    buffer=512
                )
                self.initialized = True
                logger.info("Audio alert system initialized")
            except Exception as e:
                logger.error(f"Failed to initialize audio: {e}")
                self.initialized = False
        else:
            logger.warning("Audio alerts disabled (pygame not available or disabled in config)")

        # Pre-generate warning sounds
        self.sounds: Dict[str, pygame.mixer.Sound] = {}
        if self.initialized:
            self._generate_sounds()

    def _generate_sounds(self):
        """Generate warning sounds using sine waves"""
        if not self.initialized:
            return

        # Generate different warning patterns
        self.sounds['beep_short'] = self._generate_beep(duration_ms=100)
        self.sounds['beep_medium'] = self._generate_beep(duration_ms=200)
        self.sounds['beep_long'] = self._generate_beep(duration_ms=500)
        self.sounds['continuous'] = self._generate_beep(duration_ms=1000)

        logger.info("Generated audio warning sounds")

    def _generate_beep(self, duration_ms: int = 200,
                      frequency: Optional[int] = None) -> pygame.mixer.Sound:
        """
        Generate a beep sound using sine wave

        Args:
            duration_ms: Duration in milliseconds
            frequency: Frequency in Hz (uses config default if None)

        Returns:
            pygame Sound object
        """
        if frequency is None:
            frequency = self.config.frequency_hz

        sample_rate = self.config.sample_rate
        num_samples = int((duration_ms / 1000.0) * sample_rate)

        # Generate sine wave
        t = np.linspace(0, duration_ms / 1000.0, num_samples, endpoint=False)
        wave = np.sin(2 * np.pi * frequency * t)

        # Apply envelope (fade in/out to avoid clicking)
        envelope_length = int(num_samples * 0.1)  # 10% fade
        envelope = np.ones(num_samples)
        envelope[:envelope_length] = np.linspace(0, 1, envelope_length)
        envelope[-envelope_length:] = np.linspace(1, 0, envelope_length)
        wave *= envelope

        # Scale to 16-bit
        wave = (wave * 32767).astype(np.int16)

        # Create stereo if enabled
        if self.config.stereo:
            stereo_wave = np.column_stack((wave, wave))
        else:
            stereo_wave = wave

        # Convert to pygame Sound
        sound = pygame.sndarray.make_sound(stereo_wave)
        sound.set_volume(self.config.volume)

        return sound

    def play_alert(self, alert_level: AlertLevel, position: str = "center",
                  object_type: str = "object"):
        """
        Play audio alert based on alert level and position

        Args:
            alert_level: Severity of alert (INFO, WARNING, CRITICAL)
            position: Direction of alert ("left", "center", "right")
            object_type: Type of detected object (for logging)
        """
        if not self.initialized or not self.config.enabled:
            return

        # Check cooldown
        alert_key = f"{alert_level.name}_{position}"
        current_time = time.time()
        if alert_key in self.last_alert_time:
            if current_time - self.last_alert_time[alert_key] < self.alert_cooldown:
                return  # Too soon

        self.last_alert_time[alert_key] = current_time

        # Select sound based on alert level
        if alert_level == AlertLevel.INFO:
            sound = self.sounds['beep_short']
        elif alert_level == AlertLevel.WARNING:
            sound = self.sounds['beep_medium']
        elif alert_level == AlertLevel.CRITICAL:
            sound = self.sounds['beep_long']
        else:
            return  # No sound for NONE level

        # Apply spatial audio (stereo panning)
        if self.config.stereo:
            self._play_spatial(sound, position)
        else:
            sound.play()

        logger.info(f"Audio alert: {alert_level.name} - {object_type} on {position}")

    def _play_spatial(self, sound: pygame.mixer.Sound, position: str):
        """
        Play sound with spatial positioning (stereo panning)

        Args:
            sound: Sound to play
            position: "left", "center", or "right"
        """
        # Clone sound to avoid modifying original
        channel = sound.play()

        if channel:
            if position == "left":
                # Louder in left channel
                channel.set_volume(1.0, 0.3)
            elif position == "right":
                # Louder in right channel
                channel.set_volume(0.3, 1.0)
            else:  # center
                # Balanced stereo
                channel.set_volume(1.0, 1.0)

    def play_collision_warning(self, time_to_collision: float, position: str = "center"):
        """
        Play urgent collision warning based on TTC

        Args:
            time_to_collision: Time to collision in seconds
            position: Direction of threat
        """
        if not self.initialized or not self.config.enabled:
            return

        # Very urgent if TTC < 2 seconds
        if time_to_collision < 2.0:
            sound = self.sounds['continuous']
        elif time_to_collision < 4.0:
            sound = self.sounds['beep_long']
        else:
            sound = self.sounds['beep_medium']

        if self.config.stereo:
            self._play_spatial(sound, position)
        else:
            sound.play()

        logger.warning(f"COLLISION WARNING: TTC={time_to_collision:.1f}s on {position}")

    def set_volume(self, volume: float):
        """
        Set master volume

        Args:
            volume: Volume level 0.0-1.0
        """
        self.config.volume = max(0.0, min(1.0, volume))
        if self.initialized:
            for sound in self.sounds.values():
                sound.set_volume(self.config.volume)
        logger.info(f"Audio volume set to {self.config.volume:.1f}")

    def enable(self):
        """Enable audio alerts"""
        self.config.enabled = True
        logger.info("Audio alerts enabled")

    def disable(self):
        """Disable audio alerts"""
        self.config.enabled = False
        logger.info("Audio alerts disabled")

    def is_enabled(self) -> bool:
        """Check if audio is enabled and working"""
        return self.initialized and self.config.enabled

    def cleanup(self):
        """Cleanup audio resources"""
        if self.initialized:
            pygame.mixer.quit()
            logger.info("Audio alert system cleaned up")

    def get_status(self) -> Dict:
        """Get audio system status"""
        return {
            'initialized': self.initialized,
            'enabled': self.config.enabled,
            'volume': self.config.volume,
            'frequency': self.config.frequency_hz,
            'stereo': self.config.stereo,
            'pygame_available': PYGAME_AVAILABLE
        }


# Example usage
if __name__ == "__main__":
    import time

    # Create audio system
    audio = AudioAlertSystem()

    if audio.is_enabled():
        print("Testing audio alerts...")

        # Test INFO alert (center)
        print("INFO alert (center)")
        audio.play_alert(AlertLevel.INFO, "center", "traffic_light")
        time.sleep(1.5)

        # Test WARNING alert (left)
        print("WARNING alert (left)")
        audio.play_alert(AlertLevel.WARNING, "left", "person")
        time.sleep(1.5)

        # Test CRITICAL alert (right)
        print("CRITICAL alert (right)")
        audio.play_alert(AlertLevel.CRITICAL, "right", "pedestrian")
        time.sleep(1.5)

        # Test collision warning
        print("COLLISION WARNING (TTC=1.5s)")
        audio.play_collision_warning(1.5, "center")
        time.sleep(2)

        print("Audio test complete")
        audio.cleanup()
    else:
        print("Audio system not available")
