import RPi.GPIO as GPIO
from time import sleep

LED_PIN = 17
PWM_FREQ = 90  # Frequência em Hz

def setup():
    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(LED_PIN, GPIO.OUT)
    global pwm
    pwm = GPIO.PWM(LED_PIN, PWM_FREQ)
    pwm.start(0)

def liga_leds(intensity=98):
    pwm.ChangeDutyCycle(intensity)  # Ajuste de 0 a 100

def desliga_leds():
    pwm.ChangeDutyCycle(0)

def cleanup():
    pwm.stop()
    GPIO.cleanup()
