import json
import random
import time

import paho.mqtt.client as mqtt


# ============================================================
# WAREBOT AI - MQTT TELEMETRY SIMULATOR
# ============================================================

BROKER = "test.mosquitto.org"
PORT = 1883
TOPIC = "warebot/robots/telemetry"

NUM_ROBOTS = 20


def generate_robot_telemetry(robot_number):
    """Generate one real-time robot telemetry message."""

    return {
        "robot_id": f"R{robot_number:02d}",
        "timestamp": time.time(),
        "battery": round(random.uniform(20, 100), 2),
        "motor_current": round(random.uniform(6, 16), 2),
        "temperature": round(random.uniform(35, 65), 2),
        "vibration": round(random.uniform(1, 6), 2),
        "speed": round(random.uniform(0.8, 2.2), 2),
        "operating_hours": round(random.uniform(100, 2000), 2),
        "navigation_errors": random.randint(0, 8),
    }


def on_connect(client, userdata, flags, reason_code, properties=None):
    if reason_code == 0:
        print("===================================")
        print("      WAREBOT AI MQTT SIMULATOR")
        print("===================================")
        print("MQTT Status : Connected")
        print(f"Broker      : {BROKER}")
        print(f"Topic       : {TOPIC}")
        print("-----------------------------------")
    else:
        print(f"MQTT connection failed: {reason_code}")


def main():

    client = mqtt.Client(
        mqtt.CallbackAPIVersion.VERSION2,
        client_id="warebot-telemetry-simulator"
    )

    client.on_connect = on_connect

    print("Connecting to MQTT broker...")

    try:
        client.connect(BROKER, PORT, 60)
    except Exception as e:
        print(f"Connection error: {e}")
        return

    client.loop_start()

    try:

        while True:

            for robot_number in range(1, NUM_ROBOTS + 1):

                telemetry = generate_robot_telemetry(robot_number)

                payload = json.dumps(telemetry)

                result = client.publish(
                    TOPIC,
                    payload,
                    qos=0
                )

                if result.rc == mqtt.MQTT_ERR_SUCCESS:
                    print(
                        f"Published | "
                        f"{telemetry['robot_id']} | "
                        f"Battery: {telemetry['battery']}% | "
                        f"Motor: {telemetry['motor_current']}A | "
                        f"Temp: {telemetry['temperature']}°C"
                    )
                else:
                    print(
                        f"Publish failed for {telemetry['robot_id']}"
                    )

                time.sleep(0.2)

            print("-----------------------------------")
            time.sleep(2)

    except KeyboardInterrupt:

        print("\nStopping MQTT simulator...")

    finally:

        client.loop_stop()
        client.disconnect()

        print("MQTT simulator stopped.")


if __name__ == "__main__":
    main()