import json

import paho.mqtt.client as mqtt


# ============================================================
# WAREBOT AI - MQTT TELEMETRY SUBSCRIBER
# ============================================================

BROKER = "test.mosquitto.org"
PORT = 1883
TOPIC = "warebot/robots/telemetry"


def on_connect(client, userdata, flags, reason_code, properties=None):

    if reason_code == 0:

        print("===================================")
        print("      WAREBOT AI MQTT RECEIVER")
        print("===================================")
        print("MQTT Status : Connected")
        print(f"Broker      : {BROKER}")
        print(f"Topic       : {TOPIC}")
        print("-----------------------------------")

        client.subscribe(TOPIC)

        print("Waiting for robot telemetry...\n")

    else:

        print(f"MQTT connection failed: {reason_code}")


def on_message(client, userdata, message):

    try:

        payload = message.payload.decode("utf-8")

        telemetry = json.loads(payload)

        print(
            f"Received | "
            f"{telemetry['robot_id']} | "
            f"Battery: {telemetry['battery']}% | "
            f"Motor: {telemetry['motor_current']}A | "
            f"Temp: {telemetry['temperature']}°C | "
            f"Vibration: {telemetry['vibration']}"
        )

    except Exception as e:

        print(f"Telemetry processing error: {e}")


def main():

    client = mqtt.Client(
        mqtt.CallbackAPIVersion.VERSION2,
        client_id="warebot-telemetry-receiver"
    )

    client.on_connect = on_connect
    client.on_message = on_message

    print("Connecting to MQTT broker...")

    try:

        client.connect(BROKER, PORT, 60)

    except Exception as e:

        print(f"Connection error: {e}")
        return

    try:

        client.loop_forever()

    except KeyboardInterrupt:

        print("\nStopping MQTT receiver...")

        client.disconnect()

        print("MQTT receiver stopped.")


if __name__ == "__main__":
    main()