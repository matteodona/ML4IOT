import json
import psutil
import time
import uuid
import paho.mqtt.client as mqtt

# cambia con la tua matricola
topic = 's303903'

# Create a new mqtt client
client = mqtt.Client()

# Connect to the broker
client.connect('mqtt.eclipseprojects.io', 1883)

def publish_data(topic, data):
    """
    Publish data to MQTT broker
    """
    client.publish(topic, json.dumps(data))
    print("Published!")


# Initialize variables
mac_address =  hex(uuid.getnode())
events = []


try: 
    while True:
        battery = psutil.sensors_battery()

        # create event object
        event = {
            "timestamp" : int(time.time() * 1000),
            "battery_level" : battery.percent,
            "power_plugged" : int(battery.power_plugged)
        }

        #append the event to the list
        events.append(event)


        # publish every 10 consecutive events
        if len(events) == 10:
            data_to_publish = {
                "mac_address" : mac_address,
                "events" : events
            }

            #Public data
            publish_data(topic, data_to_publish)

            #Reset event list
            events = []

        # Set the sampling rate to 1 second
        time.sleep(1) 

except  KeyboardInterrupt:
    pass

