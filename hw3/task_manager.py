import cherrypy
import json
import datetime
import os
import time


data = {
    "0xf0b61e0bfe09": {
      "timestamps": [1664630309530, 1664630310567, 1664630311542],
      "battery_levels": [90, 90, 89],
      "power_plugged": [1, 0, 0]
    },
    "0xf0b41e2abe15": {
      "timestamps": [1664630329530, 1664630330567, 1664630331542],
      "battery_levels": [60, 59, 58],
      "power_plugged": [0, 0, 1]
    }
}



def read_battery_data():
    return data



def validate_date(date_str):
    print(date_str)
    try:
        return datetime.datetime.strptime(date_str, '%Y-%m-%d'), None
    except ValueError:
        return None, 'Invalid date format. Use YYYY-MM-DD.'

def validate_mac_address(mac):
    # Placeholder for MAC address validation logic.
    return True



class Devices(object):
    exposed = True

    def GET(self, blt=None, plugged=None):
        try:
            blt = int(blt) if blt is not None else None
            plugged = int(plugged) if plugged is not None else None
        except ValueError:
            raise cherrypy.HTTPError(400, 'Invalid query parameter value.')

        battery_data = read_battery_data()
        mac_addresses = []
        for mac, data in battery_data.items():
            if blt is not None and data['battery_levels'][-1] > blt:
                continue
            if plugged is not None and data['power_plugged'][-1] != plugged:
                continue
            mac_addresses.append(mac)

        return json.dumps({'mac_addresses': mac_addresses})

class DeviceDetail(object):
    exposed = True

    def GET(self, mac_address, start_date, end_date):
        if not validate_mac_address(mac_address):
            raise cherrypy.HTTPError(400, 'Invalid MAC address.')

        start, error = validate_date(start_date)
        if error:
            raise cherrypy.HTTPError(400, error)
        end, error = validate_date(end_date)
        if error:
            raise cherrypy.HTTPError(400, error)

        start_timestamp = time.mktime(datetime.datetime.strptime(start_date, "%Y-%m-%d").timetuple()) * 1000
        end_timestamp = time.mktime(datetime.datetime.strptime(end_date, "%Y-%m-%d").timetuple()) * 1000

        if end <= start:
            raise cherrypy.HTTPError(400, 'End date must be greater than start date.')

        battery_data = read_battery_data()
        device_data = battery_data.get(mac_address)
        if not device_data:
            raise cherrypy.HTTPError(404, 'Device not found.')

        # Filter data based on the date range
        filtered_data = {
            'mac_address': mac_address,
            'timestamps': [],
            'battery_levels': [],
            'power_plugged': []
        }
        for i, timestamp in enumerate(device_data['timestamps']):
            if start_timestamp <= timestamp <= end_timestamp:
                filtered_data['timestamps'].append(timestamp)
                filtered_data['battery_levels'].append(device_data['battery_levels'][i])
                filtered_data['power_plugged'].append(device_data['power_plugged'][i])

        return json.dumps(filtered_data)

    def DELETE(self, mac_address):
        if not validate_mac_address(mac_address):
            raise cherrypy.HTTPError(400, 'Invalid MAC address.')

        battery_data = read_battery_data()
        if mac_address not in battery_data:
            raise cherrypy.HTTPError(404, 'Device not found.')

        del battery_data[mac_address]
        
        return

if __name__ == '__main__':
    cherrypy.tree.mount(Devices(), '/devices', {'/': {'request.dispatch': cherrypy.dispatch.MethodDispatcher()}})
    cherrypy.tree.mount(DeviceDetail(), '/device', {'/': {'request.dispatch': cherrypy.dispatch.MethodDispatcher()}})
    cherrypy.config.update({'server.socket_host': '0.0.0.0', 'server.socket_port': 8080})
    cherrypy.engine.start()
    cherrypy.engine.block()
