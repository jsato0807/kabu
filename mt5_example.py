import pymt5

def onData(data):
    client = data.get('client_id')
    # Login
    if data.get('type') == '1':
        # Send heartbeat
        m.send(client, {'ver':'3','type':'6'})
        # Send login OK response
        m.send(client, {'ver':'3',
                        'type':'1',
                        'login':data.get('login'),
                        'password':data.get('password'),
                        'res':'0'})

m = pymt5.PyMT5()
onConnected = m.onConnected
onDisconnected = m.onDisconnected
m.onData = onData
