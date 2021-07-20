from appEdge import app
import config

"""
INITIALIZE EDGE API
"""

# configuring Host and Port from configuration files. 
app.debug = config.DEBUG
app.run(host=config.HOST_EDGE, port=config.PORT_EDGE)
