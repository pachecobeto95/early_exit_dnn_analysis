from appCloud import app
import config

"""
INITIALIZE CLOUD API
"""

# configuring Host and Port from configuration files. 
app.debug = config.DEBUG
app.run(host=config.HOST_CLOUD, port=config.PORT_CLOUD)
