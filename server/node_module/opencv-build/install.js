if (process.env.npm_config_loglevel === 'silly') {
  log.level = 'silly'
}

const { install } = require('./build/install')

install()
