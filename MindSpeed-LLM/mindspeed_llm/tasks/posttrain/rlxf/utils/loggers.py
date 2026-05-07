# Copyright (c) 2024, HUAWEI CORPORATION.  All rights reserved.

import logging
from datetime import datetime


class Loggers(object):
    def __init__(self,
                 name='root',
                 logger_level='DEBUG',
                 ):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logger_level)

    def handle_msg(self, msg, level, iteration, steps):
        current_time = str(datetime.now()).split(".")[0]

        if isinstance(msg, dict):
            fmt_msg = f"[{current_time}] "
            fmt_msg += f"iteration: {iteration} / {steps} | "
            for key in msg:
                fmt_msg += f"{key} : {format(msg[key], '.4f')} | "
            fmt_msg = fmt_msg[:-2]
        else:
            fmt_msg = f"[{current_time}] {level} " + str(msg)
        return fmt_msg

    def info(self, msg, iteration, steps):
        format_msg = self.handle_msg(msg, "INFO", iteration, steps)
        self.logger.info(format_msg)

    def warning(self, msg, iteration, steps):
        format_msg = self.handle_msg(msg, "WARNING", iteration, steps)
        self.logger.warning(format_msg)

    def debug(self, msg, iteration, steps):
        format_msg = self.handle_msg(msg, "DEBUG", iteration, steps)
        self.logger.debug(format_msg)

    def error(self, msg, iteration, steps):
        format_msg = self.handle_msg(msg, "ERROR", iteration, steps)
        self.logger.error(format_msg)

    def flush(self):
        for handler in self.logger.handlers:
            handler.flush()