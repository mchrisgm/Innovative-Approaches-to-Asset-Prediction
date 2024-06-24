import logging

LOGGING_MODE = logging.CRITICAL


def main():
    logging.basicConfig(level=LOGGING_MODE, format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s')
    logging.info('Hello, World!')


if __name__ == '__main__':
    main()
