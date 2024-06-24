import logging


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s')
    logging.info('Hello, World!')


if __name__ == '__main__':
    main()
